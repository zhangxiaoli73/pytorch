import itertools
import os
from contextlib import nullcontext
from unittest import skip, skipIf

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
    _fused_all_gather_scaled_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    _test_mode,
    enable_symm_mem_for_group,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)
import os
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(description="test_symm")
parser.add_argument("M", type=int, default=4096, help="M value")
parser.add_argument("N", type=int, default=1792, help="N value")
parser.add_argument("K", type=int, default=4096, help="K value")

args = parser.parse_args()

print("M = ", args.M)
print("N = ", args.N)
print("K = ", args.K)

BATCH = 1
M = args.M # 512 #4096
N = args.N # 128 #1792
K = args.K # 128 #4096
Loop = 20
enable_profile = False

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29513'
dist.init_process_group(backend='xccl')

def test_allgather_matmul(rank, world_size):
    torch.xpu.set_device(rank)
    torch.use_deterministic_algorithms(True, warn_only=True)

    group = dist.group.WORLD

    torch.manual_seed(42 + rank)
    A_shard = torch.rand(M, K, device="xpu", dtype=torch.bfloat16)
    Bs = [torch.rand(K, N, device="xpu", dtype=torch.bfloat16) for _ in range(1)]

    begin_events_ref = [
        torch.xpu.Event(enable_timing=True) for _ in range(Loop-5)
    ]
    end_events_ref = [torch.xpu.Event(enable_timing=True) for _ in range(Loop-5)]

    begin_events = [
        torch.xpu.Event(enable_timing=True) for _ in range(Loop-5)
    ]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(Loop-5)]

    if enable_profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    with prof:
        # warm up fallback aten ops
        for i in range(10):
            ag_output_0, mm_outputs_0 = _fused_all_gather_matmul_fallback(
                A_shard, Bs, gather_dim=0, group_name=group.group_name
            )
        torch.xpu.synchronize()
        for i in range(Loop):
            if i >= 5:
                begin_events_ref[i-5].record()
            ag_output_0, mm_outputs_0 = _fused_all_gather_matmul_fallback(
                A_shard, Bs, gather_dim=0, group_name=group.group_name
            )
            if i >= 5:
                end_events_ref[i-5].record()
        torch.xpu.synchronize()

        # warm up symmetric ops
        for i in range(10):
            ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_matmul(
                A_shard, Bs, gather_dim=0, group_name=group.group_name
            )
        torch.xpu.synchronize()
        print(f"zl_debug start to call symm memory ", flush=True)
        for i in range(Loop):
            if i >= 5:
                begin_events[i-5].record()
            ag_output_1, mm_outputs_1 = torch.ops.symm_mem.fused_all_gather_matmul(
                A_shard, Bs, gather_dim=0, group_name=group.group_name
            )
            if i >= 5:
                end_events[i-5].record()
        torch.xpu.synchronize()

    latencies_ref = [b.elapsed_time(e) for b, e in zip(begin_events_ref, end_events_ref)]
    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]

    if enable_profile:
        prof.export_chrome_trace("./profile_kineto_trace_" + str(rank) + ".json")
    
    ''' 
    assert torch.allclose(ag_output_0, ag_output_1)
    assert ag_output_0.stride() == ag_output_1.stride()
    for mm_output_0, mm_output_1 in zip(mm_outputs_0, mm_outputs_1):
        assert torch.allclose(mm_output_0, mm_output_1)
        assert mm_output_0.stride(), mm_output_1.stride()
    '''
    dist.destroy_process_group()
    print(f"[Fallback time in rank {rank}]: average time = {sum(latencies_ref) / len(latencies_ref)} detail lists = {latencies_ref} ms")
    print(f"[Symm ops time in rank {rank}]: average time = {sum(latencies) / len(latencies)} detail lists =  {latencies} ms")


rank = dist.get_rank()
size = dist.get_world_size()
test_allgather_matmul(rank, size)


