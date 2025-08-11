
import os
from contextlib import nullcontext
from unittest import skip, skipIf

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import (
    _fused_all_gather_matmul_fallback,
    _fused_all_gather_scaled_matmul_fallback,
    _fused_matmul_reduce_scatter_fallback,
    _fused_all_gather_matmul_reducescatter_impl,
    enable_symm_mem_for_group,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29803'
dist.init_process_group(backend='xccl')
group_name = dist.group.WORLD.group_name

enable_profile=False

def matmul_shard_consumer(in_shard: torch.Tensor, Bs: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(torch.matmul(in_shard, Bs))

def test_pipeline(rank: int, world_size: int):
    torch.manual_seed(1234 + rank)
    torch.xpu.set_device(rank)

    if enable_profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()
    rows_per_rank = 2
    K = 4
    N = 3

    # 每个 rank 的 A_shard
    A_shard = torch.randn(rows_per_rank, K, device="xpu")
    # 全量 B（在每个 rank 上相同）
    B = torch.randn(K, N, device="xpu")
    print(f"zl_debug A = {A_shard} B = {B}", flush=True)

    # === 校验 A_full ===
    gathered = [torch.zeros_like(A_shard) for _ in range(world_size)]
    dist.all_gather(gathered, A_shard)
    expected_A_full = torch.cat(gathered, dim=0)

    # === 校验 final_outputs ===
    expected_product = expected_A_full @ B  # [rows_total, N]
    # 将 expected_product 按行切成 world_size 份
    scatter = torch.randn(rows_per_rank, N, device="xpu")
    dist.reduce_scatter_tensor(scatter, expected_product)
    torch.xpu.synchronize()

    kwargs_list = [{}]  # 对应 Bs 中每个矩阵的 kwargs（这里只测一个 B）

    # 调用 fused API
    with prof:
        for count in range(10):
            final_outputs = _fused_all_gather_matmul_reducescatter_impl(
                torch.ops.aten.mm.out,
                A_shard=A_shard,
                Bs=B,
                kwargs_list=kwargs_list,
                group_name=group_name,  # 默认进程组
            )
        torch.xpu.synchronize()
    if enable_profile:
         prof.export_chrome_trace("./profile_kineto_trace_" + str(rank) + ".json")

    print(f"DONE!!!!!!!!!!!!! {final_outputs.shape}", flush=True)
    # print(f"fused kernel {final_outputs} fallback = {scatter}")
    assert torch.allclose(scatter, final_outputs)
    dist.destroy_process_group()

rank = dist.get_rank()
world_size = dist.get_world_size()
test_pipeline(rank, world_size)

