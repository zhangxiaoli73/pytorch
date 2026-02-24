#!/bin/bash

LLAMA_8B_M=(2048 2048)
LLAMA_8B_N=(1536 7168)
LLAMA_8B_K=(4096 4096)

len=${#LLAMA_8B_M[@]}

for ((i=0; i<$len; i++)); do
    M=${LLAMA_8B_M[$i]}
    N=${LLAMA_8B_N[$i]}
    K=${LLAMA_8B_K[$i]}

    echo "Running fused_allgather_with_matmul with M=$M, N=$N, K=$K"
    mpirun -np 4 --prepend-rank python test_fused_allgather_matmul.py $M $N $K
    sleep 1
done


LLAMA_8B_M_1=(8192 8192)
LLAMA_8B_N_1=(4096 4096)
LLAMA_8B_K_1=(1024 3584)


for ((i=0; i<$len; i++)); do
    M=${LLAMA_8B_M_1[$i]}
    N=${LLAMA_8B_N_1[$i]}
    K=${LLAMA_8B_K_1[$i]}

    echo "Running fused_matmul_with_reducescatter with M=$M, N=$N, K=$K"
    mpirun -np 4 --prepend-rank python test_fused_matmul_reducescatter.py $M $N $K
    sleep 1
done



