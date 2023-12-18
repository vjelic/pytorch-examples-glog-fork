#!/bin/bash
# MI300 Envs
#export HSA_DISABLE_CACHE=1

# F8 specific envs
export F8_CONFIRM=0
export ENABLE_F8_GEMM=0
export ENABLE_F8_CONV=0
export F8_SIM=0


# Limit GPUs
export HIP_VISIBLE_DEVICES=0,1,2,3


# rocBLAS output
##export ROCBLAS_LAYER=2

# rocBLAS numerical checking
#export ROCBLAS_CHECK_NUMERICS=2
#export ROCBLAS_LAYER=7

#MIOpen numerical checking
#export MIOPEN_CHECK_NUMERICS=9
#export MIOPEN_CHECK_NUMERICS=1
#export MIOPEN_DUMP_TENSOR_PATH=/home/examples/imagenet/abnorm/abnorm_tens
##export MIOPEN_DISABLE_CACHE=1

#MIOpen logging
##export MIOPEN_ENABLE_LOGGING=1
###export MIOPEN_ENABLE_LOGGING_MPMT=0
##export MIOPEN_ENABLE_LOGGING_CMD=1
##export MIOPEN_LOG_LEVEL=7


python3 main.py -a resnet50 --mom 0.9 --lr 9.1 --wd 0.0002 --epochs 1 --batch-size 256 -j 32 -p 100 /root/imagenet

# with hooks to print tensors
#python3 main.py --hooks -a resnet50 --print-freq 1 --epochs 1 --gpu 0 /home/imagenet

# single GPU
#python3 main.py -a resnet50 --print-freq 1 --epochs 1 --batch-size 4 --gpu 0 /home/imagenet



#python3 main.py -a resnet50 --warmup 5 --steps 50 --epochs 1 --print-freq 1 --lr 0.1 /root/imagenet


# multi GPU sim run
#python3 main.py -a resnet50 --print-freq 1 --lr 0.01 --epochs 64 /home/imagenet



echo "Done!"
