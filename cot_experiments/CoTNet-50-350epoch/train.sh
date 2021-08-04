python3 -m torch.distributed.launch --nproc_per_node=8 train.py --folder ./cot_experiments/CoTNet-50-350epoch

# CUDA_VISIBLE_DEVICES=<GPU_0, GPU_1,...> NCCL_SOCKET_IFNAME=eth0 NCCL_NSOCKS_PERTHREAD=8 NCCL_SOCKET_NTHREADS=8 python3 -m torch.distributed.launch \
#    --nproc_per_node=<#GPU> \
#    --nnodes=<#node> \
#    --node_rank=0 \
#    --master_addr="<master node ip>" \
#    --master_port=12346 \
#    train.py --folder ./cot_experiments/CoTNet-50-350epoch
