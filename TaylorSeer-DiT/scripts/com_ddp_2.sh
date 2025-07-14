
CUDA_VISIBLE_DEVICES=5 torchrun --nnodes=1 --nproc_per_node=1  --master_port=29504 sample_ddp.py \
  --model DiT-XL/2 \
  --per-proc-batch-size 50 \
  --image-size 256 \
  --cfg-scale 1.5 \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 1 \
  --max-order 1 \
  --max_block_order 1 \
  --threshold 0.014 \
  --mid_cor 2 \



CUDA_VISIBLE_DEVICES=5 torchrun --nnodes=1 --nproc_per_node=1  --master_port=29504 sample_ddp.py \
  --model DiT-XL/2 \
  --per-proc-batch-size 50 \
  --image-size 256 \
  --cfg-scale 1.5 \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 1 \
  --max-order 1 \
  --max_block_order 2 \
  --threshold 0.014 \
  --mid_cor 2 \

CUDA_VISIBLE_DEVICES=5 torchrun --nnodes=1 --nproc_per_node=1  --master_port=29504 sample_ddp.py \
  --model DiT-XL/2 \
  --per-proc-batch-size 50 \
  --image-size 256 \
  --cfg-scale 1.5 \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 1 \
  --max-order 1 \
  --max_block_order 3 \
  --threshold 0.014 \
  --mid_cor 2 \

CUDA_VISIBLE_DEVICES=5 torchrun --nnodes=1 --nproc_per_node=1  --master_port=29504 sample_ddp.py \
  --model DiT-XL/2 \
  --per-proc-batch-size 50 \
  --image-size 256 \
  --cfg-scale 1.5 \
  --ddim-sample \
  --num-sampling-steps 50 \
  --interval 1 \
  --max-order 1 \
  --max_block_order 4 \
  --threshold 0.014 \
  --mid_cor 2 \


# CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --nproc_per_node=1 --master_port=29501 sample_ddp.py \
#   --model DiT-XL/2 \
#   --per-proc-batch-size 50 \
#   --image-size 256 \
#   --cfg-scale 1.5 \
#   --ddim-sample \
#   --num-sampling-steps 50 \
#   --interval 1 \
#   --max-order 1 \
#   --max_block_order 1 \
#   --threshold 0.025 \
#   --mid_cor 2 \



#CUDA_VISIBLE_DEVICES=3 python eva/evaluator.py /root/paddlejob/workspace/env_run/test_data/VIRTUAL_imagenet256_labeled.npz /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_dit/Taylorseer-rel0.055-mid_cor4-max_block_order1-inter4-max_order4cfg1.5-seed0-step50.npz

#CUDA_VISIBLE_DEVICES=3 python eva/evaluator.py /root/paddlejob/workspace/env_run/test_data/VIRTUAL_imagenet256_labeled.npz /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX_A800/inf_speed_dit/Taylor-rel0.055-mid_cor4-max_block_order1-1-max_order1cfg1.5-seed0-step50.npz

