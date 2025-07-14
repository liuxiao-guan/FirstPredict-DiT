#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# 设置真实图片路径
REAL_PATH="/root/paddlejob/workspace/env_run/test_data/VIRTUAL_imagenet256_labeled.npz"

# 设置包含多个生成图像子文件夹的根路径
GENERATED_ROOT="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_dit"

# 设置 FID 分辨率
RES=512
# 输出日志文件
LOG_FILE="scripts/dit_H8000_results.txt"
# echo "FID Results Log" > "$LOG_FILE"  # 清空旧文件并写入表头

# 遍历所有子目录
for GENERATED_PATH in "$GENERATED_ROOT"/*.npz; do
    # 检查路径是否已经记录过
    if grep -q "$GENERATED_PATH" "$LOG_FILE"; then
        echo "[SKIP] 已记录: $GENERATED_PATH"
        continue
    fi
    echo "Running FID for: $GENERATED_PATH"
    # 执行评估脚本并获取输出
    EVAL_OUTPUT=$(python eva/evaluator.py "$REAL_PATH" "$GENERATED_PATH")

    # 提取 Inception Score
    IS_VALUE=$(echo "$EVAL_OUTPUT" | grep -i "Inception Score" | grep -oE '[0-9]+\.[0-9]+')

    # 提取 FID
    FID_VALUE=$(echo "$EVAL_OUTPUT" | grep -i "^FID" | grep -oE '[0-9]+\.[0-9]+')

    # 提取 sFID
    SFID_VALUE=$(echo "$EVAL_OUTPUT" | grep -i "^sFID" | grep -oE '[0-9]+\.[0-9]+')

    # 提取 Precision
    PRECISION_VALUE=$(echo "$EVAL_OUTPUT" | grep -i "Precision" | grep -oE '[0-9]+\.[0-9]+')

    # 提取 Recall
    RECALL_VALUE=$(echo "$EVAL_OUTPUT" | grep -i "Recall" | grep -oE '[0-9]+\.[0-9]+')

    echo "$GENERATED_PATH : IS=$IS_VALUE, FID=$FID_VALUE, sFID=$SFID_VALUE, Precision=$PRECISION_VALUE, Recall=$RECALL_VALUE" >> "$LOG_FILE"

done


# # 自定义的生成图片路径列表（每个路径为一个子文件夹）
# GENERATED_LIST=(
#     "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k/date202506302334_step4_shift3__num100_gs3.5_seed42_h1024_w1024_bs1_sched-deterministic_cp-latest_ls-0.25_promptver-v1"
#     "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k/date202506301812_step4_shift3__num100_gs3_seed42_h1024_w1024_bs1_sched-deterministic_cp-latest_ls-0.25_promptver-v1"
#     "/root/paddlejob/workspace/env_run/output/gxl/pcm_eval_results_flux_coco1k/date202506292205_step4_shift3__num100_gs2_seed42_h1024_w1024_bs1_sched-deterministic_cp-latest_ls-0.25_promptver-v1"
# )

# # 遍历自定义路径并运行 FID
# for GENERATED_PATH in "${GENERATED_LIST[@]}"; do
#     echo "Running FID for: $GENERATED_PATH"
#     python fid_score.py "$REAL_PATH" "$GENERATED_PATH" --resolution $RES
# done