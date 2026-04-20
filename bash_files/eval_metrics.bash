python3 snapViT/eval_metrics.py   \
    --data_root /media/hdd/ale_navone/GAIA/tempovine/dataset_tempovine_test   \
    --checkpoint /home/ale_navone/ws_pytorch/GAIA/snapViT/models/tempovine_2026_03_26_consecutive_frames_mixed_loss_lambda_0_5/best_model.pth  \
    --output_dir snapViT/evaluation/random_frames_mixed_loss_lambda_0_5_1\
    --num_samples 1500 \
  --run_ground_benchmark \
  --ground_benchmark_angles "45,90,135,180, 225, 270, 315" \
  --include_ground_swap_case \