python3 snapViT/estimate_position.py \
  --data_root /media/hdd/ale_navone/GAIA/tempovine/dataset_tempovine_new \
  --checkpoint /home/ale_navone/ws_pytorch/GAIA/snapViT/models/tempovine_2026_03_16_consecutive_frames_mixed_loss/best_model.pth  \
  --output_dir snapViT/visualizations/estimate_position \
  --num_samples 20 \
  --top_k 10 \
  --grid_resolution_m 0.5 \
  --yaw_search_angles_deg "0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345" 