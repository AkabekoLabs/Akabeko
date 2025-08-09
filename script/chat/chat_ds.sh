cd ../../
# 例: /workspace/AkabekoLLM/export_hf/hf_final_epoch1_step0/ にある場合
python chat_ds.py \
  --hf_dir /workspace/AkabekoLLM/export_hf/hf_final_epoch1_step0 \
  --dtype bf16 \
  --attn_impl auto \
  --device_map auto
