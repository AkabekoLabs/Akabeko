cd ../../
LATEST_DIR=$(ls -td /Akabeko/export_hf/*/ | head -1)
echo "最新ディレクトリ: $LATEST_DIR"

python chat_ds.py \
  --hf_dir "$LATEST_DIR" \
  --dtype bf16 \
  --attn_impl auto \
  --device_map auto

