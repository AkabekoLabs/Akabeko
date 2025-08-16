cd ../../
python make_dataset.py \
  --source_type csv \
  --csv_path ./dataset/aizu_1000.csv \
  --question_field Question \
  --answer_field Answer \
  --output_dir tokenized_dataset \
  --target_tokens 5000000
