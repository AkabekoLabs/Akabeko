import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import argparse

MODEL_NAME = "Qwen/Qwen3-0.6B"
TARGET_TOKEN_COUNT = 100_000_000
CONFIG_NAME = "jpn_Jpan"
DEFAULT_OUTPUT_DIR = "tokenized_dataset"

def sanitize_filename(name):
    return name.replace("/", "__")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="出力ファイル名のベース")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    output_dir = args.output_dir
    safe_name = sanitize_filename(dataset_name)  # ← スラッシュを除去

    os.makedirs(output_dir, exist_ok=True)

    output_pt = os.path.join(output_dir, f"{safe_name}.pt")
    output_bin = os.path.join(output_dir, f"{safe_name}.bin")
    output_txt = os.path.join(output_dir, f"{safe_name}.txt")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    dataset = load_dataset(dataset_name, CONFIG_NAME, split="train", streaming=True)

    total_tokens = 0
    sample_count = 0
    records = []

    print("🔍 日本語テキストを収集中...")

    with open(output_txt, "w", encoding="utf-8") as txt_file:
        for example in dataset:
            text = example.get("text", "").strip()
            if not text:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_count = len(tokens)

            if token_count == 0:
                continue

            record = {
                "text": text,
                "input_ids": torch.tensor(tokens, dtype=torch.long)
            }
            records.append(record)

            txt_file.write(json.dumps({
                "text": text,
                "input_ids": tokens
            }, ensure_ascii=False) + "\n")

            total_tokens += token_count
            sample_count += 1

            if sample_count % 1000 == 0:
                print(f"🪄 {sample_count} 件目: 累積トークン数 {total_tokens:,}")

            if total_tokens >= TARGET_TOKEN_COUNT:
                print(f"🎯 目標到達: {total_tokens:,} トークン")
                break

    print(f"💾 .pt 保存中: {output_pt}")
    torch.save(records, output_pt, _use_new_zipfile_serialization=True)

    print(f"💾 .bin 保存中: {output_bin}")
    torch.save(records, output_bin, _use_new_zipfile_serialization=True)

    print(f"✅ 保存完了:")
    print(f"    - {output_pt}")
    print(f"    - {output_bin}")
    print(f"    - {output_txt}")

if __name__ == "__main__":
    main()

