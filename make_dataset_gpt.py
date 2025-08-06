import json
import torch
from transformers import AutoTokenizer
import os
import argparse

MODEL_NAME = "openai/gpt-oss-20b"
TARGET_TOKEN_COUNT = 100_000_000
DEFAULT_OUTPUT_DIR = "tokenized_dataset"

def sanitize_filename(name):
    return name.replace("/", "__")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="train.json or .jsonl のパス")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_dir = args.output_dir
    safe_name = sanitize_filename(os.path.basename(dataset_path))

    os.makedirs(output_dir, exist_ok=True)

    output_pt = os.path.join(output_dir, f"{safe_name}.pt")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    total_tokens = 0
    sample_count = 0
    records = []

    print("🔍 User/Assistantデータ作成中...")

    # ここでjson.load() で丸ごと読み込み
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for example in data:
        instruction = example.get("instruction", "").strip()
        input_text = example.get("input", "").strip()
        output_text = example.get("output", "").strip()

        if not instruction or not output_text:
            continue

        user_message = f"{instruction}\n{input_text}" if input_text else instruction
        print(user_message)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output_text},
        ]

        tokens = tokenizer.apply_chat_template(messages, tokenize=True)
        total_tokens += len(tokens)
        records.append(messages)
        sample_count += 1

        if sample_count % 1000 == 0:
            print(f"🪄 {sample_count} 件収集 / Tokens: {total_tokens}")

        if total_tokens >= TARGET_TOKEN_COUNT:
            print(f"🎯 Target tokens reached: {total_tokens}")
            break

    print(f"💾 .pt 保存中: {output_pt}")
    torch.save(records, output_pt, _use_new_zipfile_serialization=True)
    print(f"✅ 保存完了: {output_pt}")

if __name__ == "__main__":
    main()
