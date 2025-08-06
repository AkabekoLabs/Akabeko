import json
import torch
from datasets import load_dataset
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
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace上のデータセット名（例: myorg/mydataset）")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    output_dir = args.output_dir
    safe_name = sanitize_filename(dataset_name)

    os.makedirs(output_dir, exist_ok=True)

    output_pt = os.path.join(output_dir, f"{safe_name}.pt")
    output_txt = os.path.join(output_dir, f"{safe_name}.txt")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    total_tokens = 0
    sample_count = 0
    records = []

    print("🔍 User/Assistant 形式の会話データを収集中...")

    with open(output_txt, "w", encoding="utf-8") as txt_file:
        for example in dataset:
            instruction = example.get("instruction", "").strip()
            input_text = example.get("input", "").strip()
            output_text = example.get("output", "").strip()

            if not instruction or not output_text:
                continue

            user_message = f"{instruction}\n{input_text}" if input_text else instruction
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": output_text},
            ]

            try:
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)
            except Exception as e:
                print(f"⚠️ スキップ: {e}")
                continue

            if len(tokens) == 0:
                continue

            records.append(messages)

            txt_file.write(json.dumps({
                "messages": messages,
                "token_count": len(tokens)
            }, ensure_ascii=False) + "\n")

            total_tokens += len(tokens)
            sample_count += 1

            if sample_count % 1000 == 0:
                print(f"🪄 {sample_count} 件 / 累積トークン数 {total_tokens:,}")

            if total_tokens >= TARGET_TOKEN_COUNT:
                print(f"🎯 目標到達: {total_tokens:,} トークン")
                break

    print(f"💾 .pt 保存中: {output_pt}")
    torch.save(records, output_pt, _use_new_zipfile_serialization=True)

    print(f"✅ 保存完了:")
    print(f"    - {output_pt}")
    print(f"    - {output_txt}")

if __name__ == "__main__":
    main()
