import json
import os
import argparse
import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_OUTPUT_DIR = "tokenized_dataset"
DEFAULT_TARGET_TOKENS = 100_000_000  # 既定値は従来どおり

def sanitize_filename(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__")

def iter_hf_examples(dataset_name: str):
    # HF: instruction / input / output 形式を想定
    ds = load_dataset(dataset_name, split="train", streaming=True)
    for ex in ds:
        instruction = (ex.get("instruction") or "").strip()
        input_text  = (ex.get("input") or "").strip()
        output_text = (ex.get("output") or "").strip()
        if not instruction or not output_text:
            continue
        user_message = f"{instruction}\n{input_text}" if input_text else instruction
        yield {
            "user": user_message,
            "assistant": output_text
        }

def iter_csv_examples(csv_path: str, question_field: str, answer_field: str, encoding: str = "utf-8"):
    # CSV: Question / Answer 列を想定（列名は可変）
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if question_field not in reader.fieldnames or answer_field not in reader.fieldnames:
            raise ValueError(
                f"CSVの列名が一致しません。見つかった列: {reader.fieldnames} / "
                f"期待: '{question_field}', '{answer_field}'"
            )
        for row in reader:
            q = (row.get(question_field) or "").strip()
            a = (row.get(answer_field) or "").strip()
            if not q or not a:
                continue
            yield {
                "user": q,
                "assistant": a
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_type", type=str, choices=["hf", "csv"], default="hf",
                        help="入力ソースの種別（hf: HuggingFace, csv: ローカルCSV）")
    parser.add_argument("--dataset_name", type=str,
                        help="HuggingFace上のデータセット名（例: myorg/mydataset）")
    parser.add_argument("--csv_path", type=str, help="Question/Answer列を含むCSVファイルのパス（--source_type csv 時）")
    parser.add_argument("--question_field", type=str, default="Question",
                        help="CSVの質問カラム名（既定: Question）")
    parser.add_argument("--answer_field", type=str, default="Answer",
                        help="CSVの回答カラム名（既定: Answer）")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_tokens", type=int, default=DEFAULT_TARGET_TOKENS,
                        help="収集目標の累積トークン数（既定: 100,000,000）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大サンプル数（任意・早期打ち切り用）")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="CSV読み込み時のテキストエンコーディング（既定: utf-8）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 出力ファイル名のベースを決定
    if args.source_type == "hf":
        if not args.dataset_name:
            raise ValueError("--dataset_name が必要です（--source_type hf）")
        safe_name = sanitize_filename(args.dataset_name)
        name_for_out = safe_name
    else:
        if not args.csv_path:
            raise ValueError("--csv_path が必要です（--source_type csv）")
        base = os.path.splitext(os.path.basename(args.csv_path))[0]
        name_for_out = sanitize_filename(base)

    output_pt  = os.path.join(args.output_dir, f"{name_for_out}.pt")
    output_txt = os.path.join(args.output_dir, f"{name_for_out}.txt")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 入力イテレータを準備
    if args.source_type == "hf":
        example_iter = iter_hf_examples(args.dataset_name)
        print("🔍 Hugging Face データセットから User/Assistant 形式を収集中...")
    else:
        example_iter = iter_csv_examples(args.csv_path, args.question_field, args.answer_field, encoding=args.encoding)
        print("🔍 CSV（Question/Answer）から User/Assistant 形式を収集中...")

    total_tokens = 0
    sample_count = 0
    records = []

    with open(output_txt, "w", encoding="utf-8") as txt_file:
        for ex in example_iter:
            messages = [
                {"role": "user", "content": ex["user"]},
                {"role": "assistant", "content": ex["assistant"]},
            ]
            try:
                tokens = tokenizer.apply_chat_template(messages, tokenize=True)
            except Exception as e:
                print(f"⚠️ スキップ: {e}")
                continue

            if not tokens:
                continue

            # 1行1JSON を追記
            line = {
                "messages": messages,
                "token_count": len(tokens),
            }
            txt_file.write(json.dumps(line, ensure_ascii=False) + "\n")

            # .pt 用にメモリへ格納
            records.append(messages)

            total_tokens += len(tokens)
            sample_count += 1

            if sample_count % 1000 == 0:
                print(f"🪄 {sample_count} 件 / 累積トークン数 {total_tokens:,}")

            if args.max_samples is not None and sample_count >= args.max_samples:
                print(f"🧯 max_samples に到達: {sample_count} 件で停止")
                break

            if total_tokens >= args.target_tokens:
                print(f"🎯 目標到達: {total_tokens:,} トークン")
                break

    print(f"💾 .pt 保存中: {output_pt}")
    torch.save(records, output_pt, _use_new_zipfile_serialization=True)

    print("✅ 保存完了:")
    print(f"    - {output_pt}")
    print(f"    - {output_txt}")

if __name__ == "__main__":
    main()