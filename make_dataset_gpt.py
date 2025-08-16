import json
import os
import argparse
import csv
import io
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_OUTPUT_DIR = "tokenized_dataset"
DEFAULT_TARGET_TOKENS = 100_000_000  # 既定値は従来どおり

# 質問/回答カラム名の候補（日本語・英語・略称など）
QUESTION_FIELD_CANDIDATES = [
    "Question", "question", "質問", "問い", "お題", "Q", "user", "prompt", "instruction"
]
ANSWER_FIELD_CANDIDATES = [
    "Answer", "answer", "回答", "解答", "A", "assistant", "output", "completion"
]

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

def sniff_csv(file_path: str, encoding: str, sample_bytes: int = 8192):
    """
    区切り文字とヘッダー有無を自動推定
    """
    with open(file_path, "rb") as rb:
        raw = rb.read(sample_bytes)
    # BOM やエンコーディング差異に強いように
    sample = raw.decode(encoding, errors="replace")
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","  # フォールバック
    try:
        has_header = sniffer.has_header(sample)
    except Exception:
        has_header = True
    return delimiter, has_header

def pick_field_by_candidates(fieldnames, candidates):
    for cand in candidates:
        if cand in fieldnames:
            return cand
    # ゆるい一致（小文字化）
    lower = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def iter_csv_examples(
    csv_path: str,
    question_field: str = None,
    answer_field: str = None,
    question_index: int = None,
    answer_index: int = None,
    encoding: str = "utf-8",
    delimiter: str = None,
    no_header: bool = False,
):
    """
    CSV: Question/Answer 列を読み込み。
    - ヘッダー有 → DictReader
    - ヘッダー無 → reader（列番号で指定、未指定なら 0,1 を採用）
    - 区切り文字・ヘッダー有無は自動推定（手動指定が優先）
    """
    # 区切り/ヘッダー自動推定（未指定の場合）
    if delimiter is None or no_header is None:
        auto_delim, auto_has_header = sniff_csv(csv_path, encoding=encoding)
        if delimiter is None:
            delimiter = auto_delim
        if no_header is None:
            no_header = not auto_has_header

    # 文字化け/BOM対策として utf-8-sig も試す
    try_encodings = [encoding]
    if encoding.lower() != "utf-8-sig":
        try_encodings.append("utf-8-sig")

    last_err = None
    for enc in try_encodings:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                if not no_header:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    if reader.fieldnames is None:
                        # フィールド名が取れない場合は生読みへ切替
                        f.seek(0)
                        rdr = csv.reader(f, delimiter=delimiter)
                        for row in rdr:
                            if not row:
                                continue
                            q_idx = 0 if question_index is None else question_index
                            a_idx = 1 if answer_index is None else answer_index
                            if len(row) <= max(q_idx, a_idx):
                                continue
                            q = (row[q_idx] or "").strip()
                            a = (row[a_idx] or "").strip()
                            if q and a:
                                yield {"user": q, "assistant": a}
                        return

                    # 列名の自動推定
                    q_field = question_field or pick_field_by_candidates(reader.fieldnames, QUESTION_FIELD_CANDIDATES)
                    a_field = answer_field or pick_field_by_candidates(reader.fieldnames, ANSWER_FIELD_CANDIDATES)
                    if q_field is None or a_field is None:
                        raise ValueError(
                            f"CSVの質問/回答カラムを特定できませんでした。見つかった列: {reader.fieldnames}\n"
                            f"手動で --question_field と --answer_field を指定するか、--no_header と列番号指定をお試しください。"
                        )

                    for row in reader:
                        q = (row.get(q_field) or "").strip()
                        a = (row.get(a_field) or "").strip()
                        if q and a:
                            yield {"user": q, "assistant": a}
                else:
                    rdr = csv.reader(f, delimiter=delimiter)
                    q_idx = 0 if question_index is None else question_index
                    a_idx = 1 if answer_index is None else answer_index
                    for row in rdr:
                        if not row:
                            continue
                        if len(row) <= max(q_idx, a_idx):
                            continue
                        q = (row[q_idx] or "").strip()
                        a = (row[a_idx] or "").strip()
                        if q and a:
                            yield {"user": q, "assistant": a}
            return
        except Exception as e:
            last_err = e
            continue
    # 全エンコーディングで失敗
    raise last_err if last_err else RuntimeError("CSV読み込みに失敗しました。")

def make_out_paths(output_dir: str, name_for_out: str):
    os.makedirs(output_dir, exist_ok=True)
    output_pt    = os.path.join(output_dir, f"{name_for_out}.pt")
    output_jsonl = os.path.join(output_dir, f"{name_for_out}.jsonl")
    return output_pt, output_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_type", type=str, choices=["hf", "csv"], default="hf",
                        help="入力ソースの種別（hf: HuggingFace, csv: ローカルCSV）")

    # HF
    parser.add_argument("--dataset_name", type=str,
                        help="HuggingFace上のデータセット名（例: myorg/mydataset）")

    # CSV
    parser.add_argument("--csv_path", type=str, help="Question/Answer列を含むCSVファイルのパス（--source_type csv 時）")
    parser.add_argument("--question_field", type=str, default=None,
                        help="CSVの質問カラム名（ヘッダー有の場合）")
    parser.add_argument("--answer_field", type=str, default=None,
                        help="CSVの回答カラム名（ヘッダー有の場合）")
    parser.add_argument("--question_index", type=int, default=None,
                        help="ヘッダー無の場合の質問列番号（0始まり。未指定なら0）")
    parser.add_argument("--answer_index", type=int, default=None,
                        help="ヘッダー無の場合の回答列番号（0始まり。未指定なら1）")
    parser.add_argument("--delimiter", type=str, default=None,
                        help="CSVの区切り文字（未指定なら自動検出）")
    parser.add_argument("--no_header", action="store_true",
                        help="CSVにヘッダーが無い場合に指定（未指定なら自動判定）")
    parser.add_argument("--encoding", type=str, default="utf-8",
                        help="CSV読み込み時のテキストエンコーディング（既定: utf-8。自動で utf-8-sig も試行）")

    # 共通
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_tokens", type=int, default=DEFAULT_TARGET_TOKENS,
                        help="収集目標の累積トークン数（既定: 100,000,000）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大サンプル数（任意・早期打ち切り用）")

    args = parser.parse_args()

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

    output_pt, output_jsonl = make_out_paths(args.output_dir, name_for_out)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 入力イテレータを準備
    if args.source_type == "hf":
        example_iter = iter_hf_examples(args.dataset_name)
        print("🔍 Hugging Face データセットから User/Assistant 形式を収集中...")
    else:
        example_iter = iter_csv_examples(
            csv_path=args.csv_path,
            question_field=args.question_field,
            answer_field=args.answer_field,
            question_index=args.question_index,
            answer_index=args.answer_index,
            encoding=args.encoding,
            delimiter=args.delimiter,
            no_header=getattr(args, "no_header", None),  # None の場合は自動判定
        )
        print("🔍 CSV から User/Assistant 形式を収集中...")

    total_tokens = 0
    sample_count = 0
    records = []

    with open(output_jsonl, "w", encoding="utf-8") as jf:
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

            # 1行1JSON（学習用に使いやすい .jsonl）
            line = {
                "messages": messages,
                "token_count": len(tokens),
            }
            jf.write(json.dumps(line, ensure_ascii=False) + "\n")

            # .pt 用（messages の配列のみを格納）
            records.append(messages)

            total_tokens += len(tokens)
            sample_count += 1

            if sample_count % 1000 == 0:
                print(f"🪄 {sample_count} 件 / 累積トークン数 {total_tokens:,}")

            if args.max_samples is not None and sample_count >= args.max_samples:
                print(f"🧯 max_samples に到達: {sample_count} 件で停止")
                break

            if args.target_tokens is not None and total_tokens >= args.target_tokens:
                print(f"🎯 目標到達: {total_tokens:,} トークン")
                break

    print(f"💾 .pt 保存中: {output_pt}")
    torch.save(records, output_pt, _use_new_zipfile_serialization=True)

    print("✅ 保存完了:")
    print(f"    - {output_pt}")
    print(f"    - {output_jsonl}")

if __name__ == "__main__":
    main()
