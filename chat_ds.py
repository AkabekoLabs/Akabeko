# -*- coding: utf-8 -*-
"""Interactive chat script for a finetuned HF snapshot (e.g., gpt-oss-20b) with streaming + ESC-cancel."""

import os
import sys
import time
import argparse
import warnings
import threading
import contextlib
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# quiet HF logs
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# ---------------- SDPA / Flash2 helpers ----------------
def _fa2_is_installed() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False

def _fa2_supports_s_aux() -> bool:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
    except Exception:
        try:
            from flash_attn import flash_attn_varlen_func  # type: ignore
        except Exception:
            return False
    import inspect as _inspect
    try:
        sig = _inspect.signature(flash_attn_varlen_func)
        params = sig.parameters
        if "s_aux" in params:
            return True
        return any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values())
    except Exception:
        return False

def select_attn_impl(requested: str) -> str:
    req = requested.lower()
    if req == "sdpa":
        return "sdpa"
    if req == "flash2":
        if _fa2_is_installed() and _fa2_supports_s_aux():
            return "flash_attention_2"
        print("⚠️ flash2 requested but incompatible flash-attn found → fallback to SDPA.")
        return "sdpa"
    if _fa2_is_installed() and _fa2_supports_s_aux():
        return "flash_attention_2"
    return "sdpa"

def _build_sdpa_ctx():
    # PyTorch 2.4+ / older 両対応
    try:
        from torch.nn.attention import sdpa_kernel as _new_sdpa_kernel, SDPBackend
        def _ctx():
            return _new_sdpa_kernel(
                backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            )
        return _ctx
    except Exception:
        def _ctx():
            return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
        return _ctx

sdpa_ctx = _build_sdpa_ctx()

# ---------------- prompt helpers ----------------
def has_chat_template(tokenizer) -> bool:
    try:
        return bool(getattr(tokenizer, "chat_template", None))
    except Exception:
        return False

def build_inputs(tokenizer, messages, device=None):
    """
    messages: [{"role":"user"|"assistant"|"system", "content": str}, ...]
    device=None → CPUのまま返す（device_map=auto と相性◎）
    """
    if has_chat_template(tokenizer):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt")
    else:
        enc = tokenizer(messages[-1]["content"], return_tensors="pt")
    return enc.to(device) if device is not None else enc

# ---------------- cancel support ----------------
class EventStopping(StoppingCriteria):
    """threading.Event が立ったら停止"""
    def __init__(self, event: threading.Event):
        self.event = event
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()

def _esc_listener(cancel_event: threading.Event, alive_check=None):
    """ESC（\x1b）を拾って cancel_event を立てる。alive_check() が False なら終了。"""
    try:
        if os.name == "nt":
            import msvcrt
            while not cancel_event.is_set() and (alive_check is None or alive_check()):
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b"\x1b",):  # ESC
                        cancel_event.set()
                        break
                time.sleep(0.03)
        else:
            import termios, tty, select
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not cancel_event.is_set() and (alive_check is None or alive_check()):
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch == "\x1b":  # ESC
                            cancel_event.set()
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except Exception:
        # 端末が無い/特殊環境では無視
        pass

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dir", required=True, help="save_pretrained 済みのHFディレクトリ（safetensors分割の場所）")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", default="none", choices=["none", "auto"],
                   help="auto: accelerateで自動分散配置（VRAM厳しいとき）")
    p.add_argument("--attn_impl", default="auto", choices=["auto", "sdpa", "flash2"])
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    args = p.parse_args()

    # dtype
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # attn impl
    attn_impl = select_attn_impl(args.attn_impl)

    # load tokenizer/model
    hf_dir = Path(args.hf_dir).resolve()
    if not hf_dir.exists():
        print(f"❌ HF dir not found: {hf_dir}")
        sys.exit(1)

    print(f"🔹 Loading HF snapshot from: {hf_dir}")
    print(f"   dtype={dtype}, attn_impl={attn_impl}, device_map={args.device_map}")

    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_load_kwargs = dict(
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.device_map == "auto":
        model_load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(hf_dir), **model_load_kwargs)
    model.config.use_cache = True  # 推論

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device_map == "none":
        model.to(device)

    # banner
    try:
        hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "hidden_size_qkv", None)
    except Exception:
        hidden_size = None
    print("\n🚀 Chat ready.")
    print(f"   model={getattr(model.config, '_name_or_path', 'local')} | hidden_size={hidden_size} | dtype={dtype}")
    if torch.cuda.is_available():
        print(f"   CUDA device(s): {torch.cuda.device_count()} | using SDPA-compatible attention\n")

    # stdin UTF-8
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=getattr(model.config, "eos_token_id", tokenizer.eos_token_id),
    )

    messages = []  # chat history for chat_template
    print("Type [exit] to quit.\n")

    with sdpa_ctx():
        while True:
            try:
                user_text = input("👤 You: ")
                if user_text.strip().lower() == "[exit]":
                    print("👋 Bye!")
                    break

                messages.append({"role": "user", "content": user_text})
                inputs = build_inputs(
                    tokenizer,
                    messages,
                    device if args.device_map == "none" else None,  # auto のときは CPU のまま
                )

                # --- streaming setup ---
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_special_tokens=True,
                    skip_prompt=True,
                    decode_kwargs={"clean_up_tokenization_spaces": True},
                )

                cancel_event = threading.Event()
                stop_list = StoppingCriteriaList([EventStopping(cancel_event)])

                # 生成スレッド
                use_autocast = torch.cuda.is_available() and dtype in (torch.bfloat16, torch.float16)
                def _run_generate():
                    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else contextlib.nullcontext()
                    with torch.inference_mode():
                        with ctx:
                            model.generate(
                                **inputs,
                                streamer=streamer,
                                stopping_criteria=stop_list,
                                **gen_kwargs,
                            )

                gen_thread = threading.Thread(target=_run_generate, daemon=True)
                gen_thread.start()

                # 出力プリンタスレッド
                out_text = ""
                def _printer():
                    nonlocal out_text
                    for new_text in streamer:
                        out_text += new_text
                        print(new_text, end="", flush=True)

                print_thread = threading.Thread(target=_printer, daemon=True)

                # ESC 監視スレッド（gen_thread が生きている間だけ）
                esc_thread = threading.Thread(
                    target=_esc_listener,
                    args=(cancel_event, lambda: gen_thread.is_alive()),
                    daemon=True,
                )

                print("🤖 Akabeko: (press ESC to cancel) ", end="", flush=True)
                print_thread.start()
                esc_thread.start()

                # 待機
                gen_thread.join()
                print_thread.join()

                was_cancelled = cancel_event.is_set()
                if was_cancelled:
                    print("\n⏹ Cancelled.\n")
                    # キャンセル時は履歴に残さない（必要なら下の行を有効化）
                    # messages.append({"role": "assistant", "content": out_text})
                else:
                    print()  # 改行
                    messages.append({"role": "assistant", "content": out_text})

            except KeyboardInterrupt:
                print("\nInterrupted. Type [exit] to quit.")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
