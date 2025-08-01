# base_model.py
from transformers import Qwen2Config
from transformers import Qwen3Config

class BaseModel:

    @staticmethod
    def get_qwen3_config(size: str) -> Qwen3Config:
        if size == "0.5B":
            return Qwen3Config(
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=24,
                vocab_size=151936,
                bos_token_id=151643,
                eos_token_id=151643,
                rope_theta=10000.0,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                rms_norm_eps=1e-6,
                use_cache=True,
            )
        elif size == "0.6B":
            return Qwen3Config(
                hidden_size=1024,
                intermediate_size=4864,
                num_attention_heads=16,
                num_hidden_layers=24,
                vocab_size=151936,
                bos_token_id=151643,
                eos_token_id=151643,
                rope_theta=10000.0,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                rms_norm_eps=1e-6,
                use_cache=True,
            )
        else:
            raise ValueError(f"Unsupported Qwen3 size: {size}")

    @staticmethod
    def get_qwen2_config(size):
        """
        qwen2.5 0.5B, 1B, 3B, 7Bの設定値を取得
        """
        print(f"Size:{size}")
        if size == "0.5B":
            return Qwen2Config(
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151643,
                hidden_act="silu",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4864,            
                max_position_embeddings=513,
                max_window_layers=12,
                model_type="qwen2",
                num_attention_heads=16,
                num_hidden_layers=12,
                num_key_value_heads=16,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=1024,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,
                use_mrope=False,
                use_sliding_window=False,
                vocab_size=151936,
            )
        elif size == "1B":
            return Qwen2Config(
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151643,
                hidden_act="silu",
                hidden_size=2048,
                initializer_range=0.02,
                intermediate_size=8192,
                max_position_embeddings=513,
                max_window_layers=12,
                model_type="qwen2",
                num_attention_heads=16,
                num_hidden_layers=24,
                num_key_value_heads=16,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=1024,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,
                use_mrope=False,
                use_sliding_window=False,
                vocab_size=151936,
                use_flash_attention_2=True
            )
        elif size == "3B":
            return Qwen2Config(
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151643,
                hidden_act="silu",
                hidden_size=2560,
                initializer_range=0.02,
                intermediate_size=10240,
                max_position_embeddings=513,
                max_window_layers=12,
                model_type="qwen2",
                num_attention_heads=32,
                num_hidden_layers=32,
                num_key_value_heads=16,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=1024,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,
                use_mrope=False,
                use_sliding_window=False,
                vocab_size=151936,
                use_flash_attention_2=True 
            )
        elif size == "7B":
            return Qwen2Config(
                attention_dropout=0.0,
                bos_token_id=151643,
                eos_token_id=151643,
                hidden_act="silu",
                hidden_size=4096,
                initializer_range=0.02,
                intermediate_size=14336,
                max_position_embeddings=513,
                max_window_layers=12,
                model_type="qwen2",
                num_attention_heads=32,
                num_hidden_layers=32,
                num_key_value_heads=16,
                rms_norm_eps=1e-06,
                rope_theta=1000000.0,
                sliding_window=1024,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,
                use_mrope=False,
                use_sliding_window=False,
                vocab_size=151936,
                use_flash_attention_2=True 
            )
        else:
            raise ValueError(f"Unsupported model size: {size}")