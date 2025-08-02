from transformers import Qwen2Config
from transformers import Qwen3Config

class BaseModel:
    @staticmethod
    def get_qwen3_config(size: str) -> Qwen3Config:
        # 公称パラメータ（修正済み）
        PRESET = {
            "0.6B": dict(hidden_size=1024, num_hidden_layers=28,
                         num_attention_heads=16, num_key_value_heads=8, intermediate_size=3072),
            "1.7B": dict(hidden_size=2048, num_hidden_layers=28,
                         num_attention_heads=16, num_key_value_heads=8, intermediate_size=6144),
            "4B":   dict(hidden_size=2560, num_hidden_layers=36,
                         num_attention_heads=32, num_key_value_heads=8, intermediate_size=9728),
        }

        if size not in PRESET:
             raise ValueError(f"Unsupported size: {size}")

        cfg_kwargs = PRESET[size].copy()
        cfg_kwargs.update(
            rope_theta=1000000,
            use_sliding_window=False,
        )
        config = Qwen3Config(**cfg_kwargs)

        # セーフティネット
        if getattr(config, "num_key_value_heads", 0) in (0, None):
            config.num_key_value_heads = config.num_attention_heads

        return config

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
