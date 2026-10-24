# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Configuration for the initial dense, noncausal LLaDA port."""

from dataclasses import dataclass, fields
from typing import Any

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
GENERATION_REVISION = "9182493720ed723ef8031210d85959364e51cbe0"


@dataclass(frozen=True)
class LLaDAConfig:
    """Dimensions and numerical settings of the supported LLaDA variant."""

    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 32
    n_layers: int = 32
    mlp_hidden_size: int = 12288
    vocab_size: int = 126464
    max_sequence_length: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "n_heads",
            "n_layers",
            "mlp_hidden_size",
            "vocab_size",
            "max_sequence_length",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_heads != self.n_kv_heads:
            raise ValueError(
                "The initial port supports equal query and KV head counts"
            )
        if self.d_model % self.n_heads or self.head_dim % 2:
            raise ValueError("RoPE requires an even, integral head dimension")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> "LLaDAConfig":
        """Read HF dimensions, rejecting unsupported architecture switches."""
        supported = {
            "model_type": "llada",
            "block_type": "llama",
            "activation_type": "silu",
            "layer_norm_type": "rms",
            "block_group_size": 1,
            "include_bias": False,
            "include_qkv_bias": False,
            "attention_layer_norm": False,
            "input_emb_norm": False,
            "weight_tying": False,
            "scale_logits": False,
            "rope": True,
            "rope_full_precision": True,
            "alibi": False,
            "layer_norm_with_affine": True,
            "bias_for_layer_norm": False,
        }
        for name, expected in supported.items():
            if config.get(name, expected) != expected:
                raise ValueError(f"Unsupported {name}: {config[name]!r}")
        return cls(
            **{f.name: config[f.name] for f in fields(cls) if f.name in config}
        )
