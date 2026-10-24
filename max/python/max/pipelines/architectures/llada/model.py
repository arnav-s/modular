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

"""LLaDA ModuleV3 forward pass for unpadded token canvases.

Mirrors the pinned HF LLaDALlamaBlock: pre-RMSNorm, bidirectional
attention, split-half float32 RoPE, and a SiLU-gated MLP. Every call
recomputes the complete sequence. Parameter names match checkpoint keys
after removing the leading ``model.transformer.`` prefix.

Attention intentionally uses explicit score matrices for a correctness
baseline. Its quadratic workspace limits practical sequence lengths.
"""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, Linear, Module, ModuleList
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor

from .model_config import LLaDAConfig


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, head_dim: int) -> Tensor:
    """Rotate contiguous half-heads in float32, then restore input dtype."""
    left, right = x.cast(DType.float32).split([head_dim // 2] * 2, axis=-1)
    rotated = F.concat((-right, left), axis=-1)
    return (x.cast(DType.float32) * cos + rotated * sin).cast(x.dtype)


class LLaDABlock(Module):
    """One dense, bidirectional LLaDA transformer block."""

    def __init__(self, config: LLaDAConfig) -> None:
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ff_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.q_proj = Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = Linear(config.d_model, config.d_model, bias=False)
        self.attn_out = Linear(config.d_model, config.d_model, bias=False)
        self.ff_proj = Linear(
            config.d_model, config.mlp_hidden_size, bias=False
        )
        self.up_proj = Linear(
            config.d_model, config.mlp_hidden_size, bias=False
        )
        self.ff_out = Linear(config.mlp_hidden_size, config.d_model, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, length, _ = x.shape
        c = self.config
        normalized = self.attn_norm(x)
        q = (
            self.q_proj(normalized)
            .reshape((batch, length, c.n_heads, c.head_dim))
            .transpose(1, 2)
        )
        k = (
            self.k_proj(normalized)
            .reshape((batch, length, c.n_heads, c.head_dim))
            .transpose(1, 2)
        )
        v = (
            self.v_proj(normalized)
            .reshape((batch, length, c.n_heads, c.head_dim))
            .transpose(1, 2)
        )
        q = apply_rope(q, cos, sin, c.head_dim)
        k = apply_rope(k, cos, sin, c.head_dim)
        scores = (q.cast(DType.float32) @ k.cast(DType.float32).T) * (
            c.head_dim**-0.5
        )
        probabilities = F.softmax(scores)
        attention = (probabilities @ v.cast(DType.float32)).cast(x.dtype)
        attention = attention.transpose(1, 2).reshape(
            (batch, length, c.d_model)
        )
        x = x + self.attn_out(attention)
        normalized = self.ff_norm(x)
        return x + self.ff_out(
            F.silu(self.ff_proj(normalized)) * self.up_proj(normalized)
        )


class LLaDA(Module):
    """Full-sequence mask predictor with an untied vocabulary head.

    Inputs are token IDs ``[B, S]`` and float32 RoPE cos/sin tables
    ``[1, 1, S, head_dim]``. Output is float32 logits ``[B, S, V]``.
    Padding masks, cached decoding, and serving registration are outside
    this initial forward-pass implementation.
    """

    def __init__(self, config: LLaDAConfig) -> None:
        super().__init__()
        self.wte = Embedding(config.vocab_size, dim=config.d_model)
        self.blocks = ModuleList(
            [LLaDABlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ff_out = Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, tokens: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = self.wte(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.ff_out(self.ln_f(x)).cast(DType.float32)
