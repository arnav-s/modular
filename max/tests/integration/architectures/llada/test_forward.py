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

"""Small full-sequence LLaDA parity checks against independent PyTorch ops."""

import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.compilation import CompiledCallable
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.llada.model import LLaDA
from max.pipelines.architectures.llada.model_config import LLaDAConfig

ForwardFixture = tuple[
    LLaDAConfig,
    CompiledCallable,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]


@pytest.fixture(scope="module")
def forward_model() -> ForwardFixture:
    config = LLaDAConfig(
        d_model=8,
        n_heads=2,
        n_kv_heads=2,
        n_layers=2,
        mlp_hidden_size=12,
        vocab_size=17,
        max_sequence_length=8,
    )
    with F.lazy(), default_dtype(DType.float32):
        model = LLaDA(config).to(CPU())
    rng = np.random.default_rng(42)
    weights = {
        name: rng.normal(0, 0.2, tuple(int(d) for d in tensor.shape)).astype(
            np.float32
        )
        for name, tensor in model.parameters
    }
    for name in weights:
        if "norm.weight" in name or name == "ln_f.weight":
            weights[name] += 1.0
    compiled = model.compile(
        TensorType(DType.int64, (1, 4), device=DeviceRef.CPU()),
        TensorType(DType.float32, (1, 1, 4, 4), device=DeviceRef.CPU()),
        TensorType(DType.float32, (1, 1, 4, 4), device=DeviceRef.CPU()),
        weights=weights,
    )
    positions = torch.arange(4, dtype=torch.float32)
    inv_freq = 1.0 / config.rope_theta ** (torch.arange(0, 4, 2).float() / 4)
    phase = positions[:, None] * inv_freq[None, :]
    phase = torch.cat([phase, phase], dim=-1)[None, None]
    return (
        config,
        compiled,
        {k: torch.from_numpy(v) for k, v in weights.items()},
        phase.cos(),
        phase.sin(),
    )


def reference(
    tokens: torch.Tensor,
    config: LLaDAConfig,
    weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    def norm(x: torch.Tensor, name: str) -> torch.Tensor:
        return torch_f.rms_norm(
            x, (config.d_model,), weights[name], config.rms_norm_eps
        )

    def rotate(x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-right, left], dim=-1) * sin

    x = torch_f.embedding(tokens, weights["wte.weight"])
    for layer in range(config.n_layers):
        prefix = f"blocks.{layer}."
        h = norm(x, prefix + "attn_norm.weight")
        q, k, v = [
            torch_f.linear(h, weights[prefix + projection + ".weight"])
            .reshape(1, 4, config.n_heads, config.head_dim)
            .transpose(1, 2)
            for projection in ("q_proj", "k_proj", "v_proj")
        ]
        attention = torch_f.scaled_dot_product_attention(
            rotate(q), rotate(k), v, is_causal=False
        )
        x = x + torch_f.linear(
            attention.transpose(1, 2).reshape(1, 4, config.d_model),
            weights[prefix + "attn_out.weight"],
        )
        h = norm(x, prefix + "ff_norm.weight")
        gate = torch_f.silu(
            torch_f.linear(h, weights[prefix + "ff_proj.weight"])
        )
        up = torch_f.linear(h, weights[prefix + "up_proj.weight"])
        x = x + torch_f.linear(gate * up, weights[prefix + "ff_out.weight"])
    return torch_f.linear(norm(x, "ln_f.weight"), weights["ff_out.weight"])


def run(
    compiled: CompiledCallable,
    tokens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> np.ndarray:
    outputs = compiled.execute_raw(
        *(Buffer.from_dlpack(t) for t in (tokens, cos, sin))
    )
    return torch.from_dlpack(outputs[0]).numpy()


@pytest.mark.parametrize(
    "token_ids", [[1, 2, 3, 4], [1, 2, 3, 9], [0, 0, 0, 0]]
)
def test_forward_matches_pytorch(
    forward_model: ForwardFixture, token_ids: list[int]
) -> None:
    config, compiled, weights, cos, sin = forward_model
    tokens = torch.tensor([token_ids], dtype=torch.int64)
    expected = reference(tokens, config, weights, cos, sin).numpy()
    np.testing.assert_allclose(
        run(compiled, tokens, cos, sin), expected, atol=2e-5, rtol=2e-4
    )


def test_future_tokens_affect_earlier_logits(
    forward_model: ForwardFixture,
) -> None:
    _, compiled, _, cos, sin = forward_model
    first = run(compiled, torch.tensor([[1, 2, 3, 4]]), cos, sin)
    changed = run(compiled, torch.tensor([[1, 2, 3, 9]]), cos, sin)
    assert np.max(np.abs(first[:, 0] - changed[:, 0])) > 1e-4


def test_unsupported_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight_tying"):
        LLaDAConfig.from_hf({"weight_tying": True})
    with pytest.raises(ValueError, match="equal query"):
        LLaDAConfig(n_kv_heads=8)
