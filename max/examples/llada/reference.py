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

"""Run the frozen HF LLaDA reference and record refinement canvases.

Uses the upstream generator unchanged from a local checkout at the pinned
commit. Large model downloads occur only when this script is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
GENERATION_REVISION = "9182493720ed723ef8031210d85959364e51cbe0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument(
        "--output", type=Path, default=Path("llada-reference.json")
    )
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-first-logits", type=Path)
    args = parser.parse_args()
    if not 0 < args.steps <= args.gen_length:
        parser.error("Require 0 < steps <= gen-length")

    # Read the committed blob, rather than a possibly modified working file.
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.reference_repo),
            "show",
            f"{GENERATION_REVISION}:generate.py",
        ]
    )
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(args.seed)
    # Disable stochastic/TF32 paths; cross-framework ties can still diverge.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_loader(
        "llada_pinned_generation", loader=None
    )
    assert spec is not None
    generator = importlib.util.module_from_spec(spec)
    exec(
        compile(source, f"{GENERATION_REVISION}/generate.py", "exec"),
        generator.__dict__,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )
    model = (
        AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            code_revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(args.device)
        .eval()
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt = tokenizer(
        prompt_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(args.device)
    if prompt.shape[1] + args.gen_length > model.config.max_sequence_length:
        parser.error(
            "Prompt plus generation exceeds the model's maximum sequence length"
        )
    canvases: list[list[list[int]]] = []

    def record_canvas(
        module: Any, positional: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        canvas = positional[0] if positional else kwargs["input_ids"]
        canvases.append(canvas.detach().cpu().tolist())

    handle = model.register_forward_pre_hook(record_canvas, with_kwargs=True)
    try:
        with torch.inference_mode():
            # Optional logits fixture uses the same initial canvas as generate.
            if args.dump_first_logits:
                initial = torch.full(
                    (1, prompt.shape[1] + args.gen_length),
                    model.config.mask_token_id,
                    dtype=torch.long,
                    device=args.device,
                )
                initial[:, : prompt.shape[1]] = prompt
                logits = model(initial).logits
                torch.save(
                    {"tokens": initial.cpu(), "logits": logits.float().cpu()},
                    args.dump_first_logits,
                )
                del logits
                canvases.clear()
            output = generator.generate(
                model,
                prompt,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.gen_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
                mask_id=model.config.mask_token_id,
            )
    finally:
        handle.remove()
    final_tokens = output.cpu().tolist()
    result = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "generation_revision": GENERATION_REVISION,
        "generation_sha256": hashlib.sha256(source).hexdigest(),
        "torch_version": torch.__version__,
        "device": args.device,
        "dtype": "bfloat16",
        "seed": args.seed,
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.gen_length,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "remasking": "low_confidence",
        "prompt": args.prompt,
        "prompt_tokens": prompt.cpu().tolist(),
        "canvases_before_forward": canvases,
        "final_tokens": final_tokens,
        "text": tokenizer.decode(
            output[0, prompt.shape[1] :], skip_special_tokens=True
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["text"])


if __name__ == "__main__":
    main()
