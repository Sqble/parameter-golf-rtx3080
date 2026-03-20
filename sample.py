from __future__ import annotations

import argparse
import io
import zlib
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from train_gpt import (
    CastedLinear,
    GPT,
    dequantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "lab" / "checkpoints"


def latest_checkpoint() -> Path:
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints found under lab/checkpoints. Run a preview or set SAVE_RAW_CHECKPOINT=1 first."
        )
    return checkpoints[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample text from a saved Parameter Golf checkpoint."
    )
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=448)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--tie-embeddings", type=int, default=1)
    parser.add_argument("--tied-embed-init-std", type=float, default=0.005)
    parser.add_argument("--logit-softcap", type=float, default=30.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    return parser


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def model_logits(model: GPT, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[torch.Tensor] = []
    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, x0)
    x = model.final_norm(x)
    if model.tie_embeddings:
        logits_proj = F.linear(x, model.tok_emb.weight)
    else:
        if model.lm_head is None:
            raise RuntimeError("lm_head missing for untied-embedding model")
        logits_proj = model.lm_head(x)
    return model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)


def load_payload(path: Path) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if path.suffix == ".ptz":
        quant_state = torch.load(
            io.BytesIO(zlib.decompress(path.read_bytes())), map_location="cpu"
        )
        config: dict[str, object] = {}
        return config, dequantize_state_dict_int8(quant_state)

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        config = payload.get("config", {})
        if not isinstance(config, dict):
            config = {}
        return config, payload["state_dict"]
    if isinstance(payload, dict):
        return {}, payload
    raise TypeError(f"Unsupported checkpoint payload in {path}")


def config_value(config: dict[str, object], key: str, fallback: object) -> object:
    value = config.get(key)
    return fallback if value is None else value


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = (
        latest_checkpoint() if args.checkpoint == "latest" else Path(args.checkpoint)
    )
    config, state_dict = load_payload(checkpoint_path)
    tokenizer_value = args.tokenizer or str(
        config_value(
            config, "tokenizer_path", "./data/tokenizers/fineweb_1024_bpe.model"
        )
    )
    tokenizer_path = Path(tokenizer_value)
    if not tokenizer_path.is_absolute():
        tokenizer_path = (ROOT / tokenizer_path).resolve()

    device = choose_device(args.device)
    model = GPT(
        vocab_size=int(config_value(config, "vocab_size", args.vocab_size)),
        num_layers=int(config_value(config, "num_layers", args.num_layers)),
        model_dim=int(config_value(config, "model_dim", args.model_dim)),
        num_heads=int(config_value(config, "num_heads", args.num_heads)),
        num_kv_heads=int(config_value(config, "num_kv_heads", args.num_kv_heads)),
        mlp_mult=int(config_value(config, "mlp_mult", args.mlp_mult)),
        tie_embeddings=bool(
            int(config_value(config, "tie_embeddings", args.tie_embeddings))
        ),
        tied_embed_init_std=float(
            config_value(config, "tied_embed_init_std", args.tied_embed_init_std)
        ),
        logit_softcap=float(config_value(config, "logit_softcap", args.logit_softcap)),
        rope_base=float(config_value(config, "rope_base", args.rope_base)),
        qk_gain_init=float(config_value(config, "qk_gain_init", args.qk_gain_init)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    if device.type == "cuda":
        model = model.bfloat16()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    model.eval()

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    prompt_ids = sp.encode(args.prompt, out_type=int)
    if not prompt_ids:
        prompt_ids = [sp.bos_id()] if sp.bos_id() >= 0 else [0]
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            logits = model_logits(model, tokens)[:, -1, :]
            if args.temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / args.temperature
                if args.top_k > 0:
                    topk_vals, topk_idx = torch.topk(
                        logits, k=min(args.top_k, logits.size(-1)), dim=-1
                    )
                    probs = torch.softmax(topk_vals, dim=-1)
                    sample = torch.multinomial(probs, num_samples=1)
                    next_token = topk_idx.gather(-1, sample)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)

    print(f"checkpoint: {checkpoint_path}")
    print(f"device: {device}")
    print(f"prompt: {args.prompt!r}")
    print("---")
    print(sp.decode(tokens[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
