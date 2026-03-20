from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import zlib
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_experiment import RESULTS_PATH, ROOT, append_result, build_row
from train_gpt import (
    CastedLinear,
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    eval_val,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover an interrupted promote-style run from a saved final_model.int8.ptz artifact."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", default="promote")
    parser.add_argument("--note", default="")
    parser.add_argument("--artifact", default="final_model.int8.ptz")
    parser.add_argument("--trainer-log", default="")
    parser.add_argument("--console-log", default="")
    parser.add_argument(
        "--tokenizer", default="./data/tokenizers/fineweb_1024_bpe.model"
    )
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
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
    parser.add_argument("--train-seq-len", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=131072)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    return parser


def default_log_path(run_id: str) -> Path:
    return ROOT / "logs" / f"{run_id}.txt"


def default_console_path(run_id: str) -> Path:
    return ROOT / "lab" / "runs" / f"{run_id}.log"


def resolve_path(raw: str, fallback: Path) -> Path:
    if not raw:
        return fallback
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def grab(pattern: str, text: str) -> re.Match[str]:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not find pattern: {pattern}")
    return match


def parse_partial_summary(log_text: str) -> dict[str, object]:
    train_line = grab(
        r"train_batch_tokens:(\d+) train_seq_len:(\d+) .*max_wallclock_seconds:([0-9.]+)",
        log_text,
    )
    stop_line = grab(
        r"stopping_early: wallclock_cap train_time:(\d+)ms step:(\d+)/(\d+)",
        log_text,
    )
    peak_line = grab(
        r"peak memory allocated: (\d+) MiB reserved: (\d+) MiB",
        log_text,
    )
    prequant_line = grab(
        r"final_prequant_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)",
        log_text,
    )
    raw_line = grab(r"Serialized model: (\d+) bytes", log_text)
    code_line = grab(r"Code size: (\d+) bytes", log_text)
    int8_line = grab(r"Serialized model int8\+zlib: (\d+) bytes", log_text)
    total_line = grab(r"Total submission size int8\+zlib: (\d+) bytes", log_text)
    return {
        "steps": int(stop_line.group(2)),
        "train_batch_tokens": int(train_line.group(1)),
        "train_seq_len": int(train_line.group(2)),
        "training_seconds": int(stop_line.group(1)) / 1000.0,
        "peak_alloc_mib": int(peak_line.group(1)),
        "peak_reserved_mib": int(peak_line.group(2)),
        "prequant_val_loss": float(prequant_line.group(1)),
        "prequant_val_bpb": float(prequant_line.group(2)),
        "model_bytes_raw": int(raw_line.group(1)),
        "code_bytes": int(code_line.group(1)),
        "model_bytes_int8_zlib": int(int8_line.group(1)),
        "total_submission_bytes_int8_zlib": int(total_line.group(1)),
    }


def summary_int(summary: dict[str, object], key: str) -> int:
    value = summary[key]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"Summary field {key} is not int-like: {type(value)!r}")


def summary_float(summary: dict[str, object], key: str) -> float:
    value = summary[key]
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Summary field {key} is not float-like: {type(value)!r}")


def recover_quantized_eval(
    args: argparse.Namespace, artifact_path: Path
) -> tuple[float, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hp = Hyperparameters()
    hp.val_batch_size = args.val_batch_size
    hp.train_seq_len = args.train_seq_len
    hp.tokenizer_path = args.tokenizer
    hp.data_path = args.data_path
    hp.val_files = str(Path(args.data_path) / "fineweb_val_*.bin")
    hp.vocab_size = args.vocab_size

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_absolute():
        tokenizer_path = (ROOT / tokenizer_path).resolve()
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()
    hp.tokenizer_path = str(tokenizer_path)
    hp.data_path = str(data_path)
    hp.val_files = str(data_path / "fineweb_val_*.bin")

    sp = spm.SentencePieceProcessor()
    sp.Load(hp.tokenizer_path)
    val_tokens = load_validation_tokens(hp.val_files, hp.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, hp.vocab_size, device)
    )

    model = (
        GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=bool(args.tie_embeddings),
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
        )
        .to(device)
        .bfloat16()
    )
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    quant_state = torch.load(
        io.BytesIO(zlib.decompress(artifact_path.read_bytes())), map_location="cpu"
    )
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        hp,
        model,
        0,
        1,
        device,
        args.grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - t0
    return q_val_loss, q_val_bpb, eval_seconds


def results_contains_run_id(run_id: str) -> bool:
    if not RESULTS_PATH.exists():
        return False
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return any(row.get("run_id") == run_id for row in reader)


def append_recovered_lines(
    trainer_log: Path,
    q_val_loss: float,
    q_val_bpb: float,
    eval_seconds: float,
    summary: dict[str, object],
) -> None:
    total_seconds = summary_float(summary, "training_seconds") + eval_seconds
    lab_summary = (
        "lab_summary "
        f"steps:{summary_int(summary, 'steps')} train_batch_tokens:{summary_int(summary, 'train_batch_tokens')} "
        f"train_seq_len:{summary_int(summary, 'train_seq_len')} grad_accum_steps:{summary_int(summary, 'grad_accum_steps')} "
        f"training_seconds:{summary_float(summary, 'training_seconds'):.3f} total_seconds:{total_seconds:.3f} "
        f"peak_alloc_mib:{summary_int(summary, 'peak_alloc_mib')} peak_reserved_mib:{summary_int(summary, 'peak_reserved_mib')} "
        f"prequant_val_loss:{summary_float(summary, 'prequant_val_loss'):.8f} "
        f"prequant_val_bpb:{summary_float(summary, 'prequant_val_bpb'):.8f} "
        f"quantized_val_loss:{q_val_loss:.8f} quantized_val_bpb:{q_val_bpb:.8f} "
        f"code_bytes:{summary_int(summary, 'code_bytes')} model_bytes_raw:{summary_int(summary, 'model_bytes_raw')} "
        f"model_bytes_int8_zlib:{summary_int(summary, 'model_bytes_int8_zlib')} "
        f"total_submission_bytes_int8_zlib:{summary_int(summary, 'total_submission_bytes_int8_zlib')}"
    )
    with trainer_log.open("a", encoding="utf-8") as f:
        print(
            f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{eval_seconds * 1000.0:.0f}ms",
            file=f,
        )
        print(
            f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}",
            file=f,
        )
        print(lab_summary, file=f)


def main() -> int:
    args = build_parser().parse_args()
    trainer_log = resolve_path(args.trainer_log, default_log_path(args.run_id))
    console_log = resolve_path(args.console_log, default_console_path(args.run_id))
    artifact_path = resolve_path(args.artifact, ROOT / "final_model.int8.ptz")
    note = args.note.strip() or args.run_id

    if not trainer_log.exists():
        raise FileNotFoundError(f"Missing trainer log: {trainer_log}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact: {artifact_path}")

    summary = parse_partial_summary(read_text(trainer_log))
    summary["grad_accum_steps"] = args.grad_accum_steps
    q_val_loss, q_val_bpb, eval_seconds = recover_quantized_eval(args, artifact_path)
    append_recovered_lines(trainer_log, q_val_loss, q_val_bpb, eval_seconds, summary)

    summary["quantized_val_loss"] = q_val_loss
    summary["quantized_val_bpb"] = q_val_bpb
    summary["total_seconds"] = summary_float(summary, "training_seconds") + eval_seconds

    if not results_contains_run_id(args.run_id):
        row = build_row(
            run_id=args.run_id,
            profile=args.profile,
            status="recovered",
            note=note,
            returncode=0,
            summary=summary,
            trainer_log=trainer_log,
            console_log=console_log,
        )
        append_result(row)

    print(f"run_id: {args.run_id}")
    print(f"trainer_log: {trainer_log.relative_to(ROOT)}")
    print(f"artifact: {artifact_path.relative_to(ROOT)}")
    print(f"quantized_val_loss: {q_val_loss:.8f}")
    print(f"quantized_val_bpb: {q_val_bpb:.8f}")
    print(f"eval_seconds: {eval_seconds:.3f}")
    print(f"results_tsv: {RESULTS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
