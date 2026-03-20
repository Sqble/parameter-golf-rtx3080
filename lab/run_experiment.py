from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LAB_DIR = ROOT / "lab"
RUNS_DIR = LAB_DIR / "runs"
RESULTS_PATH = LAB_DIR / "results.tsv"
CHALLENGE_BASELINE_TOKENS = 7_224_688_640
DEFAULT_DATA_PATH = ROOT / "data" / "datasets" / "fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"

RESULT_COLUMNS = [
    "timestamp",
    "run_id",
    "commit",
    "profile",
    "status",
    "note",
    "returncode",
    "prequant_val_bpb",
    "quantized_val_bpb",
    "quant_gap",
    "total_submission_bytes_int8_zlib",
    "code_bytes",
    "model_bytes_int8_zlib",
    "peak_vram_gb",
    "training_seconds",
    "total_seconds",
    "steps",
    "tokens_seen_m",
    "tokens_per_second",
    "challenge_equiv_hours",
    "trainer_log",
    "console_log",
]

PROFILES = {
    "smoke": {
        "NUM_LAYERS": "8",
        "MODEL_DIM": "448",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_BATCH_SIZE": "65536",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "50",
        "MAX_WALLCLOCK_SECONDS": "600",
        "SKIP_QUANT_ROUNDTRIP": "1",
    },
    "preview": {
        "NUM_LAYERS": "8",
        "MODEL_DIM": "448",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_BATCH_SIZE": "65536",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "50",
        "MAX_WALLCLOCK_SECONDS": "600",
        "SKIP_QUANT_ROUNDTRIP": "1",
        "SAVE_RAW_CHECKPOINT": "1",
    },
    "screen": {
        "NUM_LAYERS": "8",
        "MODEL_DIM": "448",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TRAIN_BATCH_TOKENS": "131072",
        "VAL_BATCH_SIZE": "131072",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "100",
        "MAX_WALLCLOCK_SECONDS": "5400",
        "SKIP_QUANT_ROUNDTRIP": "1",
    },
    "promote": {
        "NUM_LAYERS": "8",
        "MODEL_DIM": "448",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TRAIN_BATCH_TOKENS": "131072",
        "VAL_BATCH_SIZE": "131072",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "100",
        "MAX_WALLCLOCK_SECONDS": "7200",
        "SKIP_QUANT_ROUNDTRIP": "0",
    },
    "officialish": {
        "NUM_LAYERS": "9",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TRAIN_BATCH_TOKENS": "131072",
        "VAL_BATCH_SIZE": "131072",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "100",
        "MAX_WALLCLOCK_SECONDS": "7200",
        "SKIP_QUANT_ROUNDTRIP": "0",
    },
}


def sanitize_note(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "run"


def ensure_results_file() -> None:
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writeheader()


def parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Overrides must look like KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        key = key.strip().upper()
        if not key:
            raise ValueError(f"Override key cannot be empty: {pair}")
        overrides[key] = value.strip()
    return overrides


def short_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "nogit"


def parse_value(raw: str):
    lowered = raw.lower()
    if lowered in {"nan", "inf", "-inf"}:
        return float(raw)
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_summary(log_path: Path) -> dict[str, object]:
    summary: dict[str, object] = {}
    if not log_path.exists():
        return summary
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("lab_summary "):
                fields = line.strip().split()[1:]
                summary = {}
                for field in fields:
                    if ":" not in field:
                        continue
                    key, value = field.split(":", 1)
                    summary[key] = parse_value(value)
    return summary


def append_result(row: dict[str, object]) -> None:
    ensure_results_file()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, delimiter="\t")
        writer.writerow(row)


def summary_int(summary: dict[str, object], key: str, default: int = 0) -> int:
    value = summary.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def summary_float(summary: dict[str, object], key: str, default: float = 0.0) -> float:
    value = summary.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def build_row(
    *,
    run_id: str,
    profile: str,
    status: str,
    note: str,
    returncode: int,
    summary: dict[str, object],
    trainer_log: Path,
    console_log: Path,
) -> dict[str, object]:
    steps = summary_int(summary, "steps", 0)
    train_batch_tokens = summary_int(summary, "train_batch_tokens", 0)
    training_seconds = summary_float(summary, "training_seconds", 0.0)
    prequant_val_bpb = summary_float(summary, "prequant_val_bpb", float("nan"))
    quantized_val_bpb = summary_float(summary, "quantized_val_bpb", float("nan"))
    peak_alloc_mib = summary_float(summary, "peak_alloc_mib", 0.0)
    tokens_seen = steps * train_batch_tokens
    tokens_seen_m = tokens_seen / 1_000_000.0
    tokens_per_second = tokens_seen / training_seconds if training_seconds > 0 else 0.0
    challenge_equiv_hours = (
        CHALLENGE_BASELINE_TOKENS / tokens_per_second / 3600.0
        if tokens_per_second > 0
        else 0.0
    )
    quant_gap = quantized_val_bpb - prequant_val_bpb
    if prequant_val_bpb != prequant_val_bpb:
        quant_gap = float("nan")
    if quantized_val_bpb != quantized_val_bpb:
        quant_gap = float("nan")
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "commit": short_commit(),
        "profile": profile,
        "status": status,
        "note": note.replace("\t", " "),
        "returncode": returncode,
        "prequant_val_bpb": f"{prequant_val_bpb:.8f}"
        if prequant_val_bpb == prequant_val_bpb
        else "nan",
        "quantized_val_bpb": f"{quantized_val_bpb:.8f}"
        if quantized_val_bpb == quantized_val_bpb
        else "nan",
        "quant_gap": f"{quant_gap:.8f}" if quant_gap == quant_gap else "nan",
        "total_submission_bytes_int8_zlib": summary_int(
            summary, "total_submission_bytes_int8_zlib", -1
        ),
        "code_bytes": summary_int(summary, "code_bytes", -1),
        "model_bytes_int8_zlib": summary_int(summary, "model_bytes_int8_zlib", -1),
        "peak_vram_gb": f"{peak_alloc_mib / 1024.0:.2f}",
        "training_seconds": f"{training_seconds:.3f}",
        "total_seconds": f"{summary_float(summary, 'total_seconds', 0.0):.3f}",
        "steps": steps,
        "tokens_seen_m": f"{tokens_seen_m:.3f}",
        "tokens_per_second": f"{tokens_per_second:.1f}",
        "challenge_equiv_hours": f"{challenge_equiv_hours:.2f}",
        "trainer_log": str(trainer_log.relative_to(ROOT))
        if trainer_log.exists()
        else "",
        "console_log": str(console_log.relative_to(ROOT))
        if console_log.exists()
        else "",
    }


def print_profiles() -> None:
    for name, env in PROFILES.items():
        print(f"[{name}]")
        for key in sorted(env):
            print(f"  {key}={env[key]}")


def resolve_training_paths(env: dict[str, str]) -> tuple[Path, Path]:
    data_path = Path(env.get("DATA_PATH", str(DEFAULT_DATA_PATH)))
    tokenizer_path = Path(env.get("TOKENIZER_PATH", str(DEFAULT_TOKENIZER_PATH)))
    if not data_path.is_absolute():
        data_path = (ROOT / data_path).resolve()
    if not tokenizer_path.is_absolute():
        tokenizer_path = (ROOT / tokenizer_path).resolve()
    return data_path, tokenizer_path


def ensure_training_assets(env: dict[str, str]) -> None:
    data_path, tokenizer_path = resolve_training_paths(env)
    train_shards = sorted(data_path.glob("fineweb_train_*.bin"))
    val_shards = sorted(data_path.glob("fineweb_val_*.bin"))
    missing: list[str] = []
    if not tokenizer_path.is_file():
        missing.append(f"tokenizer: {tokenizer_path}")
    if not train_shards:
        missing.append(f"train shards under: {data_path}")
    if not val_shards:
        missing.append(f"val shards under: {data_path}")
    if not missing:
        return

    command = ".venv\\Scripts\\python.exe data\\cached_challenge_fineweb.py --variant sp1024 --train-shards 1"
    details = "\n".join(f"- missing {item}" for item in missing)
    raise FileNotFoundError(
        "Parameter Golf data is missing.\n"
        f"{details}\n"
        "Download the minimal local assets with:\n"
        f"{command}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and log a 3080 lab experiment.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="screen")
    parser.add_argument("--note", default="")
    parser.add_argument("--status", default="pending")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--init-results", action="store_true")
    parser.add_argument("--print-profiles", action="store_true")
    args = parser.parse_args()

    if args.print_profiles:
        print_profiles()
        return 0

    if args.init_results:
        ensure_results_file()
        print(RESULTS_PATH.relative_to(ROOT))
        return 0

    overrides = parse_overrides(args.overrides)
    profile_env = dict(PROFILES[args.profile])
    profile_env.update(overrides)

    note = args.note.strip() or args.profile
    run_id = profile_env.get("RUN_ID")
    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{args.profile}_{sanitize_note(note)}"
        profile_env["RUN_ID"] = run_id

    trainer_log = ROOT / "logs" / f"{run_id}.txt"
    console_log = RUNS_DIR / f"{run_id}.log"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(profile_env)

    command = [sys.executable, "train_gpt.py"]
    print("profile:", args.profile)
    print("run_id:", run_id)
    print("note:", note)
    print("command:", " ".join(command))
    print("trainer_log:", trainer_log.relative_to(ROOT))
    print("console_log:", console_log.relative_to(ROOT))
    print("env overrides:")
    for key in sorted(profile_env):
        print(f"  {key}={profile_env[key]}")

    if args.dry_run:
        return 0

    ensure_training_assets(profile_env)

    with console_log.open("w", encoding="utf-8") as out:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=False,
        )

    summary = parse_summary(trainer_log if trainer_log.exists() else console_log)
    status = args.status
    if proc.returncode != 0:
        status = "crash"
    elif not summary:
        status = "crash"

    row = build_row(
        run_id=run_id,
        profile=args.profile,
        status=status,
        note=note,
        returncode=proc.returncode,
        summary=summary,
        trainer_log=trainer_log,
        console_log=console_log,
    )
    append_result(row)

    print("result:")
    for key in (
        "status",
        "prequant_val_bpb",
        "quantized_val_bpb",
        "peak_vram_gb",
        "training_seconds",
        "challenge_equiv_hours",
        "total_submission_bytes_int8_zlib",
    ):
        print(f"  {key}: {row[key]}")
    print(f"  results_tsv: {RESULTS_PATH.relative_to(ROOT)}")

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
