# 3080 Lab

This lab adapts the repository into an autoresearch-style local search loop for a single RTX 3080 before you spend real leaderboard compute.

## Windows setup

Create the local environment once:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
```

Or use the helper script:

```bat
setup_windows_env.cmd
```

After that, prefer either:

```bat
.venv\Scripts\python.exe lab\run_experiment.py --profile screen --note "baseline 8x448"
```

or:

```bat
run_lab.cmd --profile screen --note "baseline 8x448"
```

The extra PyTorch command matters on Windows because plain `pip install -r requirements.txt` can land you on the CPU wheel, which will make CUDA training fail.

## Data setup

The local lab also needs the published FineWeb shards and tokenizer. For a minimal 3080 setup, download one training shard plus the full fixed validation split:

```bat
.venv\Scripts\python.exe data\cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

Or use:

```bat
setup_lab_data.cmd
```

If data is missing, `lab/run_experiment.py` now fails fast with that exact command instead of crashing.

## Files

- `program.md` - instructions to point your coding agent at
- `lab/run_experiment.py` - local launcher, log parser, and TSV appender
- `sample.py` - text generation from a saved checkpoint
- `lab/results.tsv` - untracked experiment ledger created on demand
- `lab/runs/` - untracked console logs for each run
- `lab/checkpoints/` - optional raw checkpoints for preview/sampling
- `logs/` - trainer logs written by `train_gpt.py`

## Profiles

- `smoke` - 10 minute crash check, no quantized roundtrip
- `preview` - 10 minute checkpointed run so you can sample actual text
- `screen` - the default 3080 baseline: `8x448`, `TRAIN_BATCH_TOKENS=131072`, `MAX_WALLCLOCK_SECONDS=5400`, no quantized roundtrip
- `promote` - same model as `screen`, but turns quantized roundtrip back on and runs for 2 hours
- `officialish` - closer to the challenge baseline shape: `9x512`, `TRAIN_BATCH_TOKENS=131072`, quantized roundtrip on, 2 hours

Print the exact preset env with:

```bat
.venv\Scripts\python.exe lab\run_experiment.py --print-profiles
```

## First run

Initialize the results table once:

```bat
.venv\Scripts\python.exe lab\run_experiment.py --init-results
```

Then launch the recommended baseline:

```bat
.venv\Scripts\python.exe lab\run_experiment.py --profile screen --note "baseline 8x448"
```

If you want to see output first, run a checkpointed preview and sample from it:

```bat
.venv\Scripts\python.exe lab\run_experiment.py --profile preview --note "preview 8x448"
.venv\Scripts\python.exe sample.py --prompt "The meaning of life is"
```

On native Windows this uses plain `python train_gpt.py` instead of `torchrun`, which avoids the NCCL path that is meant for Linux multi-GPU runs.

## Overrides

You can override any env var that `train_gpt.py` supports:

```bat
.venv\Scripts\python.exe lab\run_experiment.py --profile screen --note "try more accumulation" --set GRAD_ACCUM_STEPS=16 --set TRAIN_BATCH_TOKENS=131072
```

Useful local knobs:

- `GRAD_ACCUM_STEPS` - lets you lower per-microstep memory without changing global batch
- `TRAIN_BATCH_TOKENS` - main throughput vs memory knob
- `MAX_WALLCLOCK_SECONDS` - set the local run budget
- `SKIP_QUANT_ROUNDTRIP=1` - faster screening runs
- `SAVE_RAW_CHECKPOINT=1` - save `lab/checkpoints/<run_id>.pt` for sampling
- `VAL_LOSS_EVERY=0` - only validate at the end

## Results

Each run appends a row to `lab/results.tsv` with:

- run id, commit, profile, status, note
- prequant and quantized `val_bpb`
- quantization gap
- size numbers
- peak VRAM
- training seconds, steps, tokens per second
- estimated challenge-equivalent hours based on the official 8xH100 baseline token count

The launcher parses the `lab_summary` line emitted by `train_gpt.py`, so the agent does not need ad hoc `grep` commands.

## Suggested loop

1. Start from `screen` and only modify `train_gpt.py`.
2. Keep the commit if prequant `val_bpb` improves materially and memory stays reasonable.
3. Promote only the best candidates with `--profile promote`.
4. Use `officialish` when you want to test whether a good 3080 idea survives closer to the real challenge shape.

## Rough equivalence

The launcher estimates challenge-equivalent hours from local tokens/sec using the official baseline training volume of about `7.2247B` tokens in 10 minutes on `8xH100`.

Treat that estimate as a ranking aid, not as proof that a local winner will transfer.
