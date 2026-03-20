# 3080 Autoresearch Lab

This repo is set up so you can run an autoresearch-style loop on a single RTX 3080 before using expensive challenge compute.

## Setup

To set up a new session, do this:

1. Create a fresh branch named `autoresearch/<date>-3080` from the current branch.
2. Read these files for context:
   - `README.md`
   - `lab/README.md`
   - `train_gpt.py`
   - `program.md`
3. Verify the dataset and tokenizer exist under `data/datasets/` and `data/tokenizers/`.
   - If they do not, run `.venv\Scripts\python.exe data\cached_challenge_fineweb.py --variant sp1024 --train-shards 1`
4. Initialize the results ledger if it does not exist yet:
   - `.venv\Scripts\python.exe lab\run_experiment.py --init-results`
5. Start with the baseline screening run:
   - `.venv\Scripts\python.exe lab\run_experiment.py --profile screen --note "baseline 8x448"`

If `.venv` does not exist yet, create it first on Windows:

- `py -3.12 -m venv .venv`
- `.venv\Scripts\activate`
- `python -m pip install --upgrade pip`
- `python -m pip install -r requirements.txt`
- `python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0`

or just run `setup_windows_env.cmd`.

For data bootstrapping, `setup_lab_data.cmd` downloads the minimal published SP-1024 assets.

Do not skip the CUDA PyTorch install on Windows, or the environment may end up CPU-only.

## Scope

The main research target is `train_gpt.py`.

You should avoid editing the lab harness unless the harness itself is broken or a tiny harness improvement clearly helps the loop. Most experiments should only change `train_gpt.py`.

## Run modes

- `screen` is the default local search profile on a 3080. It skips quantized roundtrip to increase experiments per day.
- `preview` saves a raw checkpoint so you can inspect sampled text output.
- `promote` reruns a promising idea with quantized roundtrip turned back on.
- `officialish` moves closer to the real challenge baseline shape.
- `smoke` is only for crash checks.

## What counts as improvement

Primary metric during local search:

- lower `prequant_val_bpb`

Promotion metric:

- lower `quantized_val_bpb`
- still under the artifact cap when quantized size is available

Do not trust a local change that only looks better before quantization if the quantized gap gets much worse.

## Logging

Every run appends to `lab/results.tsv` automatically.

Columns include:

- commit
- profile
- status
- prequant `val_bpb`
- quantized `val_bpb`
- quantization gap
- peak VRAM
- training seconds
- challenge-equivalent hours estimate

Trainer logs live in `logs/`. Console logs live in `lab/runs/`.

## Experiment loop

Loop like this:

1. Check the current branch and recent results in `lab/results.tsv`.
2. Pick one idea and edit `train_gpt.py`.
3. Commit the experiment before running it.
4. Run `.venv\Scripts\python.exe lab\run_experiment.py --profile screen --note "<idea>"`.
5. If it crashes, inspect the log and either fix the bug or discard the idea.
6. If `prequant_val_bpb` improves enough to justify the added complexity, keep the commit.
7. If it does not improve, reset back to the prior good commit.
8. Periodically rerun the best candidates with `--profile promote`.
9. Only spend more expensive compute once a candidate survives `promote` or `officialish`.

## Local constraints

- This machine is a single `RTX 3080 10GB`, so memory is tight.
- Use `GRAD_ACCUM_STEPS` and `TRAIN_BATCH_TOKENS` before collapsing context length.
- Prefer `VAL_LOSS_EVERY=0` for screening.
- On native Windows, use `.venv\Scripts\python.exe lab\run_experiment.py ...`, not global `python` and not `torchrun`, for local 1-GPU runs.

## Simplicity rule

All else equal, prefer simpler changes.

- Keep a tiny change with a real gain.
- Keep a simplification with equal performance.
- Discard a complicated change with marginal or noisy improvement.

## Never drift from the real task

This is still parameter golf, not a toy benchmark.

- Keep the same dataset, tokenizer, and evaluation harness.
- Treat local 3080 wins as screening signals.
- Re-check promising ideas with quantized roundtrip before trusting them.
