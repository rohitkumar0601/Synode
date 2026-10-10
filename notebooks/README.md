# Offline GPU notebook

`gpu_offline_run.ipynb` runs the GPU-only generators (TabDDPM, GReaT) on Kaggle or Colab and produces a results bundle you import locally.

## What it does

1. Detects the platform (`/kaggle/working` or `/content`) and checks a GPU is attached.
2. Clones the repo and installs `uv`; sets `STL_VENV_ROOT=<work>/.stl` and `STL_STORAGE_ROOT=<work>/store`.
3. Builds the `eval` and `synthcity` envs via `scripts/envrun.sh`, then swaps the synthcity env's torch to CUDA 12.1 and asserts `torch.cuda.is_available()`.
4. `prepare` (eval env), `generate --allow-gpu-only` (synthcity env), `evaluate` (eval env), `export-results` to `<work>/<dataset>-gpu.zip`.

Parameters (first cell): `REPO_URL`, `DATASET` (default `adult`), `SEEDS` (`[0,1,2]`), `GENERATORS` (`[tabddpm,great]`), and `EXTRA_OVERRIDES` (a list of extra Hydra overrides such as `max_rows=20000`, `budget.epochs=100`).

## CUDA torch: why not `UV_INDEX_PYTORCH_CPU_URL`

`workers/synthcity/pyproject.toml` pins torch to a named, explicit CPU index. uv has no environment variable that overrides a named index's URL (only `UV_INDEX`, `UV_DEFAULT_INDEX`, `UV_INDEX_<NAME>_USERNAME/PASSWORD`), so such a variable would silently be ignored and CPU torch would be installed. The notebook instead:

- creates the env normally (`torch==2.2.2+cpu`),
- runs `uv pip install --python <venv>/bin/python torch==2.2.2 --index https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match --reinstall-package torch` (the `nvidia-*` dependencies come from PyPI),
- sets `UV_NO_SYNC=1`. `envrun.sh` uses `uv run --project`, which re-syncs the env to the lockfile and would revert torch to CPU; `UV_NO_SYNC` (same as `uv run --no-sync`) prevents that.

The repo files are not modified. The notebook asserts `cu121` is in `torch.__version__` and CUDA is available, so a wrong build fails early.

## Time estimates

GPU runtimes are unknown until the first real run. TabDDPM carried roughly 5 minutes of fixed overhead in a CPU spike, and GReaT (LLM fine-tuning) is expected to dominate. Kaggle sessions are limited to a few hours (about 9 h; GPU quota per week), Colab free tiers are shorter, so lower `budget.epochs` or the seed list via `EXTRA_OVERRIDES` if needed. Each job is also bounded by `budget.time_cap_seconds`.

## Importing the results

Download the zip (Kaggle: Output tab / file browser; Colab: Files panel). Locally:

```bash
bash scripts/envrun.sh eval python experiments/run.py -o datasets=adult import-results path/to/adult-gpu.zip
```

The dataset must be registered locally with the same `max_rows` as in the notebook run (the import checks a dataset fingerprint; `--force` overrides). Imported jobs then appear in run `<dataset>-gpu` for `report`.
