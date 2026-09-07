# A1 preflight artifact

This reviewer-safe bundle records the source-data contract, one-episode
conversion, and GPU memory pilot used to freeze the A1 training configuration.
It intentionally excludes raw HDF5/video data and pilot checkpoint weights.

The long run is resumable and is not represented as complete here. After the
20,000-step checkpoint and H16/H8 evaluation finish, the final reviewer bundle
must be added under `artifacts/A1/full-benchmark/` together with SHA-256 inventory
for excluded raw artifacts.

Run the entire long pipeline from the repository root:

```bash
scripts/run_a1_pipeline.sh
```

Or run its stages separately:

```bash
A1_CONVERSION_WORKERS=10 scripts/run_a1_prepare_dataset.sh
PYTHONPATH="$PWD/src" .venv/bin/python -m unitree_gr00t.cli a1-train --execute
scripts/run_a1_full_benchmark.sh
```
