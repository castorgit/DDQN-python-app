# DDQN-project

This repository contains a scaffold for Double-DQN experiments.

Layout

- `src/` — main package modules
  - `agent.py` — `ReplayBuffer` and `DDQNAgent` implementations
  - `training.py` — training loop and orchestration
  - `evaluation.py` — run and record episodes
  - `config.py` — YAML config loader
  - `utils/logger.py` — simple experiment logger
- `experiments/` — runnable experiment scripts and YAML configs
  - `run_experiment.py` — run an experiment from a YAML config
  - `configs/` — example YAML configurations
- `notebooks/` — analysis and video generation notebooks
- `logs/` — saved experiment outputs and models

Quickstart

1. Create/activate venv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run an experiment with an example config:

```bash
python experiments/run_experiment.py --config experiments/configs/configuration1.yaml --out logs
```

3. Open notebooks to analyze results and generate videos.

Notes

- The package is set up to lazy-import heavy libraries (torch/gymnasium) in runtime
  so that tests and imports remain lightweight.
- The `ExperimentLogger` creates a timestamped directory under the provided `out` base.
- Models and metadata are saved under `logs/<config>_<timestamp>/agent/` and
  training logs under `logs/<config>_<timestamp>/training_logs/`.

Cleaning

- The repository may accumulate experiment outputs and Python bytecode caches
  in `logs/`, `experiments/logs/` and `__pycache__` directories. These files are
  ignored by Git but can use disk space. A helper script is provided to remove
  these artifacts when you want to reclaim space or start fresh.

Run the cleanup tool from the project root:

```bash
chmod +x scripts/clean_logs.sh
./scripts/clean_logs.sh
```

The script prompts for confirmation before deleting files. Alternatively you
can remove files manually with `rm -rf logs/ experiments/logs/` and using
`find ... -name __pycache__ -delete` to remove caches.

Next steps

- Add richer experiment tracking (TensorBoard, Weights & Biases).
- Add CLI options to tune all hyperparameters and resume training.
