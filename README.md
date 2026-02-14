# Data Warehouse for Long-Horizon Reservoir DRL Study

This folder contains the openly shared, study-specific artifacts for the manuscript:

`Long-Horizon Reinforcement Learning for Reservoir Operation`

Repository (public): `https://github.com/lovemevol/Long-Horizon-Paper.git`

## 1. Scope of this release

This release provides:
- Synthetic inflow generation code and example outputs used in the study.
- DRL experiment configuration scripts used to define the paper's training settings.

This release does not provide:
- The full internal DRL environment/system codebase.
- Restricted historical inflow observations from the source authority.

## 2. Folder structure

- `synthetic_inflow/`
- `DRL_training_config/`

### 2.1 `synthetic_inflow/`

Includes STL-AR-Markov based synthetic inflow generation code and outputs:
- `main.py`: single-run generator entry.
- `batch_run.py`: batch experiment runner for the paper-style settings.
- `test_batch.py`: quick test entry.
- `output_batch/`: shared outputs related to this study.
  - `baseline/`
  - `enhanced_flow/`
  - `comparison/`
  - `all_results.json`

### 2.2 `DRL_training_config/`

Includes study-related configuration scripts for DRL training:
- `ppo_*.py`: experiment configuration scripts.
- `1_train_seed.py`: multi-seed launcher.

Important: these files are configuration-level artifacts and depend on an external DRL environment/framework that is not included in this release.

## 3. Reproducing synthetic inflow generation

### 3.1 Environment

Recommended:
- Python `>=3.10`

Install dependencies:

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn openpyxl
```

### 3.2 Input data requirement

Place an inflow Excel file at:
- `synthetic_inflow/实测径流.xlsx`

Expected format:
- Two columns in the first sheet: `date`, `flow` (no mandatory header).
- Daily records; leap-day rows can exist (the script removes Feb 29 internally).
- Missing flow values are interpolated by the script.

Note:
- The restricted original historical observations are not distributed here.
- Users can replace this file with authorized or anonymized data in the same format.

### 3.3 Run commands

From `synthetic_inflow/`:

```bash
python main.py
```

or batch mode:

```bash
python batch_run.py
```

Optional quick test:

```bash
python test_batch.py
```

### 3.4 Main outputs

Outputs are written under:
- `synthetic_inflow/output/` (single run) or
- `synthetic_inflow/output_batch/` (batch run)

Key files:
- `synthetic_inflow_series.csv`
- `statistical_comparison.csv`
- `summary.json`
- diagnostic figures (`*.png`)

## 4. Using DRL training configurations

`DRL_training_config` provides the exact paper-related hyperparameter and setting files.

These scripts are intended for:
- Method inspection.
- Hyperparameter comparison.
- Reuse in compatible external DRL reservoir systems.

They are not standalone runnable with this folder alone because they import external modules (for example, environment entry points and training pipelines) that are outside this release.

If a compatible host framework is available, typical invocation is:

```bash
python DRL_training_config/ppo_c_his.py --seed 10
python DRL_training_config/1_train_seed.py --runs 5 --interval 5
```

## 5. Reproducibility statement

What can be reproduced from this release:
- Synthetic inflow generation workflow and its study outputs (`output_batch`).
- Paper-related DRL parameterization and experiment settings.

What cannot be fully reproduced from this release alone:
- End-to-end DRL training and evaluation without the external reservoir DRL environment/system.

## 6. Why only configuration files are shared for DRL

The full DRL environment belongs to a larger ongoing internal research system that also contains additional unpublished components not specific to this manuscript.

To balance openness and research governance, this release shares:
- All manuscript-relevant training configurations.
- Synthetic data generation pipeline.
- Study outputs needed for methodological reference and comparison.

## 7. Contact

For questions about data format mapping, configuration interpretation, or reproducibility scope, please contact:
- Email: `zaichao2027@163.com`
