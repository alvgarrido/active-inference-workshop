# Active inference workshop

Two Jupyter tutorials exploring variational free energy, belief updating, and decision making with active inference.

## Contents

| File | Purpose |
| --- | --- |
| `tutorial_1_pymdp.ipynb` | Compute posterior beliefs, surprise, and variational free energy in a categorical model, then minimize free energy with automatic differentiation. |
| `tutorial_2_two_step_task.ipynb` | Simulate an active inference agent in a two-step task, inspect its beliefs and choices, and compare reward, belief updating, and surprise across agents. |
| `models.py` | The `learn_and_act` agent, with active inference and reinforcement learning implementations. |
| `utils/twostep_environment.py` | Generate two-step environments with drifting or changing reward and transition probabilities. |
| `utils/twostep_support.py` | Probability utilities, plotting, behavioral summaries, and MATLAB/pickle file helpers. |
| `requirements.txt` | Scientific Python dependencies and Jupyter tooling. |

## Setup

Use Python 3.10-3.12 (Python 3.12 recommended). From the repository root, create and activate a virtual environment:

```sh
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```sh
source .venv/bin/activate
```

Install the dependencies and start JupyterLab:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open the tutorials in order and run their cells from top to bottom using the environment's Python kernel. The second tutorial includes a 50-agent, 200-trial experiment, which takes longer than the introductory examples.

Launch Jupyter from this directory so Python can find `models.py` and `utils/`. The two-step tutorial imports the local agent directly; that agent imports the local environment and support functions. No additional repository is downloaded. If you see `ModuleNotFoundError: models`, restart Jupyter from the repository root and check your selected kernel.

The first tutorial still uses the external `pymdp` library, installed as `inferactively-pymdp`. Its `pymdp.utils` module is separate from this repository's `utils/` directory. Requirements pin the original NumPy-based pymdp release and keep NumPy below 2 for compatibility with the existing simulation code.

## Explore the models

In the second tutorial, edit `task_dict` to change trial count, reward bounds, or volatility. Edit `model_dict` to vary learning rate (`lr`), preference precision (`lam`), forgetting (`vunsamp`, `vsamp`), or action precision (`gamma1`, `gamma2`). Rerun the agent creation and simulation cells after changing parameters.

```python
from models import learn_and_act

# Use task_dict and model_dict defined in tutorial_2_two_step_task.ipynb.
agent = learn_and_act(task=task_dict, model=model_dict, seed=1)
actions, observations, beliefs, transitions, rewards, action_probs = agent.perform_task()
```

`models.py` also implements reinforcement learning with model-based/model-free weighting and SARSA updates; its class docstring describes the required parameters. The supplied second tutorial demonstrates the active inference configuration.

## Acknowledgments

The first tutorial is adapted from the [pymdp variational free energy tutorial](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/free_energy_calculation.html). The original two-step notebook referenced [Garrid0/ActiveInferenceWorkshop](https://github.com/Garrid0/ActiveInferenceWorkshop); this checkout supplies its own model and utility modules for local execution.
