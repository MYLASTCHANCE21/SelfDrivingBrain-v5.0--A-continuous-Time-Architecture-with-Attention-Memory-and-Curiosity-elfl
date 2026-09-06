# SelfDrivingBrain v5.0

A NumPy-based continuous-time learning architecture for adaptive control. SelfDrivingBrain combines coupled latent dynamics, reward-modulated plasticity, eligibility traces, prioritized replay, optional attention, curiosity, discrete and continuous action spaces, and evolutionary mutation in one inspectable Python implementation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The repository includes the complete implementation in self_driving_brain.py and the runtime dependencies in requirements.txt.

## What it contains

- **Continuous-time latent dynamics** for the internal state vectors S, A, E, C, D, M, and Psi, integrated with Euler or RK4 steps.
- **Adaptive working-memory damping** through the trainable matrix W_M_damp. The memory dynamics use the learned damping term tanh(W_M_damp @ D) rather than a fixed omega coefficient.
- **Dual learning** with reward-modulated Hebbian updates and optional supervised action-head updates.
- **Eligibility traces and prioritized replay** for temporal credit assignment and importance-weighted replay gradients.
- **Optional multi-head latent attention** over recent states.
- **Curiosity, novelty, uncertainty, and optional goal conditioning.**
- **Discrete and continuous action environments** for local demonstrations.
- **OODA control loop**: Observe, Orient, Decide, and Act.
- **Evolutionary mutation** of the learned model, including W_M_damp.
- **Checkpoint saving** through NumPy .npz files.
- **Diagnostic numerical checks** that fail fast by default when NaN or infinity appears.

## Adaptive damping

The working-memory state M is updated using the learned damping signal:

~~~text
dM = chi * (Q * A) - tanh(W_M_damp @ D) * M + M @ P_op
~~~

W_M_damp is a real trainable parameter. It has its own Adam optimizer, eligibility trace, reward-modulated gradient, replay gradient, evolutionary inheritance and mutation path, and checkpoint entry.

The legacy omega value remains in the parameter dictionary for compatibility and reference, but it is not used in the active working-memory damping equation.

## Reproducibility

Use one master seed and pass it to every stochastic component:

~~~python
MASTER_SEED = 42

config = Config()
config.seed = MASTER_SEED
brain = config.build()

env = HiddenPatternEnv(seed=MASTER_SEED)
evolution = EvolutionEngine(seed=MASTER_SEED)
~~~

The current demo routes the seed through brain initialization, attention initialization, replay sampling, spectral normalization, environments, gradient noise, and evolutionary population generation. Results can still vary across different Python, NumPy, or BLAS implementations because of floating-point behavior.

## Numerical safety

Numerical checks are fail-fast by default:

~~~python
brain = SelfDrivingBrain(
    latent_dim=16,
    seed=42,
    fail_on_numerics=True,
)
~~~

When an invalid value is found, the diagnostic reports the affected state or parameter, learning step, integration step, count of invalid values, and sample indices.

For an explicit recovery mode in a long-running experiment:

~~~python
brain = SelfDrivingBrain(
    latent_dim=16,
    seed=42,
    fail_on_numerics=False,
)
~~~

Recovery mode replaces invalid values with bounded finite values and then applies state safety clipping. Fail-fast mode is recommended during development and experiments.

## Installation

~~~bash
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy matplotlib
~~~

## Quick start

~~~python
from self_driving_brain import (
    Config,
    AdaptiveAgent,
    HiddenPatternEnv,
)

MASTER_SEED = 42

config = Config()
config.seed = MASTER_SEED
config.latent_dim = 16
config.input_dim = 8
config.action_dim = 4
config.use_attention = True
config.use_curiosity = True

brain = config.build()
agent = AdaptiveAgent(brain)
env = HiddenPatternEnv(
    input_dim=8,
    action_count=4,
    horizon=50,
    seed=MASTER_SEED,
)

reward = agent.run_episode(env, steps=50)
print(f"Episode reward: {reward:.2f}")
~~~

To run the full built-in demonstration:

~~~bash
python self_driving_brain.py
~~~

The demo covers discrete control, continuous control, evolutionary mutation, a 200-episode training loop, reward plotting, and checkpoint creation.

## Checkpoint saving

The trained checkpoint includes the adaptive damping matrix:

~~~python
np.savez(
    "final_optimized_brain.npz",
    Phi=brain.Phi,
    W=brain.W,
    M_op=brain.M_op,
    P_op=brain.P_op,
    W_in=brain.W_in,
    W_pred=brain.W_pred,
    b_pred=brain.b_pred,
    W_act=brain.W_act,
    b_act=brain.b_act,
    W_M_damp=brain.W_M_damp,
)
~~~

Load it with NumPy when restoring model parameters:

~~~python
checkpoint = np.load("final_optimized_brain.npz")
W_M_damp = checkpoint["W_M_damp"]
~~~

## Project structure

~~~text
self_driving_brain.py       Main implementation and demonstrations
requirements.txt           Python runtime dependencies
final_optimized_brain.npz  Optional trained checkpoint generated by the demo
episode_rewards.png        Optional training curve generated by the demo
LICENSE                    MIT license
README.md                  Project documentation
~~~

## Design notes

This project is intentionally inspectable rather than an opaque end-to-end neural-network package. The dynamical state, adaptive damping, reward signal, eligibility traces, replay gradients, optimizer steps, and evolutionary mutation are all visible in the source.

The architecture is an experimental learning system, not a production safety controller. Validate behavior in a controlled environment before connecting it to real-world hardware or autonomous decisions.

## License

This project is released under the MIT License. See LICENSE for details.
