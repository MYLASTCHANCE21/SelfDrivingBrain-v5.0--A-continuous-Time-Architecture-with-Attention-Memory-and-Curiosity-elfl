# SelfDrivingBrain v5.0

A biologically‑inspired, self‑organising learning agent that combines dynamical systems, Hebbian plasticity, eligibility traces, prioritised experience replay, self‑attention, curiosity, and evolutionary optimisation — all in pure NumPy.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **Recurrent neural dynamics** – internal state variables (`S`, `A`, `E`, `C`, `D`, `M`, `Ψ`) evolve via coupled ODEs.  
- **Dual learning** – reward‑modulated Hebbian updates and supervised learning for the action head.  
- **Eligibility traces** – credit assignment over time.  
- **Prioritised experience replay** – stabilises learning with importance sampling.  
- **Self‑attention** – multi‑head attention over recent states (optional).  
- **Curiosity & novelty** – intrinsic reward drives exploration.  
- **Goal conditioning** – optional external goals.  
- **Discrete & continuous action spaces** – works with both.  
- **Evolutionary engine** – simple population‑based optimisation.  
- **Built‑in OODA loop** – Observe‑Orient‑Decide‑Act controller.  
- **Model saving/loading** – via `np.savez` / `np.load`.

---

## Installation

```bash
git clone https://github.com/MYLASTCHANCE21/SelfDrivingBrain-v5.0--A-continuous-Time-Architecture-with-Attention-Memory-and-Curiosity-elfl.git
cd SelfDrivingBrain-v5.0--A-continuous-Time-Architecture-with-Attention-Memory-and-Curiosity-elfl
python3 -m venv venv
source venv/bin/activate    # or `.\venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

---

## Quick usage

```python
from self_driving_brain import Config, AdaptiveAgent, HiddenPatternEnv, run_episode_scaled, SelfDrivingBrain
import numpy as np

# 1. Configure the brain
config = Config()
config.latent_dim = 16
config.input_dim = 8
config.action_dim = 4
config.use_attention = True
config.use_curiosity = True

brain = config.build()

# 2. Wrap in an adaptive agent
agent = AdaptiveAgent(brain)

# 3. Run a discrete episode
env = HiddenPatternEnv(input_dim=8, action_count=4, horizon=50)
total_reward = agent.run_episode(env, steps=50)
print(f"Reward: {total_reward:.2f}")

# Continuous demo
from self_driving_brain import ContinuousTorqueEnv
cont_env = ContinuousTorqueEnv(obs_dim=8, action_dim=4)
cont_agent = AdaptiveAgent(SelfDrivingBrain(latent_dim=16, input_dim=8, action_dim=4))
cont_total = cont_agent.run_episode(cont_env, steps=50)
print(f"Continuous reward: {cont_total:.2f}")

# Using the scaled episode runner (recommended for learning)
episode_rewards = []
for ep in range(200):
    total = run_episode_scaled(agent, env, steps=300, reward_scale=5.0)
    episode_rewards.append(total)

# Save the trained brain
np.savez("trained_brain.npz",
         Phi=brain.Phi, W=brain.W, M_op=brain.M_op, P_op=brain.P_op,
         W_in=brain.W_in, W_pred=brain.W_pred, b_pred=brain.b_pred,
         W_act=brain.W_act, b_act=brain.b_act)
```

---

## Evolution & Sensitivity analysis

```python
from self_driving_brain import EvolutionEngine, run_sensitivity_analysis, plot_sensitivity_results, HiddenPatternEnv

# Evolutionary optimisation
evo = EvolutionEngine(population_size=16, latent_dim=16, input_dim=8, action_dim=4)
history = evo.evolve(lambda: HiddenPatternEnv(seed=123), generations=10, steps=100)

# Sensitivity analysis example
values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
avgs = run_sensitivity_analysis(
    param_name="hebbian_lr",
    param_values=values,
    base_config_params={"latent_dim": 16, "input_dim": 8, "action_dim": 4},
    num_evaluation_steps=500,
    env_factory=lambda: HiddenPatternEnv(seed=42)
)
plot_sensitivity_results(values, avgs, "hebbian_lr")
```

---

## Notes & suggestions

- The project is pure NumPy (no GPU required).  
- For headless servers or CI, the patched script sets the non‑interactive matplotlib backend (`matplotlib.use("Agg")`).  
- Recommended requirements (requirements.txt):
  - numpy>=1.22
  - matplotlib>=3.5
