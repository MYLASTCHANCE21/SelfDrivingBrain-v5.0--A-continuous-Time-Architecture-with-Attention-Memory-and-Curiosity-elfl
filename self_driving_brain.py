from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers / CI
import matplotlib.pyplot as plt


# ======================================================================
# CONFIGURATION
# ======================================================================

class Config:
    def __init__(self):
        self.latent_dim = 8
        self.input_dim = None
        self.action_dim = None
        self.seed = None

        self.lr = 3e-3
        self.hebbian_lr = 1e-3
        self.prediction_lr = 1e-2
        self.meta_lr = 1e-3

        self.eligibility_decay = 0.95
        self.reward_baseline_decay = 0.99

        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        self.gradient_noise_std = 0.01
        self.spectral_target = 1.5

        self.enable_prediction = True
        self.enable_action_head = True
        self.enable_memory = True
        self.memory_size = 500
        self.enable_meta_learning = False

        self.use_attention = False
        self.attention_heads = 4
        self.attention_strength = 0.02
        self.use_curiosity = True
        self.use_goals = False
        self.attention_history_len = 10

        # NEW: Reward clipping range and supervised scale max
        self.reward_clip_range = 1.0
        self.supervised_scale_max = 1.0
        self.fail_on_numerics = True

    def build(self):
        return SelfDrivingBrain(
            latent_dim=self.latent_dim,
            seed=self.seed,
            lr=self.lr,
            hebbian_lr=self.hebbian_lr,
            eligibility_decay=self.eligibility_decay,
            reward_baseline_decay=self.reward_baseline_decay,
            meta_lr=self.meta_lr,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            enable_meta_learning=self.enable_meta_learning,
            enable_memory=self.enable_memory,
            memory_size=self.memory_size,
            enable_prediction=self.enable_prediction,
            prediction_lr=self.prediction_lr,
            gradient_noise_std=self.gradient_noise_std,
            spectral_target=self.spectral_target,
            input_dim=self.input_dim,
            action_dim=self.action_dim,
            enable_action_head=self.enable_action_head,
            use_attention=self.use_attention,
            attention_heads=self.attention_heads,
            attention_strength=self.attention_strength,
            use_curiosity=self.use_curiosity,
            use_goals=self.use_goals,
            attention_history_len=self.attention_history_len,
            reward_clip_range=self.reward_clip_range,           # NEW
            supervised_scale_max=self.supervised_scale_max,     # NEW
            fail_on_numerics=self.fail_on_numerics,
        )


# ======================================================================
# NUMERICAL UTILITIES
# ======================================================================

def stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


def cosine_similarity(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a) + 1e-12
    norm_b = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (norm_a * norm_b))


# ======================================================================
# ADAM OPTIMIZER
# ======================================================================

class _Adam:
    def __init__(self, shape, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.base_lr = self.lr = float(lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.wd = weight_decay
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, param, grad):
        self.t += 1
        if self.wd > 0:
            grad = grad + self.wd * param
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m[:] = 0
        self.v[:] = 0
        self.t = 0


# ======================================================================
# SPECTRAL NORMALISATION
# ======================================================================

def _spectral_normalize(W, target=1.0, eps=1e-6, rng=None):
    if W.size == 0:
        return W
    # Use the caller's seeded generator; the fixed fallback keeps direct
    # standalone calls deterministic without touching NumPy's global RNG.
    rng = rng or np.random.default_rng(0)
    u = rng.standard_normal(W.shape[0])
    for _ in range(3):
        v = W.T @ u
        v /= np.linalg.norm(v) + eps
        u = W @ v
        u /= np.linalg.norm(u) + eps
    sigma = float(u @ W @ v)
    if sigma > target:
        return W * (target / sigma)
    return W


# ======================================================================
# LATENT ATTENTION
# ======================================================================

class LatentAttention:
    def __init__(self, latent_dim, heads=4, seed=None):
        self.latent_dim = latent_dim
        self.heads = heads
        rng = np.random.default_rng(seed)
        self.Wq = rng.standard_normal((heads, latent_dim, latent_dim)) * 0.02
        self.Wk = rng.standard_normal((heads, latent_dim, latent_dim)) * 0.02
        self.Wv = rng.standard_normal((heads, latent_dim, latent_dim)) * 0.02
        self.Wo = rng.standard_normal((heads * latent_dim, latent_dim)) * 0.02

    def forward(self, states):
        states = np.asarray(states, dtype=float)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        attended_heads = []
        for h in range(self.heads):
            Q = states @ self.Wq[h]
            K = states @ self.Wk[h]
            V = states @ self.Wv[h]
            scores = Q @ K.T
            weights = stable_softmax(scores, axis=-1)
            head_out = weights @ V
            attended_heads.append(head_out)
        combined = np.concatenate(attended_heads, axis=-1)
        pooled = np.mean(combined, axis=0)
        return np.tanh(self.Wo.T @ pooled)


# ======================================================================
# MEMORY SYSTEMS
# ======================================================================

class WorkingMemory:
    def __init__(self, capacity=32):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, item):
        self.buffer.append(np.array(item, copy=True))

    def get(self):
        return np.array(self.buffer) if self.buffer else None

    def clear(self):
        self.buffer.clear()


@dataclass
class MemoryEpisode:
    state: np.ndarray
    action: np.ndarray
    reward: float
    surprise: float


class EpisodicMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def store(self, state, action, reward, surprise):
        if len(self.memory) >= self.capacity:
            self.memory.popleft()
        self.memory.append(MemoryEpisode(
            np.array(state, copy=True),
            np.array(action, copy=True),
            float(reward),
            float(surprise)
        ))

    def recall(self, state, k=5):
        if not self.memory:
            return []
        scores = [cosine_similarity(state, item.state) for item in self.memory]
        idx = np.argsort(scores)[-k:]
        return [self.memory[i] for i in idx]

    def __len__(self):
        return len(self.memory)


class SemanticMemory:
    def __init__(self):
        self.concepts = {}

    def update(self, key, vector, rate=0.05):
        vector = np.asarray(vector, dtype=float)
        if key not in self.concepts:
            self.concepts[key] = vector.copy()
        else:
            self.concepts[key] = (1 - rate) * self.concepts[key] + rate * vector

    def retrieve(self, key):
        return self.concepts.get(key, None)

    def keys(self):
        return list(self.concepts.keys())


# ======================================================================
# PRIORITISED EXPERIENCE REPLAY
# ======================================================================

@dataclass
class ReplayItem:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    priority: float
    reward_delta: float
    elig_Phi: np.ndarray
    elig_W: np.ndarray
    elig_Mop: np.ndarray
    elig_Pop: np.ndarray
    elig_Win: np.ndarray
    elig_Wact: np.ndarray
    elig_bact: np.ndarray
    elig_Wmdamp: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, rng=None):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self._rng = rng if rng is not None else np.random.default_rng(0)

    def add(self, state, action, reward, next_state, reward_delta,
            elig_Phi, elig_W, elig_Mop, elig_Pop, elig_Win, elig_Wact, elig_bact,
            elig_Wmdamp):
        priority = abs(reward_delta) + 1e-5
        self.buffer.append(ReplayItem(
            np.asarray(state, dtype=float).copy(),
            np.asarray(action, dtype=float).copy(),
            float(reward),
            np.asarray(next_state, dtype=float).copy(),
            float(priority),
            float(reward_delta),
            np.asarray(elig_Phi, dtype=float).copy(),
            np.asarray(elig_W, dtype=float).copy(),
            np.asarray(elig_Mop, dtype=float).copy(),
            np.asarray(elig_Pop, dtype=float).copy(),
            np.asarray(elig_Win, dtype=float).copy(),
            np.asarray(elig_Wact, dtype=float).copy(),
            np.asarray(elig_bact, dtype=float).copy(),
            np.asarray(elig_Wmdamp, dtype=float).copy(),
        ))

    def sample(self, batch_size, rng=None):
        if not self.buffer:
            return [], np.array([]), np.array([])
        rng = rng if rng is not None else self._rng
        priorities = np.array([x.priority for x in self.buffer], dtype=float)
        priorities = np.clip(priorities, 1e-12, None)
        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()
        n = len(self.buffer)
        size = min(batch_size, n)
        indices = rng.choice(n, size=size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / (weights.max() + 1e-12)
        return samples, indices, weights

    def update_priority(self, index, priority):
        if 0 <= index < len(self.buffer):
            self.buffer[index].priority = float(priority)

    def __len__(self):
        return len(self.buffer)


# ======================================================================
# UNCERTAINTY, CURIOSITY, GOALS
# ======================================================================

class UncertaintyEstimator:
    def __init__(self, latent_dim, history=100):
        self.latent_dim = latent_dim
        self.errors = deque(maxlen=history)

    def update(self, prediction, actual):
        error = float(np.mean((prediction - actual) ** 2))
        self.errors.append(error)
        return error

    def confidence(self):
        return 0.0 if not self.errors else float(np.exp(-np.mean(self.errors)))

    def value(self):
        return 1.0 if not self.errors else float(np.mean(self.errors))


class CuriosityEngine:
    def __init__(self, decay=0.99):
        self.decay = decay
        self.baseline_error = 1.0
        self.visited = deque(maxlen=1000)

    def novelty(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)
        if not self.visited:
            self.visited.append(state.copy())
            return 1.0
        sims = [cosine_similarity(state, old) for old in self.visited]
        self.visited.append(state.copy())
        return float(1.0 - np.mean(sims))

    def reward(self, state, prediction_error):
        nov = self.novelty(state)
        improvement = self.baseline_error - prediction_error
        self.baseline_error = self.decay * self.baseline_error + (1 - self.decay) * prediction_error
        return float(0.7 * nov + 0.3 * np.tanh(improvement))


@dataclass
class Goal:
    name: str
    target: np.ndarray
    weight: float


class GoalManager:
    def __init__(self):
        self.goals = []

    def add_goal(self, name, target, weight=1.0):
        self.goals.append(Goal(name, np.asarray(target, dtype=float), float(weight)))

    def evaluate(self, state):
        state = np.asarray(state, dtype=float)
        if not self.goals:
            return 0.0
        score = 0.0
        for g in self.goals:
            if state.shape != g.target.shape:
                raise ValueError(f"State shape {state.shape} does not match goal target shape {g.target.shape}")
            dist = np.linalg.norm(state - g.target)
            score += g.weight * np.exp(-dist)
        return float(score)

    def best_goal(self):
        return max(self.goals, key=lambda x: x.weight) if self.goals else None


# ======================================================================
# SELFDRIVINGBRAIN – CORE
# ======================================================================

class SelfDrivingBrain:
    def __init__(
        self,
        latent_dim: int = 8,
        seed: Optional[int] = None,
        lr: float = 3e-3,
        hebbian_lr: float = 1e-3,
        eligibility_decay: float = 0.95,
        reward_baseline_decay: float = 0.99,
        meta_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        enable_meta_learning: bool = False,
        enable_memory: bool = True,
        memory_size: int = 500,
        enable_prediction: bool = True,
        prediction_lr: float = 1e-2,
        gradient_noise_std: float = 0.01,
        spectral_target: float = 1.5,
        input_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        enable_action_head: bool = True,
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_strength: float = 0.02,
        use_curiosity: bool = True,
        use_goals: bool = False,
        attention_history_len: int = 10,
        reward_clip_range: float = 1.0,         # NEW
        supervised_scale_max: float = 1.0,      # NEW
        fail_on_numerics: bool = True,
    ):
        self.d = int(latent_dim)
        self.input_d = input_dim if input_dim is not None else self.d
        self.action_d = action_dim if action_dim is not None else self.d

        self._rng = np.random.default_rng(seed)
        self.action_low = -1.0 * np.ones(self.action_d)
        self.action_high = 1.0 * np.ones(self.action_d)

        self.Phi = self._rng.standard_normal((self.d, self.d)) * 0.1
        self.M_op = self._rng.standard_normal((self.d, self.d)) * 0.05
        self.P_op = self._rng.standard_normal((self.d, self.d)) * 0.05
        self.W = self._rng.standard_normal((self.d, self.d)) * 0.05
        self.W_in = self._rng.standard_normal((self.d, self.input_d)) * 0.1
        self.W_pred = self._rng.standard_normal((self.d, self.d)) * 0.05
        self.b_pred = np.zeros(self.d)
        self.W_act = self._rng.standard_normal((self.action_d, self.d)) * 0.05
        self.b_act = np.zeros(self.action_d)
        # Learned adaptive M (working memory) damping — replaces hardcoded omega
        self.W_M_damp = self._rng.standard_normal((self.d, self.d)) * 0.05

        self.h = {
            "alpha": 1.1, "beta": 0.8, "sigma": 1.2, "tau": 0.6,
            "eta": 1.3, "xi": 0.9, "zeta": 0.7,
            "mu": 0.5, "nu": 0.4, "rho": 0.6,
            "chi": 1.0, "omega": 0.5,
        }
        self._h_keys = sorted(self.h.keys())

        self.kappa = 0.15
        self.S_star = 0.0
        self.curiosity_gain = 0.05
        self.max_dS = 0.5
        self.boundary_repel = 2.0
        self.boundary_threshold = 3.5
        self.spectral_target = float(spectral_target)

        self.lr = float(lr)
        self.hebbian_lr = float(hebbian_lr)
        self.eligibility_decay = float(eligibility_decay)
        self.reward_baseline_decay = float(reward_baseline_decay)
        self.meta_lr = float(meta_lr)
        self.grad_clip = float(grad_clip)
        self.enable_meta_learning = bool(enable_meta_learning)
        self.gradient_noise_std = float(gradient_noise_std)
        self.weight_decay = float(weight_decay)
        self.enable_prediction = bool(enable_prediction)
        self.enable_action_head = bool(enable_action_head)

        self._opt_Phi = _Adam(self.Phi.shape, lr, weight_decay=weight_decay)
        self._opt_W = _Adam(self.W.shape, lr, weight_decay=weight_decay)
        self._opt_Mop = _Adam(self.M_op.shape, lr, weight_decay=weight_decay)
        self._opt_Pop = _Adam(self.P_op.shape, lr, weight_decay=weight_decay)
        self._opt_Win = _Adam(self.W_in.shape, lr, weight_decay=weight_decay)
        self._opt_Wpred = _Adam(self.W_pred.shape, prediction_lr, weight_decay=weight_decay)
        self._opt_bpred = _Adam(self.b_pred.shape, prediction_lr, weight_decay=weight_decay)
        self._opt_Wact = _Adam(self.W_act.shape, lr, weight_decay=weight_decay)
        self._opt_bact = _Adam(self.b_act.shape, lr, weight_decay=weight_decay)
        self._opt_Wmdamp = _Adam(self.W_M_damp.shape, lr, weight_decay=weight_decay)
        self._opt_h = _Adam((len(self._h_keys),), meta_lr)

        self.use_attention = use_attention
        self.attention_strength = attention_strength
        if use_attention:
            self.attention = LatentAttention(self.d, heads=attention_heads, seed=seed)
            self.attention_history_len = attention_history_len
            self._state_history = deque(maxlen=attention_history_len)

        self.use_curiosity = use_curiosity
        if use_curiosity:
            self.curiosity = CuriosityEngine()

        self.use_goals = use_goals
        if use_goals:
            self.goal_manager = GoalManager()

        self.enable_memory = bool(enable_memory)
        self.memory_size = int(memory_size)
        self._memory_batch_size = 16
        self._replay_buffer = PrioritizedReplayBuffer(
            capacity=memory_size,
            rng=self._rng,
        )

        # NEW: Store reward clipping range and supervised scale max
        self.reward_clip_range = float(reward_clip_range)
        self.supervised_scale_max = float(supervised_scale_max)
        self.fail_on_numerics = bool(fail_on_numerics)

        self.reset(reset_learning=True)

    def reset(self, S=None, A=None, E=None, C=None, D=None, M=None, Psi=None,
              t0=0.0, reset_learning=False):
        d = self.d
        eps = 1e-2
        self.S = np.zeros(d) if S is None else np.array(S, dtype=float).reshape(d)
        self.A = self._rng.normal(0, eps, d) if A is None else np.array(A, dtype=float).reshape(d)
        self.E = np.ones(d) * 0.5 if E is None else np.array(E, dtype=float).reshape(d)
        self.C = np.ones(d) * 0.5 if C is None else np.array(C, dtype=float).reshape(d)
        self.D = np.zeros(d) if D is None else np.array(D, dtype=float).reshape(d)
        self.M = np.zeros(d) if M is None else np.array(M, dtype=float).reshape(d)
        self.Psi = self._rng.normal(0, eps, d) if Psi is None else np.array(Psi, dtype=float).reshape(d)

        self.t = float(t0)
        self._step_count = 0
        self._learn_step_count = 0
        self._last_input = np.zeros(self.input_d)
        self._last_action = np.zeros(self.action_d)
        self._total_reward = 0.0
        self._reward_baseline = 0.0
        self._state_variance_ema = 1.0

        self._history = deque(maxlen=10000)
        if self.use_attention:
            self._state_history.clear()

        e = 1e-3
        self._eligibility_Phi = self._rng.standard_normal(self.Phi.shape) * e
        self._eligibility_W = self._rng.standard_normal(self.W.shape) * e
        self._eligibility_Mop = self._rng.standard_normal(self.M_op.shape) * e
        self._eligibility_Pop = self._rng.standard_normal(self.P_op.shape) * e
        self._eligibility_Win = self._rng.standard_normal(self.W_in.shape) * e
        self._eligibility_Wact = self._rng.standard_normal(self.W_act.shape) * e
        self._eligibility_bact = self._rng.standard_normal(self.b_act.shape) * e
        self._eligibility_Wmdamp = self._rng.standard_normal(self.W_M_damp.shape) * e

        self._pred_error_history = deque(maxlen=100)
        self._reward_history = []
        self._loss_history = []
        self._meta_history = []

        if reset_learning:
            for opt in [
                self._opt_Phi, self._opt_W, self._opt_Mop, self._opt_Pop,
                self._opt_Win, self._opt_Wpred, self._opt_bpred,
                self._opt_Wact, self._opt_bact, self._opt_Wmdamp, self._opt_h
            ]:
                opt.reset()
            self._replay_buffer.buffer.clear()
            self._reward_history.clear()
            self._loss_history.clear()

        return self

    def sense(self, vector: np.ndarray):
        v = np.asarray(vector, dtype=float).reshape(-1)
        if len(v) != self.input_d:
            raise ValueError(f"Expected {self.input_d} inputs, got {len(v)}")
        self._last_input = v.copy()
        self.S += self.W_in @ v
        return self

    def propose_action(self):
        raw = self.W_act @ self.S + self.b_act
        return np.tanh(raw)

    def action_to_index(self, action, available_actions):
        if not available_actions:
            raise ValueError("No available actions")
        action = np.asarray(action, dtype=float).reshape(-1)
        scores = []
        for a in available_actions:
            a = np.asarray(a, dtype=float).reshape(-1)
            m = min(len(a), len(action))
            if m == 0:
                continue
            score = np.dot(action[:m], a[:m]) / (np.linalg.norm(action[:m]) * np.linalg.norm(a[:m]) + 1e-12)
            scores.append(float(score))
        if not scores:
            raise ValueError("No valid actions to compare")
        return int(np.argmax(scores))

    def _generate_internal_drive(self):
        novelty = float(np.mean(np.abs(self.S)))
        I_single = 1.0 / (1.0 + np.exp(-5.0 * (self.curiosity_gain - novelty)))
        I = np.full(self.d, I_single)
        Q = 1.0 / (1.0 + np.exp(-np.clip(self.S, -50, 50)))
        energy = float(np.mean(self.S ** 2))
        G_single = 1.0 / (1.0 + np.exp(-5.0 * (energy - 0.5)))
        G = np.full(self.d, G_single)
        R_drag = 0.1 * np.clip(self.D, 0, 1e3)
        noise_scale = 0.02 * max(0.0, 1.0 - np.mean(self.S ** 2))
        R_stoch = self._rng.normal(0.0, noise_scale, self.d)
        return I, Q, G, R_drag, R_stoch

    def _T1(self, S, Psi): return np.tanh(S * Psi)
    def _T3(self, A, M): return np.tanh(A * M)
    def _N(self, Psi, A, M): return np.tanh(Psi * A * M)

    def compute_derivatives(self, I, Q, G, R_drag, R_stoch):
        p = self.h
        S, A, E, C, D, M, Psi = self.S, self.A, self.E, self.C, self.D, self.M, self.Psi

        learned_damp = np.tanh(self.W_M_damp @ D)  # Learned adaptive damping
        dM = p["chi"] * (Q * A) - learned_damp * M + M @ self.P_op
        dPsi = Psi @ self.M_op + self._N(Psi, A, M) + 0.01 * np.tanh(S)
        c_sat = 1.0 / (1.0 + np.exp(np.clip(C - 5.0, -60, 60)))
        dE = p["sigma"] * (Psi @ self.Phi) - p["tau"] * E * (1.0 + 0.1 * C) + 0.02 * np.ones(self.d)
        dC = p["eta"] * I + p["xi"] * Psi - p["zeta"] * E * (1.0 + 9.0 * c_sat)
        dA = p["alpha"] * (Psi * A) - p["beta"] * (E * A) + (Psi @ self.W) + 0.005 * np.tanh(Psi)
        dD = p["mu"] * (S @ self.Phi) - p["nu"] * S - p["rho"] * G

        T1 = self._T1(S, Psi)
        T2 = S @ self.Phi
        T3 = self._T3(A, M)
        external = T1 + T2 + T3 - R_drag + R_stoch
        homeo = -self.kappa * (S - self.S_star)
        wall = np.zeros_like(S)
        th = self.boundary_threshold
        for i in range(self.d):
            if S[i] > th:
                wall[i] = -self.boundary_repel * (S[i] - th)
            elif S[i] < -th:
                wall[i] = -self.boundary_repel * (S[i] + th)
        dS = external + homeo + wall
        return dS, dA, dE, dC, dD, dM, dPsi

    def _safety_clip_states(self):
        self.S = np.clip(self.S, -4.0, 4.0)
        self.A = np.clip(self.A, -1e6, 1e6)
        self.E = np.clip(self.E, 0.0, 1e6)
        self.C = np.clip(self.C, 0.05, 1e6)
        self.D = np.clip(self.D, 0.0, 1e6)
        self.M = np.clip(self.M, -1e6, 1e6)
        self.Psi = np.clip(self.Psi, -1e6, 1e6)

    def _check_numerics(self):
        arrays = {
            "S": self.S,
            "A": self.A,
            "E": self.E,
            "C": self.C,
            "D": self.D,
            "M": self.M,
            "Psi": self.Psi,
            "Phi": self.Phi,
            "M_op": self.M_op,
            "P_op": self.P_op,
            "W": self.W,
            "W_in": self.W_in,
            "W_pred": self.W_pred,
            "b_pred": self.b_pred,
            "W_act": self.W_act,
            "b_act": self.b_act,
            "W_M_damp": self.W_M_damp,
        }
        for name, arr in arrays.items():
            finite = np.isfinite(arr)
            if not np.all(finite):
                bad_count = int(arr.size - np.count_nonzero(finite))
                bad_indices = np.argwhere(~finite)[:5].tolist()
                message = (
                    f"Numerical instability in {name} at "
                    f"learn_step={self._learn_step_count}, "
                    f"step={self._step_count}: "
                    f"{bad_count} non-finite value(s); "
                    f"sample indices={bad_indices}"
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                if self.fail_on_numerics:
                    raise FloatingPointError(message)
                # Explicit opt-out recovery mode for long-running jobs.
                arr[np.isnan(arr)] = 0.0
                arr[np.isposinf(arr)] = 1e6
                arr[np.isneginf(arr)] = -1e6
        self._safety_clip_states()

    def _normalize_weights(self):
        self.Phi = _spectral_normalize(
            self.Phi, self.spectral_target, rng=self._rng
        )
        self.W = _spectral_normalize(
            self.W, self.spectral_target, rng=self._rng
        )
        self.M_op = _spectral_normalize(
            self.M_op, self.spectral_target, rng=self._rng
        )
        self.P_op = _spectral_normalize(
            self.P_op, self.spectral_target, rng=self._rng
        )
        self.W_act = _spectral_normalize(
            self.W_act, self.spectral_target, rng=self._rng
        )

    def _derivs_from_state(self, S, A, E, C, D, M, Psi):
        backup = (self.S.copy(), self.A.copy(), self.E.copy(), self.C.copy(),
                  self.D.copy(), self.M.copy(), self.Psi.copy())
        self.S, self.A, self.E, self.C, self.D, self.M, self.Psi = S, A, E, C, D, M, Psi
        result = self.compute_derivatives(*self._generate_internal_drive())
        self.S, self.A, self.E, self.C, self.D, self.M, self.Psi = backup
        return result

    def step(self, dt=0.1, integrator="rk4", clip=True):
        if integrator not in ("euler", "rk4"):
            raise ValueError("Integrator must be euler or rk4")

        if self.use_attention and len(self._state_history) > 0:
            attended = self.attention.forward(list(self._state_history))
            self.S += self.attention_strength * attended

        if integrator == "euler":
            deriv = self.compute_derivatives(*self._generate_internal_drive())
        else:
            k1 = self.compute_derivatives(*self._generate_internal_drive())
            k2 = self._derivs_from_state(
                self.S + dt * k1[0] / 2, self.A + dt * k1[1] / 2, self.E + dt * k1[2] / 2,
                self.C + dt * k1[3] / 2, self.D + dt * k1[4] / 2, self.M + dt * k1[5] / 2,
                self.Psi + dt * k1[6] / 2
            )
            k3 = self._derivs_from_state(
                self.S + dt * k2[0] / 2, self.A + dt * k2[1] / 2, self.E + dt * k2[2] / 2,
                self.C + dt * k2[3] / 2, self.D + dt * k2[4] / 2, self.M + dt * k2[5] / 2,
                self.Psi + dt * k2[6] / 2
            )
            k4 = self._derivs_from_state(
                self.S + dt * k3[0], self.A + dt * k3[1], self.E + dt * k3[2],
                self.C + dt * k3[3], self.D + dt * k3[4], self.M + dt * k3[5],
                self.Psi + dt * k3[6]
            )
            deriv = tuple((k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6.0 for i in range(7))

        dS, dA, dE, dC, dD, dM, dPsi = deriv
        norm = np.linalg.norm(dS) + 1e-12
        if norm > self.max_dS:
            dS = dS / norm * self.max_dS

        self.S += dt * dS
        self.A += dt * dA
        self.E += dt * dE
        self.C += dt * dC
        self.D += dt * dD
        self.M += dt * dM
        self.Psi += dt * dPsi

        if clip:
            self._safety_clip_states()
        self._check_numerics()

        self._state_variance_ema = 0.95 * self._state_variance_ema + 0.05 * float(np.var(self.S))
        target_kappa = np.clip(0.08 + 0.25 * self._state_variance_ema, 0.05, 0.5)
        self.kappa = 0.995 * self.kappa + 0.005 * target_kappa

        self.t += dt
        self._step_count += 1
        self._history.append({
            "t": self.t, "S": self.S.copy(), "A": self.A.copy(),
            "E": self.E.copy(), "C": self.C.copy(), "D": self.D.copy(),
            "M": self.M.copy(), "Psi": self.Psi.copy(), "dS_norm": float(norm)
        })

        if self.use_attention:
            self._state_history.append(self.S.copy())

        return self.S

    def _update_prediction(self, S_prev):
        if not self.enable_prediction:
            return
        pred = self.W_pred @ S_prev + self.b_pred
        error = self.S - pred
        self._pred_error_history.append(float(np.mean(error ** 2)))
        grad_W = -np.outer(error, S_prev)
        grad_b = -error
        self.W_pred = self._opt_Wpred.step(self.W_pred, grad_W)
        self.b_pred = self._opt_bpred.step(self.b_pred, grad_b)

    def _prediction_confidence(self):
        if hasattr(self, "_pred_error_history") and len(self._pred_error_history) > 0:
            err = float(np.mean(self._pred_error_history))
            return 1.0 / (1.0 + err)
        return 1.0

    def _action_head_lr_scale(self):
        return 1.0 / (1.0 + 0.001 * self._learn_step_count)

    def _update_eligibility_traces(self, dt):
        decay = self.eligibility_decay ** dt
        self._eligibility_Phi = decay * self._eligibility_Phi + np.outer(self.S, self.Psi)
        self._eligibility_W = decay * self._eligibility_W + np.outer(self.Psi, self.A)
        self._eligibility_Mop = decay * self._eligibility_Mop + np.outer(self.Psi, self.Psi)
        self._eligibility_Pop = decay * self._eligibility_Pop + np.outer(self.M, self.M)
        self._eligibility_Win = decay * self._eligibility_Win + np.outer(self.S, self._last_input)
        self._eligibility_Wact = decay * self._eligibility_Wact + np.outer(self._last_action, self.S)
        self._eligibility_bact = decay * self._eligibility_bact + self._last_action
        self._eligibility_Wmdamp = (
            decay * self._eligibility_Wmdamp
            + np.outer(np.tanh(self.W_M_damp @ self.D) * self.M, self.D)
        )

    def _reward_modulated_grad(self, reward_delta):
        scale = self.hebbian_lr
        return (
            -reward_delta * self._eligibility_Phi * scale,
            -reward_delta * self._eligibility_W * scale,
            -reward_delta * self._eligibility_Mop * scale,
            -reward_delta * self._eligibility_Pop * scale,
            -reward_delta * self._eligibility_Win * scale,
            -reward_delta * self._eligibility_Wact * scale,
            -reward_delta * self._eligibility_bact * scale,
            -reward_delta * self._eligibility_Wmdamp * scale,
        )

    def _store_transition(self, S_prev, reward, reward_delta):
        if not self.enable_memory:
            return
        self._replay_buffer.add(
            state=S_prev,
            action=self._last_action,
            reward=reward,
            next_state=self.S,
            reward_delta=reward_delta,
            elig_Phi=self._eligibility_Phi.copy(),
            elig_W=self._eligibility_W.copy(),
            elig_Mop=self._eligibility_Mop.copy(),
            elig_Pop=self._eligibility_Pop.copy(),
            elig_Win=self._eligibility_Win.copy(),
            elig_Wact=self._eligibility_Wact.copy(),
            elig_bact=self._eligibility_bact.copy(),
            elig_Wmdamp=self._eligibility_Wmdamp.copy(),
        )

    def _replay_memory(self, n_samples=8):
        if len(self._replay_buffer) < n_samples:
            return None
        items, indices, is_weights = self._replay_buffer.sample(n_samples, self._rng)
        if len(items) == 0:
            return None
        accum = {
            "Phi": np.zeros_like(self.Phi),
            "W": np.zeros_like(self.W),
            "M_op": np.zeros_like(self.M_op),
            "P_op": np.zeros_like(self.P_op),
            "W_in": np.zeros_like(self.W_in),
            "W_act": np.zeros_like(self.W_act),
            "b_act": np.zeros_like(self.b_act),
            "W_M_damp": np.zeros_like(self.W_M_damp),
        }
        for item, w in zip(items, is_weights):
            delta = item.reward_delta
            weight = float(w)
            accum["Phi"] += weight * (-delta * item.elig_Phi)
            accum["W"] += weight * (-delta * item.elig_W)
            accum["M_op"] += weight * (-delta * item.elig_Mop)
            accum["P_op"] += weight * (-delta * item.elig_Pop)
            accum["W_in"] += weight * (-delta * item.elig_Win)
            accum["W_act"] += weight * (-delta * item.elig_Wact)
            accum["b_act"] += weight * (-delta * item.elig_bact)
            accum["W_M_damp"] += weight * (-delta * item.elig_Wmdamp)
        scale = self.hebbian_lr / max(1, len(items))
        return {k: v * scale for k, v in accum.items()}

    def _clip_grad(self, g, max_norm):
        n = np.linalg.norm(g)
        if n > max_norm:
            g = g * (max_norm / (n + 1e-12))
        return g

    def _add_gradient_noise(self, g):
        if self.gradient_noise_std <= 0:
            return g
        decay = max(0.0, 1.0 - self._learn_step_count / 10000.0)
        noise = self._rng.normal(0.0, self.gradient_noise_std, g.shape)
        return g + noise * decay

    def _intrinsic_reward(self, S_prev, task_reward=None):
        delta = np.linalg.norm(self.S - S_prev)
        novelty = np.tanh(delta * 2.0)
        homeostasis = np.exp(-np.linalg.norm(self.S - self.S_star))
        entropy = np.tanh(np.var(self.S) * 10.0)
        pred_err = np.mean(list(self._pred_error_history)) if self._pred_error_history else 0.0
        info_gain = np.tanh(abs(np.linalg.norm(self.S) - np.linalg.norm(S_prev)) * 5.0)
        reward = 0.3 * novelty + 0.2 * homeostasis + 0.1 * entropy + 0.2 * info_gain - 0.1 * pred_err
        if task_reward is not None:
            reward += 0.5 * float(task_reward)
        # NEW: clip using reward_clip_range instead of hardcoded 1.0
        return np.clip(reward, -self.reward_clip_range, self.reward_clip_range)

    def learn_step(self, dt=0.1, integrator="rk4", reward_fn=None, action_target=None):
        S_prev = self.S.copy()
        self.step(dt=dt, integrator=integrator)
        self._update_prediction(S_prev)

        task_reward = float(reward_fn(self)) if reward_fn is not None else None
        reward = self._intrinsic_reward(S_prev, task_reward)

        self._reward_baseline = (
            self.reward_baseline_decay * self._reward_baseline
            + (1.0 - self.reward_baseline_decay) * reward
        )
        reward_delta = reward - self._reward_baseline

        self._update_eligibility_traces(dt)
        g_Phi, g_W, g_Mop, g_Pop, g_Win, g_Wact, g_bact, g_Wmdamp = self._reward_modulated_grad(reward_delta)

        replay = self._replay_memory()
        if replay is not None:
            g_Phi += replay["Phi"]
            g_W += replay["W"]
            g_Mop += replay["M_op"]
            g_Pop += replay["P_op"]
            g_Win += replay["W_in"]
            g_Wact += replay["W_act"]
            g_bact += replay["b_act"]
            g_Wmdamp += replay["W_M_damp"]

        conf = 0.0
        anneal = 0.0

        # ---- Supervised gradient (if applicable) ----
        if action_target is not None and reward > 0 and self.enable_action_head:
            conf = self._prediction_confidence()
            anneal = self._action_head_lr_scale()
            # NEW: use supervised_scale_max instead of hardcoded 1.0
            scale = float(np.clip(reward * conf * anneal, 0.0, self.supervised_scale_max))
            if scale > 0:
                target = np.asarray(action_target, dtype=float)
                if target.shape != (self.action_d,):
                    raise ValueError(f"Expected action target shape ({self.action_d},), got {target.shape}")
                out = np.tanh(self.W_act @ self.S + self.b_act)
                error = target - out
                grad_super = (error * (1.0 - out ** 2)) * scale
                g_Wact += -np.outer(grad_super, self.S)   # gradient for W_act
                g_bact += -grad_super                     # gradient for b_act

        # ---- Apply all gradients (RL + supervised combined) ----
        gradients = [
            self._add_gradient_noise(self._clip_grad(g, self.grad_clip))
            for g in (g_Phi, g_W, g_Mop, g_Pop, g_Win, g_Wact, g_bact, g_Wmdamp)
        ]
        g_Phi, g_W, g_Mop, g_Pop, g_Win, g_Wact, g_bact, g_Wmdamp = gradients

        self.Phi = self._opt_Phi.step(self.Phi, g_Phi)
        self.W = self._opt_W.step(self.W, g_W)
        self.M_op = self._opt_Mop.step(self.M_op, g_Mop)
        self.P_op = self._opt_Pop.step(self.P_op, g_Pop)
        self.W_in = self._opt_Win.step(self.W_in, g_Win)
        self.W_act = self._opt_Wact.step(self.W_act, g_Wact)   # single step
        self.b_act = self._opt_bact.step(self.b_act, g_bact)   # single step
        self.W_M_damp = self._opt_Wmdamp.step(self.W_M_damp, g_Wmdamp)

        self._normalize_weights()
        self._check_numerics()
        self._store_transition(S_prev, reward, reward_delta)

        self._total_reward += reward
        self._learn_step_count += 1
        self._reward_history.append(reward)
        self._loss_history.append(-reward)

        return {
            "reward": float(reward),
            "reward_delta": float(reward_delta),
            "loss": float(-reward),
            "learn_step": self._learn_step_count,
            "kappa": float(self.kappa),
            "memory": len(self._replay_buffer),
            "conf": float(conf),
            "anneal": float(anneal),
        }


# ======================================================================
# OODA CONTROLLER, ADAPTIVE AGENT, EVOLUTION ENGINE
# ======================================================================

class OODAController:
    def __init__(self, brain: SelfDrivingBrain):
        self.brain = brain
        self.episode_memory = []

    def observe(self, observation):
        self.brain.sense(observation)
        return self.brain.S.copy()

    def orient(self):
        pred = self.brain.W_pred @ self.brain.S + self.brain.b_pred
        action = self.brain.propose_action()
        return {"state": self.brain.S.copy(), "prediction": pred, "action": action}

    def decide_discrete(self, available_actions):
        proposal = self.brain.propose_action()
        return self.brain.action_to_index(proposal, available_actions)

    def act_discrete(self, env, action_index):
        actions = env.available_actions()
        action = np.asarray(actions[action_index], dtype=float)
        self.brain._last_action = action.copy()
        obs, reward, done, info = env.step(action)
        self.episode_memory.append({"action": action, "reward": reward, "info": info})
        return obs, reward, done, info

    def act_continuous(self, env, action_vector):
        action = np.asarray(action_vector, dtype=float).reshape(-1)
        self.brain._last_action = action.copy()
        obs, reward, done, info = env.step(action)
        self.episode_memory.append({"action": action, "reward": reward, "info": info})
        return obs, reward, done, info


class AdaptiveAgent:
    def __init__(self, brain=None):
        self.brain = brain or SelfDrivingBrain(latent_dim=32, input_dim=8, action_dim=4)
        self.controller = OODAController(self.brain)

    def run_episode(self, env, steps=100):
        obs = env.reset()
        total = 0.0
        mode = getattr(env, "action_mode", "discrete" if hasattr(env, "available_actions") else "continuous")

        for _ in range(steps):
            self.controller.observe(obs)
            orient = self.controller.orient()

            if mode == "discrete":
                idx = self.controller.decide_discrete(env.available_actions())
                obs, reward, done, info = self.controller.act_discrete(env, idx)
                target_vec = np.eye(len(env.available_actions()))[info["target"]]
                self.brain.learn_step(reward_fn=lambda b, r=reward: r, action_target=target_vec)
            elif mode == "continuous":
                action = np.asarray(orient["action"], dtype=float)
                obs, reward, done, info = self.controller.act_continuous(env, action)
                self.brain.learn_step(reward_fn=lambda b, r=reward: r)
            else:
                raise ValueError(f"Unknown action_mode: {mode}")

            total += reward
            if done:
                break

        return float(total)


class EvolutionEngine:
    def __init__(self, population_size=16, latent_dim=32, input_dim=None, action_dim=None, seed=None):
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.population = [
            AdaptiveAgent(SelfDrivingBrain(
                latent_dim=latent_dim,
                input_dim=input_dim,
                action_dim=action_dim,
                seed=int(self.rng.integers(0, 1_000_000))
            ))
            for _ in range(population_size)
        ]

    def evaluate(self, env_factory, steps=100):
        return [agent.run_episode(env_factory(), steps) for agent in self.population]

    def select(self, scores, top_k=None):
        top_k = top_k or max(1, len(scores) // 4)
        winners = np.argsort(scores)[::-1][:top_k]
        return [self.population[i] for i in winners]

    def mutate(self, winners, population_size=None, mutation_scale=0.02):
        pop_size = population_size or len(self.population)
        new_pop = []
        while len(new_pop) < pop_size:
            parent = winners[len(new_pop) % len(winners)]
            child_brain = SelfDrivingBrain(
                latent_dim=parent.brain.d,
                input_dim=parent.brain.input_d,
                action_dim=parent.brain.action_d,
                seed=int(self.rng.integers(0, 1_000_000)),
                use_attention=getattr(parent.brain, "use_attention", False),
                use_curiosity=getattr(parent.brain, "use_curiosity", True),
                use_goals=getattr(parent.brain, "use_goals", False),
                attention_history_len=getattr(parent.brain, "attention_history_len", 10),
                enable_memory=getattr(parent.brain, "enable_memory", True),
                enable_prediction=getattr(parent.brain, "enable_prediction", True),
                enable_action_head=getattr(parent.brain, "enable_action_head", True),
                fail_on_numerics=getattr(parent.brain, "fail_on_numerics", True),
            )
            for name in ["Phi", "W", "M_op", "P_op", "W_in", "W_pred", "b_pred", "W_act", "b_act", "W_M_damp"]:
                setattr(child_brain, name, getattr(parent.brain, name).copy())
            for name in ["Phi", "W", "M_op", "P_op", "W_in", "W_pred", "W_act", "W_M_damp"]:
                val = getattr(child_brain, name)
                val += self.rng.normal(0.0, mutation_scale, val.shape)
            new_pop.append(AdaptiveAgent(child_brain))
        self.population = new_pop
        return new_pop

    def evolve(self, env_factory, generations=10, steps=100):
        history = []
        for g in range(generations):
            scores = self.evaluate(env_factory, steps)
            winners = self.select(scores)
            history.append(
                {
                    "generation": g,
                    "best": float(np.max(scores)),
                    "average": float(np.mean(scores))
                }
            )
            self.mutate(winners)
        return history


# ======================================================================
# DEMO ENVIRONMENTS
# ======================================================================

class HiddenPatternEnv:
    action_mode = "discrete"
    def __init__(self, input_dim=8, action_count=4, horizon=50, seed=None):
        self.input_dim = input_dim
        self.action_count = action_count
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.actions = [np.eye(action_count)[i] for i in range(action_count)]
        self.t = 0
        self.action_weights = self.rng.normal(0, 1, (action_count, input_dim))

    def reset(self):
        self.t = 0
        self.obs = self.rng.normal(0, 1, self.input_dim)
        self.context = self.rng.normal(0, 1, self.input_dim)
        return self.obs.copy()

    def available_actions(self):
        return self.actions

    def step(self, action):
        idx = int(np.argmax(action))
        scores = self.action_weights @ self.context
        target = int(np.argmax(scores))
        reward = 1.0 if idx == target else 0.0
        self.t += 1
        done = self.t >= self.horizon
        self.obs = self.rng.normal(0, 1, self.input_dim)
        self.context = self.rng.normal(0, 1, self.input_dim)
        return self.obs.copy(), reward, done, {"target": target, "chosen": idx}


class ContinuousTorqueEnv:
    action_mode = "continuous"
    def __init__(self, obs_dim=8, action_dim=4, horizon=100, seed=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.action_low = -np.ones(action_dim)
        self.action_high = np.ones(action_dim)
        self.t = 0
        self.state = np.zeros(obs_dim)
        self.target = self.rng.uniform(-1.0, 1.0, action_dim)
        self.A = self.rng.normal(0, 0.2, (obs_dim, obs_dim))
        self.B = self.rng.normal(0, 0.1, (obs_dim, action_dim))

    def reset(self):
        self.t = 0
        self.state = self.rng.normal(0, 1, self.obs_dim)
        self.target = self.rng.uniform(-1.0, 1.0, self.action_dim)
        return self.state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(self.action_dim)
        action = np.clip(action, self.action_low, self.action_high)
        self.state = np.tanh(self.A @ self.state + self.B @ action + 0.05 * self.rng.normal(size=self.obs_dim))
        reward = float(np.clip(1.0 - np.mean((action - self.target) ** 2) / 4.0, 0.0, 1.0))
        self.t += 1
        done = self.t >= self.horizon
        info = {"target": self.target.copy(), "action": action.copy()}
        return self.state.copy(), reward, done, info


# ======================================================================
# EVALUATION AND SENSITIVITY ANALYSIS FUNCTIONS
# ======================================================================

def evaluate_reward_performance(brain_instance, num_steps, env_factory):
    """
    Corrected evaluation: steps the environment and uses the task reward.
    """
    env = env_factory()
    obs = env.reset()
    brain_instance.sense(obs)

    rewards = []
    step_count = 0
    done = False

    while step_count < num_steps:
        if done:
            obs = env.reset()
            brain_instance.sense(obs)
            done = False

        action = brain_instance.propose_action()

        if hasattr(env, "available_actions"):
            idx = brain_instance.action_to_index(action, env.available_actions())
            obs, reward, done, info = env.step(env.available_actions()[idx])
            target_vec = np.eye(len(env.available_actions()))[info["target"]] if "target" in info else None
        else:
            obs, reward, done, info = env.step(action)
            target_vec = None

        brain_instance.sense(obs)
        brain_instance.learn_step(reward_fn=lambda b, r=reward: r, action_target=target_vec)

        rewards.append(reward)
        step_count += 1

    return rewards


def run_sensitivity_analysis(param_name, param_values, base_config_params, num_evaluation_steps, env_factory):
    """
    Systematic hyperparameter tuning.
    """
    average_rewards = []
    print(f"Starting sensitivity analysis for {param_name} over {num_evaluation_steps} steps per value...")

    for p_val in param_values:
        print(f"Evaluating with {param_name} = {p_val}")

        current_config = Config()
        for key, value in base_config_params.items():
            setattr(current_config, key, value)
        setattr(current_config, param_name, p_val)

        brain_for_sensitivity = current_config.build()
        brain_for_sensitivity._memory_batch_size = 32

        rewards_for_p_val = evaluate_reward_performance(
            brain_for_sensitivity, num_evaluation_steps, env_factory=env_factory
        )
        avg_reward = np.mean(rewards_for_p_val)
        average_rewards.append(avg_reward)
        print(f"  Average reward: {avg_reward:.4f}\n")

    print(f"Sensitivity analysis for {param_name} complete.")
    return average_rewards


def plot_sensitivity_results(param_values, average_rewards, param_name, xscale='log'):
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, average_rewards, marker='o', linestyle='-')
    if xscale == 'log':
        plt.xscale('log')
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Average Reward")
    plt.title(f"Sensitivity Analysis: Average Reward vs. {param_name.replace('_', ' ').title()}")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.savefig("sensitivity_results.png", bbox_inches="tight")
    plt.close()


# ======================================================================
# RUN_EPISODE_SCALED (to use with scaled rewards)
# ======================================================================

def run_episode_scaled(agent, env, steps=300, reward_scale=5.0):
    """
    Run an episode with a scaled reward for learning.
    """
    obs = env.reset()
    total = 0.0
    mode = getattr(env, "action_mode", "discrete" if hasattr(env, "available_actions") else "continuous")
    for _ in range(steps):
        agent.controller.observe(obs)  # observe current state each step
        orient = agent.controller.orient()
        if mode == "discrete":
            idx = agent.controller.decide_discrete(env.available_actions())
            obs, reward, done, info = agent.controller.act_discrete(env, idx)
            target_vec = np.eye(len(env.available_actions()))[info["target"]]
            scaled_reward = reward * reward_scale
            agent.brain.learn_step(
                reward_fn=lambda b, r=scaled_reward: r,
                action_target=target_vec
            )
        else:
            action = np.asarray(orient["action"], dtype=float)
            obs, reward, done, info = agent.controller.act_continuous(env, action)
            scaled_reward = reward * reward_scale
            agent.brain.learn_step(reward_fn=lambda b, r=scaled_reward: r)
        total += reward
        if done:
            break
    return total


# ======================================================================
# MAIN DEMO & TRAINING LOOP
# ======================================================================

if __name__ == "__main__":
    MASTER_SEED = 42
    print("=== SelfDrivingBrain v5.0 (Merged Continuous/Discrete Final Demo) ===")

    # ------------------------------------------------------------------
    # 1. Quick demo runs
    # ------------------------------------------------------------------
    config = Config()
    config.seed = MASTER_SEED
    config.latent_dim = 16
    config.input_dim = 8
    config.action_dim = 4
    config.use_attention = True
    config.use_curiosity = True
    config.memory_size = 1000
    config.attention_history_len = 8
    config.attention_strength = 0.02

    brain = config.build()
    agent = AdaptiveAgent(brain)

    env = HiddenPatternEnv(
        input_dim=8,
        action_count=4,
        horizon=50,
        seed=MASTER_SEED,
    )
    total = agent.run_episode(env, steps=50)
    print(f"Discrete episode reward: {total:.2f}")

    print("\n--- Continuous control demo ---")
    cont_env = ContinuousTorqueEnv(
        obs_dim=8,
        action_dim=4,
        horizon=50,
        seed=MASTER_SEED,
    )
    cont_brain = SelfDrivingBrain(
        latent_dim=16,
        input_dim=8,
        action_dim=4,
        seed=MASTER_SEED,
    )
    cont_agent = AdaptiveAgent(cont_brain)
    cont_total = cont_agent.run_episode(cont_env, steps=50)
    print(f"Continuous episode reward: {cont_total:.2f}")

    print("\n--- Evolutionary run ---")
    evo = EvolutionEngine(
        population_size=4,
        latent_dim=16,
        input_dim=8,
        action_dim=4,
        seed=MASTER_SEED,
    )
    hist = evo.evolve(
        lambda: HiddenPatternEnv(seed=MASTER_SEED),
        generations=2,
        steps=30,
    )
    for h in hist:
        print(f"Gen {h['generation']}: best={h['best']:.2f}, avg={h['average']:.2f}")
    print("✅ All systems operational with fixes applied.\n")

    # ------------------------------------------------------------------
    # 2. Full training loop with the new parameters
    # ------------------------------------------------------------------
    print("=== Training loop with reward_clip_range=5.0 and supervised_scale_max=5.0 ===")
    config_train = Config()
    config_train.seed = MASTER_SEED
    config_train.latent_dim = 16
    config_train.input_dim = 8
    config_train.action_dim = 4
    config_train.use_attention = True
    config_train.use_curiosity = True
    config_train.memory_size = 5000
    config_train.hebbian_lr = 1e-3          # lower for stability
    config_train.lr = 1e-2
    config_train.eligibility_decay = 0.95    # longer credit
    config_train.attention_strength = 0.05
    config_train.gradient_noise_std = 0.001
    # NEW: set the clipping range and scale max to 5.0
    config_train.reward_clip_range = 5.0
    config_train.supervised_scale_max = 5.0

    brain_train = config_train.build()
    brain_train._memory_batch_size = 32
    agent_train = AdaptiveAgent(brain_train)
    env_train = HiddenPatternEnv(
        input_dim=8,
        action_count=4,
        horizon=50,
        seed=MASTER_SEED,
    )

    # Use the scaled episode runner
    episode_rewards = []
    for ep in range(200):
        total_ep = run_episode_scaled(agent_train, env_train, steps=300, reward_scale=5.0)
        episode_rewards.append(total_ep)
        if (ep+1) % 20 == 0:
            print(f"Episode {ep+1}: {total_ep:.2f} (avg last 20: {np.mean(episode_rewards[-20:]):.2f})")

    # Plot episode rewards with moving average
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.4, label='Episode total')
    window = 20
    ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(episode_rewards)), ma, 'r-', linewidth=2, label=f'{window}-ep MA')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Progress with Optimized Parameters & Reward Scaling")
    plt.grid(True)
    plt.legend()
    plt.savefig("episode_rewards.png", bbox_inches="tight")
    plt.close()

    print(f"Average over last 20 episodes: {np.mean(episode_rewards[-20:]):.2f}")

    # ------------------------------------------------------------------
    # 3. Save the trained brain
    # ------------------------------------------------------------------
    np.savez("final_optimized_brain.npz",
             Phi=brain_train.Phi,
             W=brain_train.W,
             M_op=brain_train.M_op,
             P_op=brain_train.P_op,
             W_in=brain_train.W_in,
             W_pred=brain_train.W_pred,
             b_pred=brain_train.b_pred,
             W_act=brain_train.W_act,
             b_act=brain_train.b_act,
             W_M_damp=brain_train.W_M_damp)
    print("Brain saved to final_optimized_brain.npz")