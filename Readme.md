
# Reinforcement Learning on FrozenLake (Stochastic) with Q‑Learning

A compact, reproducible project that trains a Q‑Learning agent on **Gymnasium’s `FrozenLake-v1`** environment in its **slippery (stochastic)** configuration.  
It demonstrates practical RL techniques: epsilon‑greedy exploration with decay, adaptive learning rate, reward shaping, and evaluation on held‑out seeds.

> **Highlights**
> - Gymnasium + `TimeLimit` wrapper for controlled episode length  
> - Q‑Learning with epsilon‑greedy policy and slow decay for stochastic dynamics  
> - Simple reward shaping via per‑step penalty to encourage faster solutions  
> - Deterministic seeding for reproducible training & evaluation  
> - Rendered demo episode to visually inspect learned behavior

---

## Project Structure

```
.
├── stochastic_aproach.py   # Main script: env setup, training, evaluation, render
├── README.md               # You are here
└── (optional) notebooks/   # Add analysis notebooks if needed
```

---

## Requirements

- **Python** 3.9+
- **Packages**
  - `gymnasium`
  - `numpy`

Install dependencies:
```bash
pip install gymnasium numpy
```

> (Optional) If you plan to render the environment locally, make sure your machine supports the required GUI packages for Gymnasium rendering. On some platforms you may need `pygame` or OS‑specific dependencies.

---

## The Environment: `FrozenLake-v1`

- **States**: grid cells on a frozen lake (start `S`, goal `G`, holes `H`, frozen tiles `F`)  
- **Actions**: `LEFT=0, DOWN=1, RIGHT=2, UP=3`  
- **Stochastic Dynamics**: with `is_slippery=True`, actions have a chance to deviate from the intended direction → harder exploration, higher variance.

---

## How It Works

### Q‑Learning Update
For state \(s\), action \(a\), reward \(r\), next state \(s'\):
\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left( r' + \gamma \max_{a'} Q(s',a') - Q(s,a) \right)
\]
- \(\alpha\): learning rate (linearly decayed from `alpha_start` to `alpha_end`)
- \(\gamma\): discount factor
- \(r'\): shaped reward = `reward - step_penalty` (small per‑step penalty)

### Exploration
- **Epsilon‑greedy**: with probability \(\varepsilon\), take a random action; else take the greedy action from the Q‑table.
- **Decay**: `epsilon = max(epsilon_min, epsilon * epsilon_decay)`  
  Slower decay is helpful for slippery dynamics (more exploration required).

---

## Usage

### 1 Train, Evaluate, and Render (single command)
Run the script directly:

```bash
python stochastic_aproach.py
```

**What happens:**
1. **Train** on a slippery environment (`seed=7`) for `episodes=100000`.
2. **Report** training success rate (SR) progression every 1000 episodes.
3. **Evaluate** the greedy policy on a fresh environment (`seed=9`) for `episodes=500`.
4. **Render** one demo episode (`seed=10`) to visualize the learned policy.

### 2 Customize Hyperparameters

All key settings are exposed in the functions:

```python
env = make_env(slippery=True, render=False, seed=7, max_steps=200)

Q, train_sr, _ = q_learning(
    env,
    episodes=100000,
    max_steps=200,
    alpha_start=1.0,
    alpha_end=0.05,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.9995,
    step_penalty=0.001
)
```

Change seeds, `episodes`, `epsilon_decay`, or `step_penalty` to explore different learning dynamics.

---

## Reproducibility

- **Seeding**:  
  - Environment seed via `env.reset(seed=...)`  
  - Action/observation space seeds (if supported)  
  - Evaluation uses a different seed than training to reduce overfitting to a single randomization.
- **TimeLimit**: caps episode length with `max_episode_steps` to keep training stable and comparable.

---

## Expected Output

During training (every 1000 episodes), you’ll see logs like:
```
Episode 1000: cumulative training SR=0.035, epsilon=0.607, alpha=0.990
Episode 2000: cumulative training SR=0.048, epsilon=0.368, alpha=0.980
...
Final training success rate (slippery): 0.XXX
Evaluation success rate (slippery): 0.XXX
```

> **Note**: Exact SR depends on seeds and hyperparameters. Slippery FrozenLake is intentionally challenging; expect gradual improvements and variance.

---

## File: `stochastic_aproach.py` (Overview)

- `make_env(slippery, render, seed, max_steps)`: creates a Gymnasium FrozenLake environment wrapped by `TimeLimit`.
- `greedy_action(q_row)`: breaks ties randomly among max‑Q actions for better state coverage.
- `q_learning(...)`: trains Q‑table with adaptive \(\alpha\), decaying \(\varepsilon\), and step penalty shaping; tracks successes.
- `evaluate(env, Q, episodes, max_steps)`: runs the greedy policy to compute success rate on held‑out episodes.
- `__main__`: orchestrates training → evaluation → one rendered demo.

---

## Results & Discussion

- **Reward Shaping**: A tiny per‑step penalty (`step_penalty=0.001`) biases the agent toward shorter paths and discourages aimless wandering.
- **Slow Epsilon Decay**: Keeps exploration alive longer in the slippery setting where deterministic policies fail more often.
- **Tie‑Breaking**: Random choice among max‑Q actions helps the agent avoid getting stuck when multiple actions look equally good.

Consider experimenting with:
- **Different seeds** to evaluate stability.
- **Learning schedules** (exponential vs. linear decay of \(\alpha\), alternate \(\varepsilon\) schedules).
- **Gamma** values (more/less emphasis on future rewards).
- **Non‑slippery baseline** (`slippery=False`) for an easier comparison.

---

## Visualizing a Run

Enable rendering by setting `render=True` in `make_env` for the watch phase:
```python
watch_env = make_env(slippery=True, render=True, seed=10, max_steps=200)
```
A window will show the agent moving on the grid; the episode terminates on reaching `G` or falling into `H`.

---

## Benchmarking Tips

- Run multiple **evaluation seeds** and report **mean ± std** of success rate.
- Log **episode length** and **return** distributions for deeper insight.
- Compare against a **random policy** baseline to quantify gains.

---

## Known Limitations

- **Tabular Q‑Learning** scales poorly to large/continuous state spaces.  
- **FrozenLake** rewards are sparse; learning can be slow and seed‑sensitive.  
- Rendering may require additional system packages depending on OS.

---

## Roadmap / Extensions

-  Add plots for training SR over time (Matplotlib)  
-  Non‑slippery baseline comparison  
-  Double Q‑Learning (reduces overestimation bias)  
-  SARSA for on‑policy comparison  
-  DQN for function approximation on larger state spaces  
-  Hyperparameter sweep and automated reporting

---

## How to Cite / Share

If you share this on LinkedIn/GitHub:
- Mention **Gymnasium `FrozenLake-v1` (slippery)**, **Q‑Learning**, **epsilon‑greedy with decay**, **reward shaping**, and **deterministic seeds**.
- Include your **training SR** and **evaluation SR** numbers, plus a short GIF/clip of the demo episode.

---

## License

This project is intended for educational and research use :
```
Copyright (c) 2025 Rahul Shil
```

---

## Acknowledgements

- **Gymnasium** (OpenAI Gym successor) for standardized RL environments
- Community resources on tabular RL and FrozenLake for best practices

---

### Author

**Rahul Shil** — Data Analyst | Aspiring Data Scientist  
Pune, Maharashtra

