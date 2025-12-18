
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np

# Actions: LEFT=0, DOWN=1, RIGHT=2, UP=3

def make_env(slippery=False, render=False, seed=42, max_steps=200):
    render_mode = "human" if render else None
    base_env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode=render_mode)
    env = TimeLimit(base_env, max_episode_steps=max_steps)
    env.reset(seed=seed)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass
    return env

def greedy_action(q_row):
    max_q = np.max(q_row)
    candidates = np.flatnonzero(q_row == max_q)
    return int(np.random.choice(candidates))

def q_learning(env,
               episodes=100000,     
               max_steps=200,
               alpha_start=1.0,     
               alpha_end=0.05,     
               gamma=0.99,         
               epsilon_start=1.0,  
               epsilon_min=0.01,   
               epsilon_decay=0.9995,   
               step_penalty= 0.001     
               ):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    successes = 0
    sr_history = []

    epsilon = epsilon_start

    for ep in range(episodes):
        # Linear decay for alpha
        frac = ep / episodes
        alpha = alpha_start - (alpha_start - alpha_end) * frac

        state, _ = env.reset()
        for t in range(max_steps):
            # exploration vs exploition
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(Q[state])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # reward shaping(optional) - per-step penalty
            shaped_reward = reward - step_penalty

            # Q-learning update - SARSA and SARSAMAX
            best_next = 0.0 if done else np.max(Q[next_state])
            td_target = shaped_reward + gamma * best_next
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state

            if done:
                if terminated and reward > 0:
                    successes += 1
                break

        # decay epsilon slowly (slippery lake needs more exploration)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Track progress every 1000 episodes
        if (ep + 1) % 1000 == 0:
            sr = successes / (ep + 1)
            print(f"Episode {ep+1}: cumulative training SR={sr:.3f}, epsilon={epsilon:.3f}, alpha={alpha:.3f}")

    return Q, successes / episodes, sr_history

def evaluate(env, Q, episodes=500, max_steps=200):
    # Run greedy policy on the  Q-table and compute success rate.
    wins = 0
    for _ in range(episodes):
        state, _ = env.reset()
        for _ in range(max_steps):
            action = greedy_action(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if terminated and reward > 0:
                    wins += 1
                break
    return wins / episodes

if __name__ == "__main__":
    # Train on stochastic (slippery) lake
    env = make_env(slippery=True, render=False, seed=7, max_steps=200)
    Q, train_sr, _ = q_learning(env)
    print(f"Final training success rate (slippery): {train_sr:.3f}")

    # Evaluate on the same stochastic environment
    eval_env = make_env(slippery=True, render=False, seed=9, max_steps=200)
    eval_sr = evaluate(eval_env, Q, episodes=500, max_steps=200)
    print(f"Evaluation success rate (slippery): {eval_sr:.3f}")

    # Watch one episode with rendering
    watch_env = make_env(slippery=True, render=True, seed=10, max_steps=200)
    state, _ = watch_env.reset()
    done = False
    while not done:
        action = greedy_action(Q[state])
        state, reward, terminated, truncated, info = watch_env.step(action)
        done = terminated or truncated
    watch_env.close()
