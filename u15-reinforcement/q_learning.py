"""Unit 15: Q-learning: learning the grid-world policy by trial and error.

The agent does not know the transition probabilities. It updates
Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
after every move, exploring with an epsilon-greedy rule.
"""
import numpy as np
from gridworld import NAMES, START, STATES, show_policy, step, terminal

rng = np.random.default_rng(465)
alpha, gamma = 0.1, 0.9
Q = {(s, a): 0.0 for s in STATES for a in NAMES}
returns = []
for episode in range(1, 3001):
    eps = max(0.05, 1.0 - episode / 1500)      # explore less over time
    s, total, done, moves = START, 0.0, False, 0
    while not done and moves < 100:
        if rng.random() < eps:
            a = NAMES[rng.integers(4)]
        else:
            a = max(NAMES, key=lambda b: Q[(s, b)])
        n, r, done = step(s, a, rng)
        target = r if done else r + gamma * max(Q[(n, b)] for b in NAMES)
        Q[(s, a)] += alpha * (target - Q[(s, a)])
        s, total, moves = n, total + r, moves + 1
    returns.append(total)
    if episode in (100, 500, 1000, 2000, 3000):
        print(f"episodes {episode - 99:4d}-{episode}: mean return "
              f"{np.mean(returns[-100:]):6.2f}")

policy = {s: max(NAMES, key=lambda a: Q[(s, a)])
          for s in STATES if not terminal(s)}
print("learned policy:")
show_policy(policy)
print(f"Q(start, right) = {Q[(START, 'right')]:.2f}, "
      f"Q(start, down) = {Q[(START, 'down')]:.2f}")
