"""Unit 15: solving the grid-world MDP by value iteration.

Bellman update: V(s) <- max_a sum_s' P(s'|s,a) [R + gamma * V(s')].
"""
from gridworld import NAMES, STATES, show_policy, terminal, transitions

gamma = 0.9
V = {s: 0.0 for s in STATES}
for sweep in range(1, 100):
    delta = 0.0
    for s in STATES:
        if terminal(s):
            continue
        best = max(sum(p * (r + gamma * V[n] * (not terminal(n)))
                       for p, n, r in transitions(s, a)) for a in NAMES)
        delta = max(delta, abs(best - V[s]))
        V[s] = best
    if delta < 1e-4:
        break
print(f"converged after {sweep} sweeps")


def q(s, a):
    return sum(p * (r + gamma * V[n] * (not terminal(n)))
               for p, n, r in transitions(s, a))


policy = {s: max(NAMES, key=lambda a: q(s, a))
          for s in STATES if not terminal(s)}
print("state values V(s):")
for r in range(4):
    print("  " + " ".join(f"{V.get((r, c), 0):6.2f}" for c in range(4)))
print("optimal policy:")
show_policy(policy)
