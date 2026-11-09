"""Unit 15: a 4 x 4 grid world, shared by value iteration and Q-learning.

A delivery robot starts at S (top left) and must reach the goal G
(bottom right, reward +10). The cell X is a pothole (reward -10, the
episode ends). Every other move costs -1. With probability 0.1 the
robot slips and moves in a random direction instead.
"""
import numpy as np

ROWS, COLS = 4, 4
START, GOAL, PIT = (0, 0), (3, 3), (1, 2)
WALL = (2, 1)                                  # a building: cannot enter
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
NAMES = list(ACTIONS)
SLIP = 0.1
STATES = [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != WALL]


def terminal(s):
    return s in (GOAL, PIT)


def move(s, a):
    dr, dc = ACTIONS[a]
    n = (s[0] + dr, s[1] + dc)
    if not (0 <= n[0] < ROWS and 0 <= n[1] < COLS) or n == WALL:
        return s                               # bump: stay where you are
    return n


def reward(n):
    return 10 if n == GOAL else -10 if n == PIT else -1


def transitions(s, a):
    """List of (probability, next_state, reward) for action a in s."""
    out = []
    for b in NAMES:
        p = (1 - SLIP) + SLIP / 4 if b == a else SLIP / 4
        n = move(s, b)
        out.append((p, n, reward(n)))
    return out


def step(s, a, rng):
    """Sample one move, as the learning agent experiences it."""
    if rng.random() < SLIP:
        a = NAMES[rng.integers(4)]
    n = move(s, a)
    return n, reward(n), terminal(n)


def show_policy(policy):
    arrow = {"up": "^", "down": "v", "left": "<", "right": ">"}
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = (r, c)
            row.append("G" if s == GOAL else "X" if s == PIT else
                       "#" if s == WALL else arrow[policy[s]])
        print("   " + " ".join(row))
