# Unit 15: Markov decision processes, value iteration, Q-learning

Programs for Unit 15 of the RSU-CSC 465 lecture note.

| Program | What it does |
|---|---|
| `gridworld.py` | a 4 x 4 grid world, shared by value iteration and Q-learning |
| `q_learning.py` | Q-learning: learning the grid-world policy by trial and error |
| `value_iteration.py` | solving the grid-world MDP by value iteration |

Run a program from this folder, e.g. `python value_iteration.py` (`gridworld.py` is imported by the other two).
Figures are saved in `figures/` here. Run them all from the top folder with
`python tests/run_all.py u15`.
