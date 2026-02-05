# AI-TTT

Monte Carlo Tree Search (MCTS) Visualization

This project implements a Tic-Tac-Toe AI using Monte Carlo Tree Search (MCTS) with a real-time graphical tree visualization built in Pygame. The visualization shows how the search tree grows during simulations and how the algorithm evaluates moves.

Each node represents a game state and displays:

V (Visits): number of times the node has been explored

Q (Value): average reward from simulations (Q = total_value / visits)

The tree updates as MCTS runs through its four phases:

Selection (UCT-based traversal)

Expansion (adding new moves)

Simulation (random rollout)

Backpropagation (updating statistics)

This visualization helps demonstrate how MCTS balances exploration and exploitation and gradually focuses on stronger moves.
