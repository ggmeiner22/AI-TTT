import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

def check_winner(board: List[str]) -> Optional[str]:
    for a, b, c in WIN_LINES:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None

def is_draw(board: List[str]) -> bool:
    return "." not in board and check_winner(board) is None

def legal_moves(board: List[str]) -> List[int]:
    return [i for i, v in enumerate(board) if v == "."]

def apply_move(board: List[str], move: int, player: str) -> List[str]:
    nb = board[:]  # copy
    nb[move] = player
    return nb

def other(player: str) -> str:
    return "O" if player == "X" else "X"

def render(board: List[str]) -> None:
    def cell(i: int) -> str:
        return str(i) if board[i] == "." else board[i]
    print()
    print(f" {cell(0)} | {cell(1)} | {cell(2)}")
    print("---+---+---")
    print(f" {cell(3)} | {cell(4)} | {cell(5)}")
    print("---+---+---")
    print(f" {cell(6)} | {cell(7)} | {cell(8)}")
    print()

# -----------------------------
# MCTS
# -----------------------------
@dataclass
class Node:
    board: Tuple[str, ...]                 # immutable for dict keys
    player_to_move: str                   # whose turn at this node
    parent: Optional["Node"] = None
    move_from_parent: Optional[int] = None

    children: Dict[int, "Node"] = field(default_factory=dict)  # move -> child
    untried_moves: List[int] = field(default_factory=list)

    visits: int = 0
    total_value: float = 0.0              # from root player's perspective

    def is_terminal(self) -> bool:
        b = list(self.board)
        return check_winner(b) is not None or is_draw(b)

    def uct_score(self, child: "Node", c: float) -> float:
        # UCT = Q/N + c * sqrt(ln(N_parent)/N_child)
        if child.visits == 0:
            return float("inf")
        exploit = child.total_value / child.visits
        explore = c * math.sqrt(math.log(self.visits) / child.visits)
        return exploit + explore

def rollout(board: List[str], player_to_move: str) -> Optional[str]:
    # random playout to terminal; return winner ('X'/'O') or None for draw
    p = player_to_move
    b = board[:]
    while True:
        w = check_winner(b)
        if w is not None:
            return w
        if is_draw(b):
            return None
        m = random.choice(legal_moves(b))
        b[m] = p
        p = other(p)

def mcts_best_move(
    board: List[str],
    player_to_move: str,
    iterations: int = 5000,
    c: float = 1.4,
) -> int:
    """
    Returns the chosen move for player_to_move using MCTS.
    Value is always tracked from the *root player's* perspective.
    """
    root_player = player_to_move
    root = Node(board=tuple(board), player_to_move=player_to_move)
    root.untried_moves = legal_moves(board)

    for _ in range(iterations):
        node = root
        b = list(root.board)
        p = node.player_to_move

        # 1) SELECTION
        while (not node.is_terminal()) and (len(node.untried_moves) == 0):
            # pick best UCT child
            best_move, best_child, best_score = None, None, -1e18
            for mv, ch in node.children.items():
                score = node.uct_score(ch, c)
                if score > best_score:
                    best_score = score
                    best_move, best_child = mv, ch
            # descend
            node = best_child
            b = apply_move(b, node.move_from_parent, other(node.player_to_move))
            p = node.player_to_move

        # 2) EXPANSION
        if (not node.is_terminal()) and node.untried_moves:
            mv = random.choice(node.untried_moves)
            node.untried_moves.remove(mv)

            # apply mv for current player at node
            b = apply_move(b, mv, node.player_to_move)
            next_player = other(node.player_to_move)

            child = Node(
                board=tuple(b),
                player_to_move=next_player,
                parent=node,
                move_from_parent=mv,
            )
            child.untried_moves = legal_moves(b)
            node.children[mv] = child
            node = child
            p = node.player_to_move

        # 3) SIMULATION (ROLLOUT)
        winner = rollout(b, p)

        # Convert terminal outcome to reward from root_player perspective
        if winner is None:
            reward = 0.0
        elif winner == root_player:
            reward = 1.0
        else:
            reward = -1.0

        # 4) BACKPROPAGATION
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.total_value += reward
            cur = cur.parent

    # Choose move with highest visit count (standard)
    if not root.children:
        # no legal moves (shouldn't happen unless terminal)
        return random.choice(legal_moves(board))

    best_mv = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
    return best_mv

# -----------------------------
# Play loop + scoreboard
# -----------------------------
def play_one_game_mcts(human: str, iterations: int) -> str:
    board = ["." for _ in range(9)]
    ai = other(human)
    turn = "X"

    print("\nCells are numbered 0-8.")
    render(board)

    while True:
        w = check_winner(board)
        if w is not None:
            print(f"{w} wins!")
            return "HUMAN" if w == human else "AI"
        if is_draw(board):
            print("It's a draw.")
            return "DRAW"

        if turn == human:
            moves = legal_moves(board)
            while True:
                raw = input(f"Your move {moves} (or 'q' to quit this match): ").strip().lower()
                if raw in ("q", "quit", "exit"):
                    return "QUIT_MATCH"
                try:
                    mv = int(raw)
                    if mv in moves:
                        break
                except ValueError:
                    pass
                print("Invalid move. Try again.")
            board[mv] = human
        else:
            mv = mcts_best_move(board, turn, iterations=iterations)
            board[mv] = turn
            print(f"AI plays: {mv}")

        render(board)
        turn = other(turn)

def main():
    stats = {"wins": 0, "losses": 0, "ties": 0}

    # You can tweak difficulty by changing iterations:
    # ~500: easy-ish, ~2000: decent, ~5000+: strong (and still instant on most PCs)
    iterations = 5000

    human_choice = None
    while True:
        if human_choice not in ("X", "O"):
            pick = input("Play as X (first) or O? [X/O] (or 'q' to quit): ").strip().upper()
            if pick in ("Q", "QUIT", "EXIT"):
                break
            if pick not in ("X", "O"):
                print("Please type X or O.")
                continue
            human_choice = pick

        result = play_one_game_mcts(human_choice, iterations)

        if result == "HUMAN":
            stats["wins"] += 1
            print("Result: You win!")
        elif result == "AI":
            stats["losses"] += 1
            print("Result: You lose!")
        elif result == "DRAW":
            stats["ties"] += 1
            print("Result: Tie.")
        elif result == "QUIT_MATCH":
            print("You quit the current match (not counted).")

        print(f"\nScoreboard: Wins={stats['wins']}  Losses={stats['losses']}  Ties={stats['ties']}\n")

        again = input("Play again? [Y/n] (or 'c' to change X/O): ").strip().lower()
        if again in ("n", "no", "q", "quit", "exit"):
            break
        if again in ("c", "change"):
            human_choice = None

    print(f"\nFinal Scoreboard: Wins={stats['wins']}  Losses={stats['losses']}  Ties={stats['ties']}")
    print("Thanks for playing!")

if __name__ == "__main__":
    main()
