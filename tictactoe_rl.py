import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Tic-Tac-Toe environment
# -----------------------------
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diags
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

def board_to_state(board: List[str]) -> str:
    return "".join(board)

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
# Q-learning agent
# -----------------------------
class QAgent:
    def __init__(self, alpha=0.4, gamma=0.95, epsilon=0.2):
        self.Q: Dict[Tuple[str, int], float] = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state: str, moves: List[int], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.choice(moves)

        qs = [(self.Q[(state, a)], a) for a in moves]
        max_q = max(qs, key=lambda x: x[0])[0]
        best = [a for q, a in qs if q == max_q]
        return random.choice(best)

    def update(self, s: str, a: int, r: float, s2: str, moves2: List[int], terminal: bool) -> None:
        old = self.Q[(s, a)]
        if terminal:
            target = r
        else:
            next_best = max(self.Q[(s2, a2)] for a2 in moves2) if moves2 else 0.0
            target = r + self.gamma * next_best
        self.Q[(s, a)] = old + self.alpha * (target - old)

# -----------------------------
# Training via self-play
# -----------------------------
def train_self_play(
    episodes: int = 200_000,
    alpha: float = 0.4,
    gamma: float = 0.95,
    epsilon_start: float = 0.6,
    epsilon_end: float = 0.05,
) -> QAgent:
    agent = QAgent(alpha=alpha, gamma=gamma, epsilon=epsilon_start)

    for ep in range(1, episodes + 1):
        t = ep / episodes
        agent.epsilon = epsilon_start + t * (epsilon_end - epsilon_start)

        board = ["." for _ in range(9)]
        player = "X"
        last_sa = {"X": None, "O": None}

        while True:
            state = board_to_state(board)
            moves = legal_moves(board)
            action = agent.choose_action(state, moves, explore=True)
            board[action] = player

            winner = check_winner(board)
            draw = is_draw(board)
            last_sa[player] = (state, action)

            if winner or draw:
                if draw:
                    rX, rO = 0.0, 0.0
                else:
                    rX, rO = (1.0, -1.0) if winner == "X" else (-1.0, 1.0)

                s, a = last_sa[player]
                agent.update(s, a, rX if player == "X" else rO, s2="", moves2=[], terminal=True)

                other = "O" if player == "X" else "X"
                if last_sa[other] is not None:
                    s_oth, a_oth = last_sa[other]
                    agent.update(s_oth, a_oth, rX if other == "X" else rO, s2="", moves2=[], terminal=True)
                break

            other = "O" if player == "X" else "X"
            if last_sa[other] is not None:
                s_prev, a_prev = last_sa[other]
                s2 = board_to_state(board)
                moves2 = legal_moves(board)
                agent.update(s_prev, a_prev, 0.0, s2, moves2, terminal=False)

            player = "O" if player == "X" else "X"

    agent.epsilon = 0.0
    return agent

# -----------------------------
# One game vs human (returns result)
# -----------------------------
def play_one_game(agent: QAgent, human: str) -> str:
    board = ["." for _ in range(9)]
    ai = "O" if human == "X" else "X"
    turn = "X"

    print("\nCells are numbered 0-8.")
    render(board)

    while True:
        winner = check_winner(board)
        if winner:
            print(f"{winner} wins!")
            return "HUMAN" if winner == human else "AI"
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
                    choice = int(raw)
                    if choice in moves:
                        break
                except ValueError:
                    pass
                print("Invalid move. Try again.")
            board[choice] = human
        else:
            state = board_to_state(board)
            moves = legal_moves(board)
            action = agent.choose_action(state, moves, explore=False)
            board[action] = ai
            print(f"AI plays: {action}")

        render(board)
        turn = "O" if turn == "X" else "X"

# -----------------------------
# Loop games + scoreboard
# -----------------------------
def main():
    print("Training RL agent (Q-learning) by self-play...")
    agent = train_self_play(episodes=200_000)
    print("Done.\n")

    stats = {"wins": 0, "losses": 0, "ties": 0}

    # Default choice; player can change each game
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

        result = play_one_game(agent, human_choice)

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
            human_choice = None  # force re-pick next loop

    print(f"\nFinal Scoreboard: Wins={stats['wins']}  Losses={stats['losses']}  Ties={stats['ties']}")
    print("Thanks for playing!")

if __name__ == "__main__":
    main()
