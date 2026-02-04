# tree_visualize.py
from __future__ import annotations
import math
import pygame
from typing import Optional

# DO NOT import your main file
# Node and q_value are imported by type only at runtime via forward refs

class PygameTreeViewer:
    def __init__(self, width=1200, height=800, font_size=16):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MCTS Tree Viewer (Pygame)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, font_size)
        self.width = width
        self.height = height
        self.paused = False

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True

    def draw_text(self, text: str, x: int, y: int):
        surf = self.font.render(text, True, (240, 240, 240))
        self.screen.blit(surf, (x, y))

    def render_tree(
        self,
        root: "Node",
        q_value_fn=None,
        depth: int = 3,
        max_children: int = 6,
        sort_by: str = "visits",
        c: float = 1.4,
        iteration: int = 0,
    ) -> bool:
        if not self.handle_events():
            return False
        if q_value_fn is None:
            raise ValueError("q_value_fn must be provided")


        self.screen.fill((20, 20, 24))

        # --- simple BFS layout ---
        levels = [[root]]
        edges = []
        seen = {root}

        for _ in range(depth):
            next_level = []
            for parent in levels[-1]:
                items = list(parent.children.items())

                def key(item):
                    mv, ch = item
                    if sort_by == "visits":
                        return (-ch.visits, mv)
                    if sort_by == "q":
                        return (-q_value_fn(ch), mv)
                    if sort_by == "move":
                        return (mv,)
                    # UCT
                    if ch.visits == 0:
                        return (-float("inf"), mv)
                    if parent.visits == 0:
                        uct = float("inf")
                    else:
                        exploit = q_value_fn(ch)
                        explore = c * math.sqrt(math.log(parent.visits) / ch.visits)
                        uct = exploit + explore
                    return (-uct, mv)

                items.sort(key=key)
                items = items[:max_children]

                for mv, ch in items:
                    edges.append((parent, ch, mv))
                    if ch not in seen:
                        seen.add(ch)
                        next_level.append(ch)

            if not next_level:
                break
            levels.append(next_level)

        # assign positions
        positions = {}
        h_gap = (self.height - 120) // max(1, len(levels) - 1)
        for i, lvl in enumerate(levels):
            y = 60 + i * h_gap
            w_gap = (self.width - 80) // max(1, len(lvl))
            for j, node in enumerate(lvl):
                x = 40 + j * w_gap
                positions[node] = (x, y)

        # draw edges
        for parent, child, mv in edges:
            if parent in positions and child in positions:
                pygame.draw.line(
                    self.screen,
                    (130, 130, 150),
                    positions[parent],
                    positions[child],
                    2,
                )
                mx = (positions[parent][0] + positions[child][0]) // 2
                my = (positions[parent][1] + positions[child][1]) // 2
                self.draw_text(str(mv), mx + 4, my + 2)

        # draw nodes
        for node, (x, y) in positions.items():
            pygame.draw.circle(self.screen, (80, 100, 140), (x, y), 18)
            pygame.draw.circle(self.screen, (220, 220, 240), (x, y), 18, 2)

            mv = node.move_from_parent if node.move_from_parent is not None else "ROOT"
            self.draw_text(str(mv), x - 14, y - 32)
            self.draw_text(f"V={node.visits}", x - 20, y - 8)
            self.draw_text(f"Q={q_value_fn(node):.2f}", x - 24, y + 8)

        self.draw_text(
            f"Iter {iteration} | depth={depth} | SPACE pause | ESC quit",
            20,
            20,
        )

        pygame.display.flip()
        self.clock.tick(60)
        return True
