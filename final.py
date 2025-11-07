#!/usr/bin/env python3
"""
HYBRID GRID MICROMOUSE SOLVER (16×16 ou 32×32)
- Toujours en GRILLE (pas de pixels)
- Départ : coin à une seule ouverture (auto-détecté)
- Zone 3×3 auto-détectée
- Animation 3 phases + stats + image finale
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import heapq
import os
from typing import Tuple, List, Set, Optional

# ====================== DÉTECTION COIN DÉPART ======================
def get_single_exit_corner(grid: np.ndarray) -> Tuple[int, int]:
    size = grid.shape[0]
    corners = [(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)]
    for x, y in corners:
        walls = grid[y, x]
        if bin(walls).count('1') == 3:          # 3 murs → 1 seule ouverture
            return (x, y)
    # Fallback : coin avec le plus de murs
    best, max_w = None, -1
    for x, y in corners:
        w = bin(grid[y, x]).count('1')
        if w > max_w:
            max_w = w
            best = (x, y)
    return best

# ====================== DÉTECTEUR DE MURS (GRILLE) ======================
class WallMazeDetector:
    def __init__(self, image_path: str, grid_size: int = 16):
        self.image_path = image_path
        self.grid_size = grid_size
        self.original = None
        self.binary = None
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.cell_h = self.cell_w = 0

    def load_and_process(self):
        self.original = cv2.imread(self.image_path)
        if self.original is None:
            raise ValueError(f"Impossible de charger {self.image_path}")
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.binary = binary
        h, w = binary.shape
        self.cell_h, self.cell_w = h / self.grid_size, w / self.grid_size
        return binary

    def _check_wall(self, y1, y2, x1, x2, thresh=0.35):
        y1 = max(0, int(y1)); y2 = min(self.binary.shape[0], int(y2))
        x1 = max(0, int(x1)); x2 = min(self.binary.shape[1], int(x2))
        region = self.binary[y1:y2, x1:x2]
        return np.mean(region == 255) > thresh if region.size > 0 else False

    def detect_walls(self):
        if self.binary is None:
            self.load_and_process()
        thick_y = max(2, int(self.cell_h * 0.12))
        thick_x = max(2, int(self.cell_w * 0.12))
        margin_y = int(self.cell_h * 0.22)
        margin_x = int(self.cell_w * 0.22)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                yn = i * self.cell_h
                ys = (i+1) * self.cell_h
                xw = j * self.cell_w
                xe = (j+1) * self.cell_w

                if i == 0 or self._check_wall(yn - thick_y, yn + thick_y, xw + margin_x, xe - margin_x):
                    self.grid[i, j] |= 1   # NORD
                if i == self.grid_size-1 or self._check_wall(ys - thick_y, ys + thick_y, xw + margin_x, xe - margin_x):
                    self.grid[i, j] |= 4   # SUD
                if j == 0 or self._check_wall(yn + margin_y, ys - margin_y, xw - thick_x, xw + thick_x):
                    self.grid[i, j] |= 8   # OUEST
                if j == self.grid_size-1 or self._check_wall(yn + margin_y, ys - margin_y, xe - thick_x, xe + thick_x):
                    self.grid[i, j] |= 2   # EST
        return self.grid

# ====================== DÉTECTEUR ZONE 3×3 ======================
class GoalZoneDetector:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.size = grid.shape[0]

    def find_3x3(self) -> Optional[Tuple[Tuple[int,int], Set[Tuple[int,int]], Tuple[int,int]]]:
        for ty in range(self.size - 2):
            for lx in range(self.size - 2):
                zone = {(lx+dx, ty+dy) for dy in range(3) for dx in range(3)}
                openings = []
                for x,y in zone:
                    w = self.grid[y,x]
                    if not (w & 1) and y > 0 and (x,y-1) not in zone: openings.append((x,y))
                    if not (w & 4) and y < self.size-1 and (x,y+1) not in zone: openings.append((x,y))
                    if not (w & 8) and x > 0 and (x-1,y) not in zone: openings.append((x,y))
                    if not (w & 2) and x < self.size-1 and (x+1,y) not in zone: openings.append((x,y))
                if len(openings) != 1: continue

                # Vérif murs internes absents
                valid = True
                for x,y in zone:
                    w = self.grid[y,x]
                    if (x,y-1) in zone and ((w & 1) or (self.grid[y-1,x] & 4)): valid = False
                    if (x,y+1) in zone and ((w & 4) or (self.grid[y+1,x] & 1)): valid = False
                    if (x-1,y) in zone and ((w & 8) or (self.grid[y,x-1] & 2)): valid = False
                    if (x+1,y) in zone and ((w & 2) or (self.grid[y,x+1] & 8)): valid = False
                if valid:
                    center = (lx+1, ty+1)
                    entry = openings[0]
                    return center, zone, entry
        return None

# ====================== SOLVEUR GRILLE UNIVERSEL ======================
class GridMazeSolver:
    def __init__(self, detector: WallMazeDetector):
        self.detector = detector
        self.grid = detector.grid
        self.size = detector.grid_size
        self.start = get_single_exit_corner(self.grid)
        goal_info = GoalZoneDetector(self.grid).find_3x3()
        if not goal_info:
            raise ValueError("Zone 3×3 non trouvée")
        self.goal_center, self.goal_set, self.entry = goal_info

    def neighbors(self, pos):
        x, y = pos
        w = self.grid[y, x]
        nei = []
        if y > 0 and not (w & 1) and not (self.grid[y-1, x] & 4): nei.append((x, y-1))
        if y < self.size-1 and not (w & 4) and not (self.grid[y+1, x] & 1): nei.append((x, y+1))
        if x > 0 and not (w & 8) and not (self.grid[y, x-1] & 2): nei.append((x-1, y))
        if x < self.size-1 and not (w & 2) and not (self.grid[y, x+1] & 8): nei.append((x+1, y))
        return nei

    def a_star(self, turn_penalty=5, return_path=False):
        def h(p): return abs(p[0]-self.entry[0]) + abs(p[1]-self.entry[1])
        open_set = [(h(self.start), 0, self.start, None)]
        came_from = {}
        g_score = {self.start: 0}
        closed = set()

        while open_set:
            _, g, cur, parent = heapq.heappop(open_set)
            if cur in closed: continue
            closed.add(cur)
            came_from[cur] = parent

            if cur in self.goal_set:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = came_from.get(cur)
                path.reverse()
                if return_path:
                    return path[::-1], closed  # retour
                return path, closed

            prev_dir = None if parent is None else (cur[0]-parent[0], cur[1]-parent[1])
            for nei in self.neighbors(cur):
                if nei == parent or nei in closed: continue
                dir_new = (nei[0]-cur[0], nei[1]-cur[1])
                cost = 1 + (turn_penalty if prev_dir and prev_dir != dir_new else 0)
                tent_g = g + cost
                if nei not in g_score or tent_g < g_score[nei]:
                    g_score[nei] = tent_g
                    f = tent_g + h(nei)
                    heapq.heappush(open_set, (f, tent_g, nei, cur))
        return [], closed

# ====================== ANIMATION UNIFIÉE ======================
def unified_animation(detector: WallMazeDetector, solver: GridMazeSolver,
                      path_to_goal: List[Tuple[int,int]],
                      path_return: List[Tuple[int,int]],
                      speed_path: List[Tuple[int,int]],
                      speeds: List[float]):
    print("\nLancement de l'animation 3 phases...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    size = solver.size
    cell_h, cell_w = detector.cell_h, detector.cell_w

    def draw_base(ax):
        ax.set_facecolor('lightgray')
        for i in range(size):
            for j in range(size):
                walls = detector.grid[i, j]
                x, y = j, i
                if walls & 1: ax.plot([x, x+1], [y, y], 'k-', lw=3)
                if walls & 2: ax.plot([x+1, x+1], [y, y+1], 'k-', lw=3)
                if walls & 4: ax.plot([x, x+1], [y+1, y+1], 'k-', lw=3)
                if walls & 8: ax.plot([x, x], [y, y+1], 'k-', lw=3)
        # Zone 3×3
        for gx, gy in solver.goal_set:
            rect = Rectangle((gx, gy), 1, 1, facecolor='gold', alpha=0.4, edgecolor='orange', lw=2)
            ax.add_patch(rect)
        # Entrée
        ex, ey = solver.entry
        ax.plot(ex+0.5, ey+0.5, 'y*', ms=25, markeredgecolor='red', markeredgewidth=3)
        # Départ
        sx, sy = solver.start
        ax.plot(sx+0.5, sy+0.5, 'go', ms=20, markeredgecolor='darkgreen', markeredgewidth=3)
        ax.set_xlim(0, size); ax.set_ylim(0, size)
        ax.set_aspect('equal'); ax.invert_yaxis()
        ax.set_xticks(np.arange(0, size+1, 1))
        ax.set_yticks(np.arange(0, size+1, 1))
        ax.grid(True, alpha=0.3)

    for ax in axs: draw_base(ax)
    axs[0].set_title("Exploration → Goal", fontsize=14, fontweight='bold')
    axs[1].set_title("Retour au départ", fontsize=14, fontweight='bold')
    axs[2].set_title("Speed Run Optimisé", fontsize=14, fontweight='bold', color='cyan')

    line1, = axs[0].plot([], [], 'b-', lw=3, alpha=0.8)
    robot1, = axs[0].plot([], [], 'ro', ms=12, markeredgecolor='darkred', markeredgewidth=2)
    line2, = axs[1].plot([], [], 'orange', lw=3, alpha=0.8)
    robot2, = axs[1].plot([], [], 'ro', ms=12, markeredgecolor='darkred', markeredgewidth=2)
    line3, = axs[2].plot([], [], lw=3, alpha=0.8)
    robot3, = axs[2].plot([], [], 'co', ms=14, markeredgecolor='cyan', markeredgewidth=3)

    skip = max(1, len(path_to_goal)//150)
    frames = (len(path_to_goal) + len(path_return) + len(speed_path)) // skip

    def anim(frame):
        p1_end = len(path_to_goal)//skip
        p2_end = p1_end + len(path_return)//skip

        if frame < p1_end:
            idx = min(frame*skip, len(path_to_goal)-1)
            px = [p[0]+0.5 for p in path_to_goal[:idx+1]]
            py = [p[1]+0.5 for p in path_to_goal[:idx+1]]
            line1.set_data(px, py)
            robot1.set_data([px[-1]], [py[-1]])
            axs[0].set_title(f"Exploration [{idx+1}/{len(path_to_goal)}]")
        elif frame < p2_end:
            idx = min((frame-p1_end)*skip, len(path_return)-1)
            # Trace exploration en fond
            px1 = [p[0]+0.5 for p in path_to_goal]
            py1 = [p[1]+0.5 for p in path_to_goal]
            axs[1].plot(px1, py1, 'b-', lw=2, alpha=0.4)
            # Trace retour
            px = [p[0]+0.5 for p in path_return[:idx+1]]
            py = [p[1]+0.5 for p in path_return[:idx+1]]
            line2.set_data(px, py)
            robot2.set_data([px[-1]], [py[-1]])
            axs[1].set_title(f"Retour [{idx+1}/{len(path_return)}]")
        else:
            idx = min((frame-p2_end)*skip, len(speed_path)-1)
            # Fond
            px1 = [p[0]+0.5 for p in path_to_goal]
            py1 = [p[1]+0.5 for p in path_to_goal]
            axs[2].plot(px1, py1, 'b-', lw=1.5, alpha=0.3)
            px2 = [p[0]+0.5 for p in path_return]
            py2 = [p[1]+0.5 for p in path_return]
            axs[2].plot(px2, py2, 'orange', lw=1.5, alpha=0.3)
            # Speed run avec couleur vitesse
            max_s = max(speeds) if speeds else 1
            for i in range(idx):
                ratio = speeds[i] / max_s
                color = (1-ratio, ratio, 0.2)
                axs[2].plot([speed_path[i][0]+0.5, speed_path[i+1][0]+0.5],
                            [speed_path[i][1]+0.5, speed_path[i+1][1]+0.5],
                            color=color, lw=3)
            robot3.set_data([speed_path[idx][0]+0.5], [speed_path[idx][1]+0.5])
            axs[2].set_title(f"Speed Run [{idx+1}/{len(speed_path)}]", color='cyan', fontweight='bold')
        return line1, robot1, line2, robot2, line3, robot3

    anim_obj = FuncAnimation(fig, anim, frames=frames, interval=40, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()

# ====================== MAIN ======================
def main():
    print("\n" + "="*75)
    print(" HYBRID GRID MICROMOUSE SOLVER (16×16 ou 32×32) ".center(75))
    print("="*75)
    print(" • Grille uniquement (pas de pixels)")
    print(" • Départ : coin à 1 ouverture")
    print(" • Zone 3×3 auto-détectée")
    print(" • Animation 3 phases + stats")
    print()

    path = input("Chemin image labyrinthe : ").strip()
    if not os.path.exists(path):
        print("Image introuvable !")
        return

    choice = input("Taille ? [16] ou [32] (défaut 32) : ").strip()
    size = 16 if choice == "16" else 32
    print(f"\nMode {size}×{size} activé")

    detector = WallMazeDetector(path, grid_size=size)
    print("Détection des murs...")
    detector.detect_walls()

    solver = GridMazeSolver(detector)
    print(f"Départ : {solver.start}")
    print(f"Zone 3×3 centrée en {solver.goal_center} (entrée {solver.entry})")

    # Phase 1 : exploration
    print("Phase 1 : Exploration → Goal")
    path_to_goal, explored1 = solver.a_star(turn_penalty=5)

    # Phase 2 : retour
    print("Phase 2 : Retour au départ")
    path_return, _ = solver.a_star(turn_penalty=5, return_path=True)

    # Phase 3 : speed run optimisé (moins de virages)
    print("Phase 3 : Speed Run (virages minimisés)")
    speed_path, explored3 = solver.a_star(turn_penalty=12)  # forte pénalité → moins de virages

    # Calcul vitesses simulées
    speeds = []
    cur_speed = 0
    max_speed = 3.0
    for i in range(len(speed_path)-1):
        # virage ?
        if i > 0:
            v1 = (speed_path[i][0] - speed_path[i-1][0], speed_path[i][1] - speed_path[i-1][1])
            v2 = (speed_path[i+1][0] - speed_path[i][0], speed_path[i+1][1] - speed_path[i][1])
            turn = abs(v1[0]*v2[1] - v1[1]*v2[0]) > 0  # produit vectoriel
        else:
            turn = False
        target = max_speed if not turn else max_speed * 0.5
        cur_speed = cur_speed * 0.9 + target * 0.1
        speeds.append(cur_speed)
    speeds.append(speeds[-1] if speeds else max_speed)

    print(f"\nRÉSUMÉ")
    print(f"Exploration   : {len(path_to_goal)} cellules")
    print(f"Retour        : {len(path_return)} cellules")
    print(f"Speed Run     : {len(speed_path)} cellules (optimisé)")
    print(f"Explorées     : {len(explored1 | explored3)} cellules")

    # Animation
    unified_animation(detector, solver, path_to_goal, path_return, speed_path, speeds)

    # Image finale
    fig, ax = plt.subplots(figsize=(10,10))
    for i in range(size):
        for j in range(size):
            walls = detector.grid[i,j]
            x,y = j,i
            if walls & 1: ax.plot([x,x+1],[y,y],'k-',lw=3)
            if walls & 2: ax.plot([x+1,x+1],[y,y+1],'k-',lw=3)
            if walls & 4: ax.plot([x,x+1],[y+1,y+1],'k-',lw=3)
            if walls & 8: ax.plot([x,x],[y,y+1],'k-',lw=3)
    for g in solver.goal_set:
        ax.add_patch(Rectangle((g[0],g[1]),1,1,facecolor='gold',alpha=0.5,edgecolor='orange',lw=2))
    px = [p[0]+0.5 for p in speed_path]
    py = [p[1]+0.5 for p in speed_path]
    ax.plot(px, py, 'cyan', lw=4, alpha=0.9)
    ax.plot(px, py, 'yo', ms=8, mec='orange', mew=2)
    ax.plot(solver.start[0]+0.5, solver.start[1]+0.5, 'go', ms=20, mec='darkgreen', mew=3)
    ax.set_title(f"Solution {size}×{size} - Chemin optimal : {len(speed_path)} cellules", fontsize=16)
    ax.set_xlim(0,size); ax.set_ylim(0,size); ax.set_aspect('equal'); ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("solution_finale_grid.png", dpi=200, bbox_inches='tight')
    print("\nImage sauvegardée : solution_finale_grid.png")
    print("TERMINE !")

if __name__ == "__main__":
    main()