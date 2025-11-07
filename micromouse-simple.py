#!/usr/bin/env python3
"""
MicroMouse Solver - Version Simplifiée avec Simulation
Détecte automatiquement une zone 3×3 fermée avec une seule ouverture
Départ : coin ayant une seule ouverture (3 murs)
Affiche une simulation en direct et sauvegarde une image de la solution
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import heapq
from typing import List, Tuple, Set, Optional

def get_single_exit_corner(grid):
    size = grid.shape[0]
    coins = [
        (0, 0),
        (0, size - 1),
        (size - 1, 0),
        (size - 1, size - 1)
    ]
    for (x, y) in coins:
        walls = grid[y, x]
        # 3 murs -> une seule ouverture
        if bin(walls).count("1") == 3:
            return (x, y)
    raise ValueError("Aucun coin avec une seule ouverture trouvé")

class WallMazeDetector:
    """
    Détecte les murs du labyrinthe (et non les cellules-murs)
    Utilise un masque binaire pour stocker les murs de chaque cellule.
    1: NORD, 2: EST, 4: SUD, 8: OUEST
    """
    def __init__(self, image_path: str, grid_size: int = 16):
        self.image_path = image_path
        self.grid_size = grid_size
        self.original_image = None
        self.binary_image = None
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.cell_h = 0
        self.cell_w = 0

    def load_and_process(self):
        """Charge et traite l'image"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Impossible de charger: {self.image_path}")

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.binary_image = binary

        h, w = self.binary_image.shape
        self.cell_h = h / self.grid_size
        self.cell_w = w / self.grid_size

        print(f"✓ Image chargée: {self.original_image.shape}")
        print(f"✓ Ratio murs (blanc): {np.mean(binary==255):.1%}")
        return binary

    def _check_wall_region(self, y_start, y_end, x_start, x_end, threshold=0.4):
        """Vérifie si une région contient un mur"""
        y_start, y_end = int(max(0, y_start)), int(min(self.binary_image.shape[0], y_end))
        x_start, x_end = int(max(0, x_start)), int(min(self.binary_image.shape[1], x_end))
        region = self.binary_image[y_start:y_end, x_start:x_end]
        if region.size == 0:
            return False
        wall_ratio = np.mean(region == 255)
        return wall_ratio > threshold

    def detect_walls(self):
        """Détecte les MURS entre les cellules."""
        if self.binary_image is None:
            self.load_and_process()

        h, w = self.binary_image.shape
        wall_thickness_y = max(2, int(self.cell_h * 0.1))
        wall_thickness_x = max(2, int(self.cell_w * 0.1))
        margin_y = int(self.cell_h * 0.2)
        margin_x = int(self.cell_w * 0.2)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        print("\n✓ Détection des murs (N, E, S, O) pour chaque cellule...")

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y_north_wall = i * self.cell_h
                y_south_wall = (i + 1) * self.cell_h
                x_west_wall = j * self.cell_w
                x_east_wall = (j + 1) * self.cell_w

                # Mur NORD
                if i == 0 or self._check_wall_region(
                    y_north_wall - wall_thickness_y, y_north_wall + wall_thickness_y,
                    x_west_wall + margin_x, x_east_wall - margin_x
                ):
                    self.grid[i, j] |= 1
                # Mur SUD
                if i == self.grid_size - 1 or self._check_wall_region(
                    y_south_wall - wall_thickness_y, y_south_wall + wall_thickness_y,
                    x_west_wall + margin_x, x_east_wall - margin_x
                ):
                    self.grid[i, j] |= 4
                # Mur OUEST
                if j == 0 or self._check_wall_region(
                    y_north_wall + margin_y, y_south_wall - margin_y,
                    x_west_wall - wall_thickness_x, x_west_wall + wall_thickness_x
                ):
                    self.grid[i, j] |= 8
                # Mur EST
                if j == self.grid_size - 1 or self._check_wall_region(
                    y_north_wall + margin_y, y_south_wall - margin_y,
                    x_east_wall - wall_thickness_x, x_east_wall + wall_thickness_x
                ):
                    self.grid[i, j] |= 2

        open_passages = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                walls = self.grid[i, j]
                if not (walls & 1) and i > 0: open_passages += 1
                if not (walls & 2) and j < self.grid_size - 1: open_passages += 1

        print(f"✓ Grille {self.grid_size}×{self.grid_size} analysée.")
        print(f"  • {open_passages} passages internes ouverts détectés.")
        return self.grid

class GoalZoneDetector:
    """Détecte automatiquement une zone 3×3 fermée avec une seule ouverture"""
    def __init__(self, grid):
        self.grid = grid
        self.size = len(grid)

    def find_3x3_goal_zone(self) -> Optional[Tuple[Tuple[int, int], Set[Tuple[int, int]], Tuple[int, int]]]:
        print("\n✓ Recherche de zone 3×3 fermée avec une seule ouverture...")

        for top_y in range(self.size - 2):
            for left_x in range(self.size - 2):
                result = self._check_3x3_zone(top_y, left_x)
                if result:
                    center, goal_set, entry_point = result
                    print(f"✓ Zone 3×3 trouvée!")
                    print(f"  • Centre: {center}")
                    print(f"  • Point d'entrée: {entry_point}")
                    return result

        print("✗ Aucune zone 3×3 valide trouvée")
        return None

    def _check_3x3_zone(self, top_y: int, left_x: int) -> Optional[Tuple[Tuple[int, int], Set[Tuple[int, int]], Tuple[int, int]]]:
        zone_cells = set()
        for dy in range(3):
            for dx in range(3):
                zone_cells.add((left_x + dx, top_y + dy))

        openings = []
        for x, y in zone_cells:
            walls = self.grid[y, x]
            # NORD
            if not (walls & 1) and y > 0:
                neighbor = (x, y - 1)
                if neighbor not in zone_cells:
                    openings.append((x, y))
            # SUD
            if not (walls & 4) and y < self.size - 1:
                neighbor = (x, y + 1)
                if neighbor not in zone_cells:
                    openings.append((x, y))
            # OUEST
            if not (walls & 8) and x > 0:
                neighbor = (x - 1, y)
                if neighbor not in zone_cells:
                    openings.append((x, y))
            # EST
            if not (walls & 2) and x < self.size - 1:
                neighbor = (x + 1, y)
                if neighbor not in zone_cells:
                    openings.append((x, y))

        if len(openings) != 1:
            return None

        # Vérification: Aucun mur interne
        for x, y in zone_cells:
            walls = self.grid[y, x]
            if (x, y - 1) in zone_cells and ((walls & 1) or (self.grid[y - 1, x] & 4)):
                return None
            if (x, y + 1) in zone_cells and ((walls & 4) or (self.grid[y + 1, x] & 1)):
                return None
            if (x - 1, y) in zone_cells and ((walls & 8) or (self.grid[y, x - 1] & 2)):
                return None
            if (x + 1, y) in zone_cells and ((walls & 2) or (self.grid[y, x + 1] & 8)):
                return None

        center = (left_x + 1, top_y + 1)
        entry_point = openings[0]
        return (center, zone_cells, entry_point)

class MazeSolverWith3x3Goal:
    """Résolveur pour atteindre une zone 3×3 avec une seule ouverture."""
    def __init__(self, grid, start_pos: Tuple[int, int], goal_zone_info: Tuple[Tuple[int, int], Set[Tuple[int, int]], Tuple[int, int]]):
        self.grid = grid
        self.size = len(grid)
        self.start = start_pos
        self.goal_center, self.goal_set, self.entry_point = goal_zone_info
        self.goal = None

        print(f"\n✓ Configuration du solveur:")
        print(f"  • Départ: {self.start}")
        print(f"  • Zone 3×3 centrée sur: {self.goal_center}")
        print(f"  • Point d'entrée: {self.entry_point}")

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        walls = self.grid[y, x]
        neighbors = []
        # NORD (y-1)
        if y > 0 and not (walls & 1):
            if not (self.grid[y - 1, x] & 4):
                neighbors.append((x, y - 1))
        # SUD (y+1)
        if y < self.size - 1 and not (walls & 4):
            if not (self.grid[y + 1, x] & 1):
                neighbors.append((x, y + 1))
        # OUEST (x-1)
        if x > 0 and not (walls & 8):
            if not (self.grid[y, x - 1] & 2):
                neighbors.append((x - 1, y))
        # EST (x+1)
        if x < self.size - 1 and not (walls & 2):
            if not (self.grid[y, x + 1] & 8):
                neighbors.append((x + 1, y))
        return neighbors

    def a_star(self, turn_penalty: int = 5) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        def heuristic(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - self.entry_point[0]) + abs(pos[1] - self.entry_point[1])

        open_set = [(heuristic(self.start), 0, self.start, None)]
        came_from = {}
        g_score = {self.start: 0}
        closed_set = set()

        print(f"  • Lancement de A* avec pénalité de virage = {turn_penalty}")
        path_found = False

        while open_set:
            f, g, current, parent = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            came_from[current] = parent
            if current in self.goal_set:
                self.goal = current
                path_found = True
                break
            current_dir = None
            if parent:
                current_dir = (current[0] - parent[0], current[1] - parent[1])
            for neighbor in self.get_neighbors(current):
                if neighbor == parent or neighbor in closed_set:
                    continue
                move_cost = 1
                neighbor_dir = (neighbor[0] - current[0], neighbor[1] - current[1])
                if current_dir and current_dir != neighbor_dir:
                    move_cost += turn_penalty
                tentative_g = g + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))
        path = []
        if path_found:
            current = self.goal
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            path.reverse()
        return path, list(closed_set)

class AnimationVisualizer:
    """Visualiseur avec animation en direct"""
    def __init__(self, detector: WallMazeDetector, solver: MazeSolverWith3x3Goal):
        self.detector = detector
        self.solver = solver
        self.size = solver.size

    def create_animation(self, path, explored):
        print("\n✓ Affichage de l'animation...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_facecolor('white')
        # Base: dessiner les murs
        for i in range(self.size):
            for j in range(self.size):
                walls = self.detector.grid[i, j]
                x, y = j, i
                if walls & 1:  # NORD
                    ax.plot([x, x + 1], [y, y], 'k-', linewidth=3, zorder=5)
                if walls & 2:  # EST
                    ax.plot([x + 1, x + 1], [y, y + 1], 'k-', linewidth=3, zorder=5)
                if walls & 4:  # SUD
                    ax.plot([x, x + 1], [y + 1, y + 1], 'k-', linewidth=3, zorder=5)
                if walls & 8:  # OUEST
                    ax.plot([x, x], [y, y + 1], 'k-', linewidth=3, zorder=5)
        # Dessiner la zone 3×3
        for g in self.solver.goal_set:
            rect = Rectangle((g[0], g[1]), 1, 1, facecolor='gold', alpha=0.3, edgecolor='orange', linewidth=2, zorder=2)
            ax.add_patch(rect)
        # Point d'entrée
        ex, ey = self.solver.entry_point
        ax.plot(ex + 0.5, ey + 0.5, 'y*', markersize=25, markeredgecolor='red', markeredgewidth=3, label='Entrée Zone', zorder=10)
        # Start
        sx, sy = self.solver.start
        ax.plot(sx + 0.5, sy + 0.5, 'go', markersize=20, markeredgecolor='darkgreen', markeredgewidth=3, label='Départ', zorder=10)
        # Préparer l'animation
        skip = max(1, len(path) // 100)
        total_frames = len(path) // skip
        path_line, = ax.plot([], [], 'b-', linewidth=3, alpha=0.8, zorder=6)
        path_dots, = ax.plot([], [], 'yo', markersize=8, markeredgecolor='orange', markeredgewidth=2, zorder=7)
        robot, = ax.plot([], [], 'ro', markersize=15, markeredgecolor='darkred', markeredgewidth=3, zorder=10)
        def animate(frame):
            current_idx = min(frame * skip, len(path) - 1)
            px = [p[0] + 0.5 for p in path[:current_idx+1]]
            py = [p[1] + 0.5 for p in path[:current_idx+1]]
            path_line.set_data(px, py)
            path_dots.set_data(px, py)
            robot.set_data([path[current_idx][0] + 0.5], [path[current_idx][1] + 0.5])
            ax.set_title(f'Résolution du Labyrinthe [{current_idx+1}/{len(path)} cellules]', fontsize=16, fontweight='bold')
            return path_line, path_dots, robot
        major_ticks = np.arange(0, self.size + 1, 1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(which='major', alpha=0.3, linestyle='-', zorder=0)
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.legend(fontsize=12)
        anim = FuncAnimation(fig, animate, frames=total_frames, interval=50, blit=True, repeat=True)
        plt.tight_layout()
        print("  • Animation en cours... (fermez la fenêtre pour continuer)")
        plt.show()
        return anim

    def save_solution_image(self, path, explored, filename='maze_solution.png'):
        print(f"\n✓ Création de l'image de solution...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_facecolor('white')
        # Dessiner les murs
        for i in range(self.size):
            for j in range(self.size):
                walls = self.detector.grid[i, j]
                x, y = j, i
                if walls & 1:
                    ax.plot([x, x + 1], [y, y], 'k-', linewidth=3, zorder=5)
                if walls & 2:
                    ax.plot([x + 1, x + 1], [y, y + 1], 'k-', linewidth=3, zorder=5)
                if walls & 4:
                    ax.plot([x, x + 1], [y + 1, y + 1], 'k-', linewidth=3, zorder=5)
                if walls & 8:
                    ax.plot([x, x], [y, y + 1], 'k-', linewidth=3, zorder=5)
        # Cellules explorées
        for pos in explored:
            if pos != self.solver.start and pos not in self.solver.goal_set:
                rect = Rectangle((pos[0], pos[1]), 1, 1, facecolor='lightyellow', alpha=0.5, zorder=1)
                ax.add_patch(rect)
        # Zone 3×3
        for g in self.solver.goal_set:
            rect = Rectangle((g[0], g[1]), 1, 1, facecolor='gold', alpha=0.4, edgecolor='orange', linewidth=2, zorder=2)
            ax.add_patch(rect)
        # Point d'entrée
        ex, ey = self.solver.entry_point
        ax.plot(ex + 0.5, ey + 0.5, 'y*', markersize=25, markeredgecolor='red', markeredgewidth=3, label='Entrée Zone', zorder=10)
        # Chemin
        if len(path) > 1:
            px = [p[0] + 0.5 for p in path]
            py = [p[1] + 0.5 for p in path]
            ax.plot(px, py, 'b-', linewidth=4, alpha=0.8, zorder=6)
            ax.plot(px, py, 'yo', markersize=10, markeredgecolor='orange', markeredgewidth=2, zorder=7)
        # Start et Goal
        sx, sy = self.solver.start
        ax.plot(sx + 0.5, sy + 0.5, 'go', markersize=20, markeredgecolor='darkgreen', markeredgewidth=3, label='Départ', zorder=10)
        if self.solver.goal:
            gx, gy = self.solver.goal
            ax.plot(gx + 0.5, gy + 0.5, 'ro', markersize=20, markeredgecolor='darkred', markeredgewidth=3, label='Arrivée', zorder=10)
        major_ticks = np.arange(0, self.size + 1, 1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(which='major', alpha=0.3, linestyle='-', zorder=0)
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        stats_text = f"Chemin: {len(path)} cellules | Explorées: {len(explored)} | Efficacité: {100*len(path)/len(explored):.1f}%"
        ax.set_title(f'Solution du Labyrinthe {self.size}×{self.size}\n{stats_text}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Image sauvegardée: {filename}")
        plt.close()

def main(image_path: str, grid_size: int = 32):
    print("\n" + "="*80)
    print(f"{'🤖 MICROMOUSE SOLVER - SIMULATION SIMPLIFIÉE':^80}")
    print("="*80)
    # Étape 1: Détection des murs
    print("\n[1] DÉTECTION DES MURS")
    detector = WallMazeDetector(image_path, grid_size=grid_size)
    detector.load_and_process()
    grid = detector.detect_walls()
    # Étape 2: Détection de la zone 3×3
    print("\n[2] DÉTECTION DE LA ZONE 3×3")
    goal_detector = GoalZoneDetector(grid)
    goal_zone_info = goal_detector.find_3x3_goal_zone()
    if goal_zone_info is None:
        print("\n❌ ERREUR: Aucune zone 3×3 valide trouvée!")
        return
    # Détection automatique du coin à une seule ouverture
    print("\n[3] DÉTECTION DU DÉPART (COIN À UNE SEULE OUVERTURE)")
    start_pos = get_single_exit_corner(grid)
    print(f"✓ Point de départ choisi : {start_pos}")
    # Étape 4: Résolution
    print("\n[4] RÉSOLUTION AVEC A*")
    solver = MazeSolverWith3x3Goal(grid, start_pos, goal_zone_info)
    path, explored = solver.a_star(turn_penalty=5)
    if len(path) == 0:
        print("\n❌ ERREUR: Aucun chemin trouvé!")
        return
    print(f"\n✓ Résolution réussie!")
    print(f"  • Chemin: {len(path)} cellules")
    print(f"  • Explorées: {len(explored)} cellules")
    print(f"  • Efficacité: {100*len(path)/len(explored):.1f}%")
    # Étape 5: Animation et visualisation
    print("\n[5] VISUALISATION")
    viz = AnimationVisualizer(detector, solver)
    viz.create_animation(path, explored)
    viz.save_solution_image(path, explored, filename='maze_solution.png')
    print("\n" + "="*80)
    print(f"{'✅ TERMINÉ AVEC SUCCÈS!':^80}")
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Entrez le chemin de l'image du labyrinthe: ").strip()
    if not image_path:
        print("❌ Aucun chemin d'image fourni!")
        sys.exit(1)
    import os
    if not os.path.exists(image_path):
        print(f"❌ ERREUR: Image '{image_path}' introuvable")
        sys.exit(1)
    main(
        image_path=image_path,
        grid_size=32  # ou 16 selon le labyrinthe
    )
