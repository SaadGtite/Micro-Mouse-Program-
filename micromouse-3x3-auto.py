#!/usr/bin/env python3
"""
MicroMouse Solver - Version Automatique pour Zone 3×3
Détecte automatiquement une zone 3×3 fermée avec une seule ouverture
Utilise le système de grillage mural (bitmask) et A* avec pénalité de virage
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
from typing import List, Tuple, Set, Optional
import os


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
        # La grille stocke maintenant les masques binaires des murs
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        self.cell_h = 0
        self.cell_w = 0

    def load_and_process(self):
        """Charge et traite l'image"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Impossible de charger: {self.image_path}")
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Binarisation inversée (murs = blanc = 255)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Nettoyer légèrement
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
        """
        Détecte les MURS entre les cellules.
        Stocke un masque binaire (bitmask) pour chaque cellule.
        1: NORD, 2: EST, 4: SUD, 8: OUEST
        """
        if self.binary_image is None:
            self.load_and_process()
        
        h, w = self.binary_image.shape
        
        # Épaisseur supposée des murs pour l'échantillonnage
        wall_thickness_y = max(2, int(self.cell_h * 0.1))
        wall_thickness_x = max(2, int(self.cell_w * 0.1))
        
        # Marge pour éviter les coins
        margin_y = int(self.cell_h * 0.2)
        margin_x = int(self.cell_w * 0.2)
        
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        print("\n✓ Détection des murs (N, E, S, O) pour chaque cellule...")
        
        for i in range(self.grid_size):  # Ligne (y)
            for j in range(self.grid_size):  # Colonne (x)
                
                # Coordonnées des frontières de la cellule
                y_north_wall = i * self.cell_h
                y_south_wall = (i + 1) * self.cell_h
                x_west_wall = j * self.cell_w
                x_east_wall = (j + 1) * self.cell_w
                
                # Échantillonnage pour le mur NORD
                if i == 0 or self._check_wall_region(
                    y_north_wall - wall_thickness_y, y_north_wall + wall_thickness_y,
                    x_west_wall + margin_x, x_east_wall - margin_x
                ):
                    self.grid[i, j] |= 1  # Mur NORD
                
                # Échantillonnage pour le mur SUD
                if i == self.grid_size - 1 or self._check_wall_region(
                    y_south_wall - wall_thickness_y, y_south_wall + wall_thickness_y,
                    x_west_wall + margin_x, x_east_wall - margin_x
                ):
                    self.grid[i, j] |= 4  # Mur SUD
                
                # Échantillonnage pour le mur OUEST
                if j == 0 or self._check_wall_region(
                    y_north_wall + margin_y, y_south_wall - margin_y,
                    x_west_wall - wall_thickness_x, x_west_wall + wall_thickness_x
                ):
                    self.grid[i, j] |= 8  # Mur OUEST
                
                # Échantillonnage pour le mur EST
                if j == self.grid_size - 1 or self._check_wall_region(
                    y_north_wall + margin_y, y_south_wall - margin_y,
                    x_east_wall - wall_thickness_x, x_east_wall + wall_thickness_x
                ):
                    self.grid[i, j] |= 2  # Mur EST
        
        # Compter les passages ouverts
        open_passages = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                walls = self.grid[i, j]
                if not (walls & 1) and i > 0: open_passages += 1  # Nord ouvert
                if not (walls & 2) and j < self.grid_size - 1: open_passages += 1  # Est ouvert
        
        print(f"✓ Grille {self.grid_size}×{self.grid_size} analysée.")
        print(f"  • {open_passages} passages internes ouverts détectés.")
        
        return self.grid


class GoalZoneDetector:
    """
    Détecte automatiquement une zone 3×3 fermée avec une seule ouverture
    Tous les passages internes sont ouverts (aucun mur entre les cellules de la zone)
    """
    def __init__(self, grid):
        self.grid = grid
        self.size = len(grid)
    
    def find_3x3_goal_zone(self) -> Optional[Tuple[Tuple[int, int], Set[Tuple[int, int]], Tuple[int, int]]]:
        print("\n✓ Recherche de zone 3×3 fermée avec une seule ouverture et sans mur interne...")
        for top_y in range(self.size - 2):
            for left_x in range(self.size - 2):
                result = self._check_3x3_zone(top_y, left_x)
                if result:
                    center, goal_set, entry_point = result
                    print(f"✓ Zone 3×3 trouvée!")
                    print(f"  • Centre: {center}")
                    print(f"  • Point d'entrée: {entry_point}")
                    print(f"  • Cellules de la zone: {sorted(goal_set)}")
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

        # Vérification A: Exactement une ouverture
        if len(openings) != 1:
            return None

        # Vérification B: Aucun mur entre les cellules internes
        for x, y in zone_cells:
            walls = self.grid[y, x]
            # NORD: Adjacent interne
            if (x, y - 1) in zone_cells and ((walls & 1) or (self.grid[y - 1, x] & 4)):
                return None
            # SUD
            if (x, y + 1) in zone_cells and ((walls & 4) or (self.grid[y + 1, x] & 1)):
                return None
            # OUEST
            if (x - 1, y) in zone_cells and ((walls & 8) or (self.grid[y, x - 1] & 2)):
                return None
            # EST
            if (x + 1, y) in zone_cells and ((walls & 2) or (self.grid[y, x + 1] & 8)):
                return None

        center = (left_x + 1, top_y + 1)
        entry_point = openings[0]
        return (center, zone_cells, entry_point)



class MazeSolverWith3x3Goal:
    """
    Résolveur pour atteindre une zone 3×3 avec une seule ouverture.
    Utilise A* avec pénalité de virage.
    """

    def __init__(self, grid, start_pos: Tuple[int, int], goal_zone_info: Tuple[Tuple[int, int], Set[Tuple[int, int]], Tuple[int, int]]):
        self.grid = grid
        self.size = len(grid)
        self.start = start_pos
        
        # Décomposer les informations de la zone d'arrivée
        self.goal_center, self.goal_set, self.entry_point = goal_zone_info
        self.goal = None  # Sera défini quand A* atteindra la zone
        
        print(f"\n✓ Configuration du solveur (grille {self.size}×{self.size}):")
        print(f"  • Départ: {self.start}")
        print(f"  • Zone 3×3 centrée sur: {self.goal_center}")
        print(f"  • Point d'entrée: {self.entry_point}")

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Retourne les voisins accessibles en utilisant le masque binaire de murs.
        """
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
        """
        Algorithme A* modifié pour atteindre n'importe quelle cellule de la zone 3×3.
        """

        def heuristic(pos: Tuple[int, int]) -> int:
            """Heuristique: distance de Manhattan au point d'entrée"""
            return abs(pos[0] - self.entry_point[0]) + abs(pos[1] - self.entry_point[1])

        # open_set: (f_score, g_score, (x, y), (parent_x, parent_y))
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
            
            # Succès: on atteint n'importe quelle cellule de la zone 3×3
            if current in self.goal_set:
                self.goal = current
                path_found = True
                break
            
            # Direction depuis le parent
            current_dir = None
            if parent:
                current_dir = (current[0] - parent[0], current[1] - parent[1])
            
            for neighbor in self.get_neighbors(current):
                if neighbor == parent or neighbor in closed_set:
                    continue
                
                # Coût du mouvement
                move_cost = 1
                neighbor_dir = (neighbor[0] - current[0], neighbor[1] - current[1])
                
                # Pénalité de virage
                if current_dir and current_dir != neighbor_dir:
                    move_cost += turn_penalty
                
                tentative_g = g + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))
        
        # Reconstruire le chemin
        path = []
        if path_found:
            current = self.goal
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            path.reverse()
        
        return path, list(closed_set)

    def extract_arcs(self, path):
        """Extrait les arcs (segments droits) du chemin"""
        if len(path) < 2:
            return []
        
        arcs = []
        arc_start = path[0]
        current_direction = None
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            direction = (x2 - x1, y2 - y1)
            
            if current_direction is None:
                current_direction = direction
            elif direction != current_direction:
                arc_length = abs(path[i][0] - arc_start[0]) + abs(path[i][1] - arc_start[1]) + 1
                
                arcs.append({
                    'id': len(arcs) + 1,
                    'start': arc_start,
                    'end': path[i],
                    'direction': current_direction,
                    'length': arc_length,
                    'cells': self._get_cells_in_arc(arc_start, path[i])
                })
                
                arc_start = path[i]
                current_direction = direction
        
        # Dernier arc
        arc_length = abs(path[-1][0] - arc_start[0]) + abs(path[-1][1] - arc_start[1]) + 1
        arcs.append({
            'id': len(arcs) + 1,
            'start': arc_start,
            'end': path[-1],
            'direction': current_direction,
            'length': arc_length,
            'cells': self._get_cells_in_arc(arc_start, path[-1])
        })
        
        return arcs

    def _get_cells_in_arc(self, start, end):
        """Retourne toutes les cellules dans un arc"""
        cells = []
        x1, y1 = start
        x2, y2 = end
        
        if y1 == y2:  # Mouvement horizontal
            step = 1 if x2 > x1 else -1
            for x in range(x1, x2 + step, step):
                cells.append((x, y1))
        elif x1 == x2:  # Mouvement vertical
            step = 1 if y2 > y1 else -1
            for y in range(y1, y2 + step, step):
                cells.append((x1, y))
        
        return cells


class AdvancedVisualizer:
    """Visualiseur avec affichage complet"""

    def __init__(self, detector: WallMazeDetector, solver: MazeSolverWith3x3Goal):
        self.detector = detector
        self.solver = solver
        self.size = solver.size

    def visualize_complete(self, path, arcs, explored):
        """Crée une visualisation complète avec la zone 3×3"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # === IMAGE ORIGINALE ===
        ax1 = axes[0]
        img_rgb = cv2.cvtColor(self.detector.original_image, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_rgb)
        ax1.set_title('Image Originale avec Chemin vers Zone 3×3', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        h, w = img_rgb.shape[:2]
        cell_h = h / self.size
        cell_w = w / self.size
        
        # Dessiner la zone 3×3
        for gx, gy in self.solver.goal_set:
            rect = Rectangle((gx * cell_w, gy * cell_h), cell_w, cell_h,
                           facecolor='yellow', alpha=0.3, edgecolor='gold', linewidth=3)
            ax1.add_patch(rect)
        
        # Marquer le point d'entrée
        ex, ey = self.solver.entry_point
        ax1.plot((ex + 0.5) * cell_w, (ey + 0.5) * cell_h, 'y*', 
                markersize=30, markeredgecolor='red', markeredgewidth=3, label='Entrée Zone', zorder=10)
        
        # Dessiner le chemin
        if len(path) > 1:
            path_px = [(p[0] + 0.5) * cell_w for p in path]
            path_py = [(p[1] + 0.5) * cell_h for p in path]
            
            ax1.plot(path_px, path_py, 'b-', linewidth=5, alpha=0.8, label=f'Chemin ({len(path)} cellules)')
            ax1.plot(path_px, path_py, 'yo', markersize=8, markeredgecolor='orange', markeredgewidth=2)
            
            # Numéros des arcs
            for arc in arcs:
                mid_x = ((arc['start'][0] + arc['end'][0]) / 2 + 0.5) * cell_w
                mid_y = ((arc['start'][1] + arc['end'][1]) / 2 + 0.5) * cell_h
                ax1.text(mid_x, mid_y, str(arc['id']),
                        fontsize=14, fontweight='bold', color='white',
                        bbox=dict(boxstyle='circle,pad=0.3', facecolor='red',
                                 edgecolor='darkred', linewidth=2))
            
            # Start et Goal
            ax1.plot(path_px[0], path_py[0], 'go', markersize=25,
                    markeredgecolor='darkgreen', markeredgewidth=4, label='Départ', zorder=10)
            ax1.plot(path_px[-1], path_py[-1], 'ro', markersize=25,
                    markeredgecolor='darkred', markeredgewidth=4, label='Arrivée', zorder=10)
            
            ax1.legend(fontsize=14, loc='upper right')
        
        # === GRILLE DÉTECTÉE ===
        ax2 = axes[1]
        ax2.set_facecolor('white')
        
        # Dessiner les murs
        for i in range(self.size):
            for j in range(self.size):
                walls = self.detector.grid[i, j]
                x, y = j, i
                
                if walls & 1:  # NORD
                    ax2.plot([x, x + 1], [y, y], 'k-', linewidth=4, zorder=5)
                if walls & 2:  # EST
                    ax2.plot([x + 1, x + 1], [y, y + 1], 'k-', linewidth=4, zorder=5)
                if walls & 4:  # SUD
                    ax2.plot([x, x + 1], [y + 1, y + 1], 'k-', linewidth=4, zorder=5)
                if walls & 8:  # OUEST
                    ax2.plot([x, x], [y, y + 1], 'k-', linewidth=4, zorder=5)
        
        # Cellules explorées
        for pos in explored:
            if pos != self.solver.start and pos not in self.solver.goal_set:
                rect = Rectangle((pos[0], pos[1]), 1, 1,
                               facecolor='lightyellow', alpha=0.5, zorder=1)
                ax2.add_patch(rect)
        
        # Zone 3×3
        for g in self.solver.goal_set:
            rect = Rectangle((g[0], g[1]), 1, 1,
                           facecolor='gold', alpha=0.4, edgecolor='orange', linewidth=2, zorder=2)
            ax2.add_patch(rect)
        
        # Point d'entrée
        ex, ey = self.solver.entry_point
        ax2.plot(ex + 0.5, ey + 0.5, 'y*', markersize=25,
                markeredgecolor='red', markeredgewidth=3, label='Entrée Zone', zorder=10)
        
        # Chemin
        if len(path) > 1:
            px = [p[0] + 0.5 for p in path]
            py = [p[1] + 0.5 for p in path]
            ax2.plot(px, py, 'b-', linewidth=4, alpha=0.8, zorder=6)
            ax2.plot(px, py, 'yo', markersize=10, markeredgecolor='orange', markeredgewidth=2, zorder=7)
            
            # Numéros des arcs
            for arc in arcs:
                mid_x = (arc['start'][0] + arc['end'][0]) / 2 + 0.5
                mid_y = (arc['start'][1] + arc['end'][1]) / 2 + 0.5
                ax2.text(mid_x, mid_y, str(arc['id']),
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor='red',
                                 edgecolor='darkred', linewidth=2), zorder=8)
        
        # Start et Goal
        sx, sy = self.solver.start
        ax2.plot(sx + 0.5, sy + 0.5, 'go', markersize=20,
                markeredgecolor='darkgreen', markeredgewidth=3, label='Départ', zorder=10)
        
        if self.solver.goal:
            gx, gy = self.solver.goal
            ax2.plot(gx + 0.5, gy + 0.5, 'ro', markersize=20,
                    markeredgecolor='darkred', markeredgewidth=3, label='Arrivée', zorder=10)
        
        # Grille
        major_ticks = np.arange(0, self.size + 1, 1)
        minor_ticks = np.arange(0, self.size + 1, 0.5)
        
        ax2.set_xticks(major_ticks)
        ax2.set_yticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.set_yticks(minor_ticks, minor=True)
        
        ax2.grid(which='major', alpha=0.4, linestyle='-', zorder=0)
        ax2.grid(which='minor', alpha=0.2, linestyle=':', zorder=0)
        
        ax2.tick_params(axis='x', which='minor', bottom=False, labelbottom=False)
        ax2.tick_params(axis='y', which='minor', left=False, labelleft=False)
        
        if self.size > 20:
            for i, label in enumerate(ax2.get_xticklabels()):
                if i % 2 != 0 and i != 0:
                    label.set_visible(False)
            for i, label in enumerate(ax2.get_yticklabels()):
                if i % 2 != 0 and i != 0:
                    label.set_visible(False)
        
        ax2.set_xlim(0, self.size)
        ax2.set_ylim(0, self.size)
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        ax2.set_title(f'Grille {self.size}×{self.size} - Zone 3×3 avec {len(arcs)} Arcs', 
                     fontsize=16, fontweight='bold')
        ax2.legend(fontsize=14)
        
        plt.tight_layout()
        return fig


def save_arcs_report(path, arcs, goal_info, output_file='outputs/arcs_zone_3x3.txt'):
    """Sauvegarde un rapport détaillé"""
    
    direction_symbols = {
        (0, -1): '↑ NORD',
        (0, 1): '↓ SUD',
        (1, 0): '→ EST',
        (-1, 0): '← OUEST',
        None: '? INCONNU'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*90 + "\n")
        f.write(" "*20 + "🤖 RAPPORT DE NAVIGATION VERS ZONE 3×3\n")
        f.write("="*90 + "\n\n")
        
        f.write(f"📍 INFORMATIONS SUR LA ZONE 3×3\n")
        f.write("-"*90 + "\n")
        center, goal_set, entry = goal_info
        f.write(f"  • Centre de la zone : {center}\n")
        f.write(f"  • Point d'entrée    : {entry}\n")
        f.write(f"  • Cellules de la zone: {sorted(goal_set)}\n\n")
        
        f.write(f"📊 STATISTIQUES DU CHEMIN\n")
        f.write("-"*90 + "\n")
        f.write(f"  • Longueur totale   : {len(path)} cellules\n")
        f.write(f"  • Nombre d'arcs     : {len(arcs)}\n")
        f.write(f"  • Nombre de virages : {len(arcs) - 1}\n")
        
        if arcs:
            avg_length = sum(arc['length'] for arc in arcs) / len(arcs)
            f.write(f"  • Longueur moy/arc  : {avg_length:.1f} cellules\n")
        
        f.write("\n" + "="*90 + "\n\n")
        
        for arc in arcs:
            direction = direction_symbols.get(arc['direction'], '?')
            
            f.write(f"📍 ARC #{arc['id']}\n")
            f.write("-"*90 + "\n")
            f.write(f"  Départ    : {arc['start']}\n")
            f.write(f"  Arrivée   : {arc['end']}\n")
            f.write(f"  Direction : {direction}\n")
            f.write(f"  Longueur  : {arc['length']} cellules\n")
            f.write(f"  Cellules  : {arc['cells']}\n\n")
        
        f.write("="*90 + "\n")
        f.write("📋 CHEMIN COMPLET\n")
        f.write("="*90 + "\n\n")
        
        for i, pos in enumerate(path):
            marker = ""
            if pos == path[0]:
                marker = " ← DÉPART"
            elif pos in goal_set:
                marker = " ← ZONE 3×3"
            if pos == entry:
                marker += " (ENTRÉE)"
            
            f.write(f"  Étape {i+1:3d}: {pos}{marker}\n")


def main(image_path: str, 
         grid_size: int = 32, 
         start_pos: Tuple[int, int] = (1, 30),
         auto_detect_goal: bool = True):
    """
    Fonction principale avec détection automatique de la zone 3×3
    
    Args:
        image_path: Chemin vers l'image du labyrinthe
        grid_size: Taille de la grille (16, 32, etc.)
        start_pos: Position de départ (x, y)
        auto_detect_goal: Si True, détecte automatiquement la zone 3×3
    """
    
    print("\n" + "="*90)
    print(" "*15 + f"🤖 MICROMOUSE SOLVER - ZONE 3×3 AUTOMATIQUE - {grid_size}×{grid_size}")
    print("="*90)
    
    os.makedirs('outputs', exist_ok=True)
    
    # Étape 1: Détection des murs
    print("\n[ÉTAPE 1] DÉTECTION DES MURS")
    print("-"*90)
    detector = WallMazeDetector(image_path, grid_size=grid_size)
    detector.load_and_process()
    grid = detector.detect_walls()
    
    # Étape 2: Détection de la zone 3×3
    print("\n[ÉTAPE 2] DÉTECTION DE LA ZONE 3×3")
    print("-"*90)
    
    if auto_detect_goal:
        goal_detector = GoalZoneDetector(grid)
        goal_zone_info = goal_detector.find_3x3_goal_zone()
        
        if goal_zone_info is None:
            print("\n❌ ERREUR: Aucune zone 3×3 valide trouvée!")
            print("    Vérifiez que votre labyrinthe contient une zone 3×3 avec exactement 1 ouverture.")
            return
    else:
        print("✗ Détection automatique désactivée")
        return
    
    # Étape 3: Résolution
    print("\n[ÉTAPE 3] RÉSOLUTION AVEC A*")
    print("-"*90)
    solver = MazeSolverWith3x3Goal(grid, start_pos, goal_zone_info)
    path, explored = solver.a_star(turn_penalty=5)
    
    if len(path) == 0:
        print("\n❌ ERREUR: Aucun chemin trouvé!")
        return
    
    print(f"\n✓ Résolution réussie!")
    print(f"  • Chemin trouvé     : {len(path)} cellules")
    print(f"  • Cellules explorées: {len(explored)}")
    print(f"  • Efficacité        : {100*len(path)/len(explored):.1f}%")
    
    # Étape 4: Extraction des arcs
    print("\n[ÉTAPE 4] EXTRACTION DES ARCS")
    print("-"*90)
    arcs = solver.extract_arcs(path)
    print(f"✓ {len(arcs)} arcs extraits, {len(arcs)-1} virages")
    
    # Étape 5: Visualisation
    print("\n[ÉTAPE 5] VISUALISATION")
    print("-"*90)
    viz = AdvancedVisualizer(detector, solver)
    fig = viz.visualize_complete(path, arcs, explored)
    
    output_file = f'outputs/solution_3x3_{grid_size}x{grid_size}.png'
    fig.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Image: {output_file}")
    plt.close()
    
    # Étape 6: Rapport
    print("\n[ÉTAPE 6] RAPPORT")
    print("-"*90)
    report_file = f'outputs/rapport_3x3_{grid_size}x{grid_size}.txt'
    save_arcs_report(path, arcs, goal_zone_info, output_file=report_file)
    print(f"✓ Rapport: {report_file}")
    
    print("\n" + "="*90)
    print(" "*30 + "✅ TERMINÉ AVEC SUCCÈS!")
    print("="*90 + "\n")


if __name__ == "__main__":
    
    # Configuration pour votre labyrinthe 32×32
    print("\n--- TEST AVEC ZONE 3×3 AUTOMATIQUE ---")
    
    image_path = "maze3.png"
    
    if not os.path.exists(image_path):
        print(f"ERREUR: Image '{image_path}' introuvable")
        print("Assurez-vous que l'image est dans le même dossier que le script.")
    else:
        main(
            image_path=image_path,
            grid_size=32,
            start_pos=(1, 30),  # Point de départ
            auto_detect_goal=True  # Détection automatique de la zone 3×3
        )
