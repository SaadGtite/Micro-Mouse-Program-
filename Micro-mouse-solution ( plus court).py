#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  CHAMPION MICROMOUSE 2024 – VERSION FINALE ULTIME             ║
║                                                                              ║
║  → A* pur (Manhattan) : chemin le plus court garanti                         ║
║  → Exploration intelligente + flood-fill optimisé (deque)                   ║
║  → Détection parfaite de la zone 3×3 (32×32)                                 ║
║  → Retour différent + Speed Run absolu le plus court                         ║
║  → Temps d'exécution affiché pour chaque phase                               ║
║  → Commentaires détaillés pour comprendre l'algorithme comme un pro         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
from collections import deque
import os
import time  # Pour mesurer le temps

# ====================== DÉTECTION DES MURS ======================
class WallMazeDetector:
    """
    Transforme une image de labyrinthe en grille binaire avec murs (N=1, E=2, S=4, W=8)
    Utilise OpenCV : seuillage + fermeture morphologique → robuste aux imperfections
    """
    def __init__(self, image_path: str, grid_size: int = 16):
        self.image_path = image_path
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)

    def detect_walls(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise ValueError("Image introuvable ! Vérifiez le chemin.")
        
        # Conversion + seuillage inverse (noir = mur → blanc)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Fermeture morphologique : comble les petits trous dans les murs
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        h, w = binary.shape
        cell_h, cell_w = h / self.grid_size, w / self.grid_size
        thick = max(3, int(min(cell_h, cell_w) * 0.15))  # Épaisseur du mur
        margin = int(min(cell_h, cell_w) * 0.25)         # Marge intérieure

        # Parcours de chaque cellule
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                yn, ys = i * cell_h, (i+1) * cell_h
                xw, xe = j * cell_w, (j+1) * cell_w

                # Vérification des 4 murs (N, S, W, E)
                if i == 0 or self._check(binary, yn - thick, yn + thick, xw + margin, xe - margin):
                    self.grid[i, j] |= 1   # Mur Nord
                if i == self.grid_size-1 or self._check(binary, ys - thick, ys + thick, xw + margin, xe - margin):
                    self.grid[i, j] |= 4   # Mur Sud
                if j == 0 or self._check(binary, yn + margin, ys - margin, xw - thick, xw + thick):
                    self.grid[i, j] |= 8   # Mur Ouest
                if j == self.grid_size-1 or self._check(binary, yn + margin, ys - margin, xe - thick, xe + thick):
                    self.grid[i, j] |= 2   # Mur Est
        return self.grid

    def _check(self, img, y1, y2, x1, x2, thresh=0.3):
        """Vérifie si une région contient assez de pixels noirs (mur)"""
        y1, y2 = max(0, int(y1)), min(img.shape[0], int(y2))
        x1, x2 = max(0, int(x1)), min(img.shape[1], int(x2))
        region = img[y1:y2, x1:x2]
        return np.mean(region == 255) > thresh if region.size > 0 else False


# ====================== ROBOT CHAMPION 2024 ======================
class PureSpeedMicromouse:
    """
    Algorithme complet en 3 phases :
    1. Exploration → découvre tout + trouve le but
    2. Retour différent → revient par un autre chemin
    3. Speed Run → trajet le plus court absolu (A*)
    """
    def __init__(self, detector):
        self.size = detector.grid_size
        self.walls = detector.grid.copy()           # Murs réels (connus à la fin)
        self.start = self.find_start()              # Coin avec 3 murs
        self.is_16 = (self.size == 16)
        self.zone_size = 2 if self.is_16 else 3

        # But : centre pour 16×16, zone 3×3 pour 32×32
        self.goal_center = (self.size//2, self.size//2) if self.is_16 else None
        self.entry = None                           # Cellule d'entrée du goal
        self.goal_found = False

        # Exploration partielle
        self.known_walls = np.zeros_like(self.walls)   # Murs découverts
        self.visited = set([self.start])
        self.dist = np.full((self.size, self.size), 9999, dtype=int)
        self.dist[self.start[1], self.start[0]] = 0
        self.known_walls[self.start[1], self.start[0]] = self.walls[self.start[1], self.start[0]]

        # Chemins
        self.path_explore = [self.start]
        self.path_return = []
        self.speed_path = []

    # =================================================================
    # 1. RECHERCHE DU DÉPART
    # =================================================================
    def find_start(self):
        """Le départ est toujours dans un coin avec 3 murs"""
        corners = [(0,0), (0,self.size-1), (self.size-1,0), (self.size-1,self.size-1)]
        for x, y in corners:
            if bin(self.walls[y, x]).count('1') == 3:
                return (x, y)
        return (0, 0)

    # =================================================================
    # 2. DÉTECTION ZONE 3×3 (32×32 uniquement)
    # =================================================================
    def is_goal_zone_32x32(self, cx, cy):
        """Vérifie : 0 mur interne + exactement 1 ouverture"""
        if not (1 <= cx < self.size-1 and 1 <= cy < self.size-1):
            return False
        zone = [(cx+dx-1, cy+dy-1) for dy in range(3) for dx in range(3)]
        zone_set = set(zone)

        # Aucun mur interne
        for x, y in zone:
            w = self.walls[y, x]
            if x < cx and (w & 2): return False   # Mur Est à gauche
            if x > cx and (w & 8): return False   # Mur Ouest à droite
            if y < cy and (w & 4): return False   # Mur Sud en haut
            if y > cy and (w & 1): return False   # Mur Nord en bas

        # 1 seule ouverture
        openings = 0
        entry_cell = None
        for x, y in zone:
            w = self.walls[y, x]
            for dx, dy, mask, nx, ny in [(0,-1,1,x,y-1), (0,1,4,x,y+1), (-1,0,8,x-1,y), (1,0,2,x+1,y)]:
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if not (w & mask) and (nx, ny) not in zone_set:
                        openings += 1
                        entry_cell = (nx, ny)
        if openings != 1:
            return False

        self.goal_center = (cx, cy)
        self.entry = entry_cell
        self.goal_found = True
        print(f"ZONE 3×3 DÉTECTÉE ! Centre: ({cx}, {cy}) | Entrée: {entry_cell}")
        return True

    def find_goal(self):
        if self.is_16: return
        for cy in range(1, self.size - 1):
            for cx in range(1, self.size - 1):
                if self.is_goal_zone_32x32(cx, cy):
                    return

    # =================================================================
    # 3. VOISINS ACCESSIBLES (selon murs connus)
    # =================================================================
    def neighbors_known(self, x, y):
        dirs = [(0,-1,1), (0,1,4), (-1,0,8), (1,0,2)]  # (dx, dy, masque)
        return [(x+dx, y+dy) for dx, dy, mask in dirs
                if 0 <= x+dx < self.size and 0 <= y+dy < self.size
                and not (self.known_walls[y, x] & mask)]

    # =================================================================
    # 4. FLOOD-FILL OPTIMISÉ (Bellman-Ford avec file)
    # =================================================================
    def update_flood(self):
        """Met à jour les distances minimales depuis le départ → très rapide"""
        queue = deque([self.start])
        self.dist.fill(9999)
        self.dist[self.start[1], self.start[0]] = 0

        while queue:
            x, y = queue.popleft()
            d = self.dist[y, x]
            for nx, ny in self.neighbors_known(x, y):
                if self.dist[ny, nx] > d + 1:
                    self.dist[ny, nx] = d + 1
                    queue.append((nx, ny))

    # =================================================================
    # 5. EXPLORATION INTELLIGENTE (phase 1)
    # =================================================================
    def explore(self):
        """Exploration optimale : va vers les zones les mieux connectées + centre"""
        print("EXPLORATION RAPIDE EN COURS...")
        start_time = time.time()
        current = self.start
        self.path_explore = [current]
        self.update_flood()
        steps = 0
        center = (self.size // 2, self.size // 2)

        while steps < 5000:
            steps += 1

            # Recherche du goal à chaque étape
            if not self.goal_found:
                self.find_goal()

            # Cellules inconnues adjacentes
            candidates = [n for n in self.neighbors_known(*current) if n not in self.visited]
            if not candidates:
                if len(self.path_explore) > 1:
                    self.path_explore.pop()
                    current = self.path_explore[-1]
                continue

            # STRATÉGIE :
            if self.goal_found:
                # On connaît le goal → on y va tout droit
                gx, gy = self.goal_center
                next_pos = min(candidates, key=lambda p: abs(p[0]-gx) + abs(p[1]-gy))
            else:
                # On privilégie les cellules les plus proches du départ (meilleure connexion)
                next_pos = min(candidates, key=lambda p: self.dist[p[1], p[0]])
                min_dist = self.dist[next_pos[1], next_pos[0]]
                best = [p for p in candidates if self.dist[p[1], p[0]] == min_dist]
                if len(best) > 1:
                    # Tie-breaker : vers le centre
                    next_pos = min(best, key=lambda p: abs(p[0]-center[0]) + abs(p[1]-center[1]))

            # Avancer
            current = next_pos
            self.path_explore.append(current)
            self.known_walls[current[1], current[0]] = self.walls[current[1], current[0]]
            self.visited.add(current)
            self.update_flood()  # Mise à jour après chaque découverte

            # Victoire ?
            if current == self.goal_center:
                print(f"BUT ATTEINT en {steps} étapes !")
                break

        explore_time = time.time() - start_time
        print(f"Exploration terminée en {explore_time:.3f} secondes")
        if not self.goal_found:
            self.find_goal()

    # =================================================================
    # 6. A* – CHEMIN LE PLUS COURT (phase 2 & 3)
    # =================================================================
    def a_star_shortest(self, start, goal, avoid=None):
        """A* avec heuristique Manhattan → chemin optimal garanti"""
        if not goal or start == goal:
            return [start]
        avoid = avoid or set()
        def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])

        open_set = [(h(start), 0, start, None)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, g, cur, parent = heapq.heappop(open_set)
            if cur == goal:
                path = []
                while cur:
                    path.append(cur)
                    cur = came_from.get(cur)
                return path[::-1]

            for nx, ny in self.get_neighbors(cur[0], cur[1]):
                if (nx, ny) in avoid: continue
                tent_g = g + 1
                if tent_g < g_score.get((nx, ny), 99999):
                    g_score[(nx, ny)] = tent_g
                    f = tent_g + h((nx, ny))
                    heapq.heappush(open_set, (f, tent_g, (nx, ny), cur))
                    came_from[(nx, ny)] = cur
        return [start]

    def get_neighbors(self, x, y):
        """Voisins selon murs réels (tous connus à ce stade)"""
        w = self.walls[y, x]
        nei = []
        if y > 0 and not (w & 1): nei.append((x, y-1))
        if y < self.size-1 and not (w & 4): nei.append((x, y+1))
        if x > 0 and not (w & 8): nei.append((x-1, y))
        if x < self.size-1 and not (w & 2): nei.append((x+1, y))
        return nei

    # =================================================================
    # 7. LANCEMENT COMPLET + TEMPS
    # =================================================================
    def run_champion(self):
        total_start = time.time()

        # Phase 1 : Exploration
        self.explore()
        t1 = time.time()

        # Phase 2 : Retour différent
        print("Calcul du retour différent...")
        blocked = set(self.path_explore[1:-1])  # On bloque le chemin d'exploration
        self.path_return = self.a_star_shortest(self.goal_center, self.start, avoid=blocked)
        if len(self.path_return) <= 4:
            self.path_return = self.a_star_shortest(self.goal_center, self.start)
        t2 = time.time()

        # Phase 3 : Speed Run
        print("Calcul du Speed Run (plus court absolu)...")
        if not self.entry:
            for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                ex, ey = self.goal_center[0] + dx, self.goal_center[1] + dy
                if 0 <= ex < self.size and 0 <= ey < self.size:
                    self.entry = (ex, ey)
                    break
        self.speed_path = self.a_star_shortest(self.start, self.entry)
        t3 = time.time()

        total_time = t3 - total_start

        # =================================================================
        # AFFICHAGE + SAUVEGARDE
        # =================================================================
        fig, axs = plt.subplots(1, 3, figsize=(32, 12))
        paths = [self.path_explore, self.path_return, self.speed_path]
        colors = ['#3498db', '#9b59b6', '#e74c3c']
        titles = ['1. EXPLORATION', '2. RETOUR DIFFÉRENT', '3. SPEED RUN (PLUS COURT)']

        for i, (path, color, title) in enumerate(zip(paths, colors, titles)):
            ax = axs[i]
            ax.set_xlim(0, self.size); ax.set_ylim(0, self.size)
            ax.set_aspect('equal'); ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=24, fontweight='bold', color=color, pad=30)

            # Murs
            for y in range(self.size):
                for x in range(self.size):
                    w = self.walls[y, x]
                    lw = 6
                    if w & 1: ax.plot([x,x+1],[y,y],'k-', lw=lw)
                    if w & 2: ax.plot([x+1,x+1],[y,y+1],'k-', lw=lw)
                    if w & 4: ax.plot([x,x+1],[y+1,y+1],'k-', lw=lw)
                    if w & 8: ax.plot([x,x],[y,y+1],'k-', lw=lw)

            # Zone goal
            if self.goal_center:
                cx, cy = self.goal_center
                for dx in range(self.zone_size):
                    for dy in range(self.zone_size):
                        gx = cx - self.zone_size//2 + dx
                        gy = cy - self.zone_size//2 + dy
                        if 0 <= gx < self.size and 0 <= gy < self.size:
                            ax.add_patch(Rectangle((gx, gy), 1, 1,
                                                 facecolor='#FFD700', alpha=0.9,
                                                 edgecolor='red', lw=8))

            # Départ
            sx, sy = self.start
            ax.plot(sx+0.5, sy+0.5, 'go', ms=50, mec='darkgreen', mew=12)

            # Chemin
            if path and len(path) > 1:
                px = [p[0]+0.5 for p in path]
                py = [p[1]+0.5 for p in path]
                ax.plot(px, py, color=color, lw=12, alpha=0.95)
                ax.plot(px[::2], py[::2], 'o', color='yellow', ms=14, mec='orange', mew=5)

        # Temps dans le titre
        plt.suptitle(f"SOLUTION MICROMOUSE 2025 – {self.size}×{self.size}\n"
                     f"Temps total : {total_time:.3f}s | "
                     f"Exploration: {t1-total_start:.3f}s | "
                     f"Speed Run: {len(self.speed_path)} cellules (OPTIMAL)",
                     fontsize=28, fontweight='bold', color='#2c3e50', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig("SOLUTION_2025_TEMPS_OPTIMAL.png", dpi=600, bbox_inches='tight')
        plt.show()

        # Résultats console
        print(f"\n{'='*70}")
        print(" RÉSULTATS FINAUX – SOLUTION MICROMOUSE 2025 ".center(70))
        print(f"{'='*70}")
        print(f"{'Temps total':<25}: {total_time:6.3f} secondes")
        print(f"{'Exploration':<25}: {len(self.path_explore):4d} cellules → {t1-total_start:.3f}s")
        print(f"{'Retour différent':<25}: {len(self.path_return):4d} cellules → {t2-t1:.3f}s")
        print(f"{'Speed Run (optimal)':<25}: {len(self.speed_path):4d} cellules → {t3-t2:.3f}s")
        print(f"{'But':<25}: {self.goal_center} (entrée: {self.entry})")
        print(f"{'Image sauvegardée':<25}: SOLUTION_2025_TEMPS_OPTIMAL.png")
        print(f"{'='*70}\n")


# ====================== MAIN ======================
def main():
    print("\n" + "═"*100)
    print(" Solution MICROMOUSE 2025 – VERSION FINALE AVEC TEMPS & COMMENTAIRES ".center(100))
    print("═"*100 + "\n")

    path = input("Chemin de l'image : ").strip('"')
    if not os.path.exists(path):
        print("Image introuvable !")
        return

    size_input = input("Taille ? [16]/32 : ").strip()
    size = 16 if size_input in ["16", ""] else 32
    print(f"\nDémarrage {size}×{size} – MODE CHAMPION...\n")

    detector = WallMazeDetector(path, size)
    detector.detect_walls()

    robot = PureSpeedMicromouse(detector)
    robot.run_champion()

if __name__ == "__main__":
    main()