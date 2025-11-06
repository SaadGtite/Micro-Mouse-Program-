# Micromouse Maze Solver - Optimized with Diagonal Movement
# Allows 45° turns for smoother, faster navigation

import cv2
import numpy as np
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.morphology import medial_axis
import time
import math

class OptimizedMazeSolver:
    def __init__(self, image_path):
        """Initialize the optimized maze solver with diagonal movement."""
        self.image_path = image_path
        self.original_image = None
        self.binary_maze = None
        self.skeleton = None
        self.distance_transform = None
        self.start_pixel = None
        self.goal_region = None
        self.goal_pixel = None
        
    def load_and_process_image(self):
        """Load image and convert to binary."""
        print("Loading maze image...")
        
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        self.binary_maze = (binary > 127).astype(np.uint8)
        
        print(f"Maze size: {self.binary_maze.shape[0]}x{self.binary_maze.shape[1]} pixels")
        print(f"Path pixels: {np.sum(self.binary_maze)}")
        
        return self.binary_maze
    
    def extract_path_centerline(self):
        """Extract skeleton of paths."""
        print("\nExtracting path centerlines...")
        
        skeleton, distance = medial_axis(self.binary_maze, return_distance=True)
        self.skeleton = skeleton.astype(np.uint8)
        self.distance_transform = distance
        
        print(f"Skeleton pixels: {np.sum(self.skeleton)}")
        return self.skeleton, distance
    
    def find_start_position(self):
        """Find start at bottom-left."""
        print("\nFinding start position...")
        
        height, width = self.skeleton.shape
        
        for y in range(height - 1, max(height - 50, 0), -1):
            for x in range(0, min(50, width)):
                if self.skeleton[y, x] == 1:
                    self.start_pixel = (y, x)
                    print(f"Start: {self.start_pixel}")
                    return self.start_pixel
        
        skeleton_pixels = np.argwhere(self.skeleton == 1)
        if len(skeleton_pixels) > 0:
            skeleton_pixels = skeleton_pixels[np.lexsort((skeleton_pixels[:, 1], -skeleton_pixels[:, 0]))]
            self.start_pixel = tuple(skeleton_pixels[0])
            return self.start_pixel
        
        raise ValueError("Could not find start!")
    
    def find_goal_region(self):
        """Find goal region in center."""
        print("\nFinding goal region...")
        
        height, width = self.skeleton.shape
        center_y, center_x = height // 2, width // 2
        search_radius = min(height, width) // 4
        
        goal_candidates = []
        for y in range(max(0, center_y - search_radius), min(height, center_y + search_radius)):
            for x in range(max(0, center_x - search_radius), min(width, center_x + search_radius)):
                if self.skeleton[y, x] == 1:
                    dist_from_center = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                    score = self.distance_transform[y, x] - dist_from_center * 0.1
                    goal_candidates.append((score, y, x))
        
        goal_candidates.sort(reverse=True)
        
        self.goal_region = []
        for i in range(min(20, len(goal_candidates))):
            score, y, x = goal_candidates[i]
            self.goal_region.append((y, x))
        
        self.goal_pixel = (goal_candidates[0][1], goal_candidates[0][2])
        print(f"Goal: {self.goal_pixel}")
        return self.goal_region
    
    def get_skeleton_neighbors(self, pixel):
        """Get neighboring skeleton pixels (8-connected for diagonal movement)."""
        y, x = pixel
        neighbors = []
        
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        # This allows 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315° movements
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                
                ny, nx = y + dy, x + dx
                
                if (0 <= ny < self.skeleton.shape[0] and 
                    0 <= nx < self.skeleton.shape[1] and 
                    self.skeleton[ny, nx] == 1):
                    neighbors.append((ny, nx))
        
        return neighbors
    
    def pixel_distance(self, p1, p2):
        """Euclidean distance."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_turn_angle(self, prev_pos, current_pos, next_pos):
        """
        Calculate turn angle between three positions.
        Returns angle in degrees (0° = straight, 45° = diagonal, 90° = right angle, etc.)
        """
        if prev_pos is None:
            return 0
        
        # Vector from prev to current
        v1 = (current_pos[0] - prev_pos[0], current_pos[1] - prev_pos[1])
        # Vector from current to next
        v2 = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
        
        # Calculate angle
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        det = v1[0]*v2[1] - v1[1]*v2[0]
        angle = math.degrees(math.atan2(det, dot))
        
        return abs(angle)
    
    def is_diagonal_move(self, pos1, pos2):
        """Check if move is diagonal (45°)."""
        dy = abs(pos2[0] - pos1[0])
        dx = abs(pos2[1] - pos1[1])
        return dy == 1 and dx == 1
    
    def a_star_exploration(self, start, goal_pixels):
        """A* with center preference (exploration)."""
        counter = 0
        open_set = [(0, counter, start, [start])]
        heapq.heapify(open_set)
        
        g_score = {start: 0}
        visited = set()
        
        while open_set:
            f, _, current, path = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current in goal_pixels:
                return path, visited
            
            for neighbor in self.get_skeleton_neighbors(current):
                if neighbor not in visited:
                    distance_cost = self.pixel_distance(current, neighbor)
                    center_bonus = self.distance_transform[neighbor] * 0.3
                    tentative_g = g_score[current] + distance_cost - center_bonus
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        h = min(self.pixel_distance(neighbor, g) for g in goal_pixels)
                        f_score = tentative_g + h
                        
                        counter += 1
                        new_path = path + [neighbor]
                        heapq.heappush(open_set, (f_score, counter, neighbor, new_path))
        
        return None, visited
    
    def a_star_optimal_with_angles(self, start, goal_pixels):
        """
        Optimized A* that considers turn angles.
        Diagonal moves (45°) are preferred over right-angle turns (90°).
        """
        counter = 0
        # State: (position, previous_position) to track direction
        open_set = [(0, counter, start, None, [start])]
        heapq.heapify(open_set)
        
        g_score = {(start, None): 0}
        closed = set()
        
        while open_set:
            f, _, current, prev, path = heapq.heappop(open_set)
            
            state = (current, prev)
            if state in closed:
                continue
            
            closed.add(state)
            
            if current in goal_pixels:
                return path
            
            for neighbor in self.get_skeleton_neighbors(current):
                neighbor_state = (neighbor, current)
                
                if neighbor_state not in closed:
                    # Base distance cost
                    distance_cost = self.pixel_distance(current, neighbor)
                    
                    # Diagonal bonus (45° moves are √2 ≈ 1.414, but faster in practice)
                    if self.is_diagonal_move(current, neighbor):
                        diagonal_bonus = 0.2  # Slight preference for diagonal
                        distance_cost -= diagonal_bonus
                    
                    # Turn angle penalty
                    turn_penalty = 0
                    if prev is not None:
                        turn_angle = self.calculate_turn_angle(prev, current, neighbor)
                        
                        # Penalty based on turn sharpness
                        if turn_angle < 10:  # Almost straight
                            turn_penalty = 0
                        elif turn_angle < 50:  # Slight turn (including 45°)
                            turn_penalty = 0.1
                        elif turn_angle < 95:  # Right angle (90°)
                            turn_penalty = 0.5
                        else:  # Sharp turn (>90°)
                            turn_penalty = 1.0
                    
                    tentative_g = g_score[state] + distance_cost + turn_penalty
                    
                    if neighbor_state not in g_score or tentative_g < g_score[neighbor_state]:
                        g_score[neighbor_state] = tentative_g
                        h = min(self.pixel_distance(neighbor, g) for g in goal_pixels)
                        f_score = tentative_g + h
                        
                        counter += 1
                        new_path = path + [neighbor]
                        heapq.heappush(open_set, (f_score, counter, neighbor, current, new_path))
        
        return None
    
    def exploration_phase(self):
        """Exploration phase."""
        print("\n" + "="*60)
        print("EXPLORATION PHASE")
        print("="*60)
        
        print("Phase 1: Finding path to goal...")
        path_to_goal, visited_forward = self.a_star_exploration(
            self.start_pixel, self.goal_region
        )
        
        if path_to_goal is None:
            print("ERROR: No path found!")
            return None, None, None
        
        goal_reached = path_to_goal[-1]
        print(f"  ✓ Path to goal: {len(path_to_goal)} pixels")
        
        print("\nPhase 2: Finding return path...")
        path_to_start, visited_backward = self.a_star_exploration(
            goal_reached, [self.start_pixel]
        )
        
        if path_to_start is None:
            print("ERROR: No return path!")
            return path_to_goal, None, visited_forward
        
        print(f"  ✓ Return path: {len(path_to_start)} pixels")
        
        all_explored = visited_forward | visited_backward
        print(f"\n✓ Total explored: {len(all_explored)} pixels")
        
        return path_to_goal, path_to_start, all_explored
    
    def speed_run_phase(self):
        """Speed run phase with angle optimization."""
        print("\n" + "="*60)
        print("SPEED RUN PHASE (Angle-Optimized)")
        print("="*60)
        print("Finding optimal path with smooth turns...")
        
        optimal_path = self.a_star_optimal_with_angles(
            self.start_pixel, self.goal_region
        )
        
        if optimal_path is None:
            print("ERROR: No optimal path!")
            return None, None
        
        print(f"  ✓ Optimal path: {len(optimal_path)} pixels")
        
        # Analyze turn types
        turn_analysis = self.analyze_turns(optimal_path)
        print(f"  ✓ Turn analysis:")
        print(f"    - Straight sections: {turn_analysis['straight']}")
        print(f"    - 45° turns: {turn_analysis['diagonal']}")
        print(f"    - 90° turns: {turn_analysis['right_angle']}")
        print(f"    - Sharp turns (>90°): {turn_analysis['sharp']}")
        
        speeds = self.calculate_speeds_with_angles(optimal_path)
        return optimal_path, speeds
    
    def analyze_turns(self, path):
        """Analyze types of turns in the path."""
        if len(path) < 3:
            return {'straight': 0, 'diagonal': 0, 'right_angle': 0, 'sharp': 0}
        
        straight = 0
        diagonal = 0
        right_angle = 0
        sharp = 0
        
        for i in range(1, len(path) - 1):
            angle = self.calculate_turn_angle(path[i-1], path[i], path[i+1])
            
            if angle < 10:
                straight += 1
            elif angle < 50:
                diagonal += 1
            elif angle < 95:
                right_angle += 1
            else:
                sharp += 1
        
        return {
            'straight': straight,
            'diagonal': diagonal,
            'right_angle': right_angle,
            'sharp': sharp
        }
    
    def calculate_speeds_with_angles(self, path):
        """
        Calculate speed profile considering turn angles.
        45° turns can be taken faster than 90° turns.
        """
        if len(path) < 3:
            return [1.0] * len(path)
        
        speeds = []
        max_speed = 3.0
        acceleration = 2.5  # Increased acceleration
        current_speed = 0
        
        for i in range(len(path) - 1):
            # Calculate turn angle
            if i < len(path) - 2:
                angle = self.calculate_turn_angle(path[i-1] if i > 0 else None, 
                                                  path[i], path[i+1])
            else:
                angle = 0
            
            # Check if diagonal move
            is_diagonal = self.is_diagonal_move(path[i], path[i+1]) if i < len(path) - 1 else False
            
            # Distance from walls
            wall_distance = self.distance_transform[path[i]]
            
            # Target speed based on turn angle
            if angle < 10:
                # Straight - maximum speed
                target_speed = max_speed
            elif angle < 50:
                # Slight turn or 45° diagonal - can maintain high speed
                target_speed = max_speed * 0.85
            elif angle < 95:
                # 90° turn - moderate speed reduction
                target_speed = max_speed * 0.5
            else:
                # Sharp turn - significant reduction
                target_speed = max_speed * 0.3
            
            # Bonus for diagonal moves (smoother)
            if is_diagonal and angle < 50:
                target_speed = min(target_speed * 1.1, max_speed)
            
            # Adjust for wall distance (can go faster in wide corridors)
            if wall_distance > 5:
                target_speed = min(target_speed * 1.1, max_speed)
            
            # Apply acceleration limits
            if target_speed > current_speed:
                current_speed = min(current_speed + acceleration * 0.12, target_speed)
            else:
                current_speed = max(current_speed - acceleration * 0.18, target_speed)
            
            speeds.append(current_speed)
        
        return speeds
    
    def create_animation(self, path_to_goal, path_return, speed_path, speeds):
        """Create live animation."""
        print("\nShowing live animation...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        base_maze = np.zeros((*self.skeleton.shape, 3))
        base_maze[self.binary_maze == 1] = [0.95, 0.95, 0.95]
        base_maze[self.binary_maze == 0] = [0, 0, 0]
        
        for gy, gx in self.goal_region[:5]:
            base_maze[max(0, gy-2):gy+3, max(0, gx-2):gx+3] = [1, 1, 0.7]
        
        im1 = axes[0].imshow(base_maze.copy())
        im2 = axes[1].imshow(base_maze.copy())
        im3 = axes[2].imshow(base_maze.copy())
        
        axes[0].set_title('Phase 1: Exploration to Goal', fontsize=14, fontweight='bold')
        axes[1].set_title('Phase 2: Return to Start', fontsize=14, fontweight='bold')
        axes[2].set_title('Phase 3: Optimized Speed Run', fontsize=14, fontweight='bold')
        
        for ax in axes:
            ax.axis('off')
        
        skip = max(1, len(path_to_goal) // 200)
        
        phase1_frames = len(path_to_goal) // skip
        phase2_frames = len(path_return) // skip if path_return else 0
        phase3_frames = len(speed_path) // skip if speed_path else 0
        total_frames = phase1_frames + phase2_frames + phase3_frames
        
        def animate(frame):
            if frame < phase1_frames:
                display1 = base_maze.copy()
                current_idx = min(frame * skip, len(path_to_goal) - 1)
                
                for i in range(0, current_idx, max(1, skip // 2)):
                    py, px = path_to_goal[i]
                    display1[py, px] = [0.6, 0.8, 1.0]
                
                for i in range(max(0, current_idx - 10), current_idx + 1):
                    py, px = path_to_goal[i]
                    display1[py, px] = [0, 0.4, 1]
                
                if current_idx < len(path_to_goal):
                    ry, rx = path_to_goal[current_idx]
                    display1[max(0, ry-3):ry+4, max(0, rx-3):rx+4] = [0, 1, 0]
                
                sy, sx = self.start_pixel
                display1[max(0, sy-2):sy+3, max(0, sx-2):sx+3] = [0, 1, 0.5]
                
                im1.set_array(display1)
                axes[0].set_title(f'Phase 1: Exploration [{current_idx}/{len(path_to_goal)}]', 
                                 fontsize=14, fontweight='bold')
                
                return [im1]
            
            elif frame < phase1_frames + phase2_frames:
                display1 = base_maze.copy()
                for py, px in path_to_goal[::skip]:
                    display1[py, px] = [0.4, 0.6, 0.9]
                im1.set_array(display1)
                
                display2 = base_maze.copy()
                phase2_idx = frame - phase1_frames
                current_idx = min(phase2_idx * skip, len(path_return) - 1)
                
                for i in range(0, current_idx, max(1, skip // 2)):
                    py, px = path_return[i]
                    display2[py, px] = [1, 0.7, 0.3]
                
                for i in range(max(0, current_idx - 10), current_idx + 1):
                    py, px = path_return[i]
                    display2[py, px] = [1, 0.4, 0]
                
                if current_idx < len(path_return):
                    ry, rx = path_return[current_idx]
                    display2[max(0, ry-3):ry+4, max(0, rx-3):rx+4] = [1, 0.5, 0]
                
                im2.set_array(display2)
                axes[1].set_title(f'Phase 2: Return [{current_idx}/{len(path_return)}]', 
                                 fontsize=14, fontweight='bold')
                
                return [im1, im2]
            
            else:
                display1 = base_maze.copy()
                display2 = base_maze.copy()
                
                for py, px in path_to_goal[::skip]:
                    display1[py, px] = [0.5, 0.6, 0.8]
                for py, px in path_return[::skip]:
                    display2[py, px] = [0.8, 0.6, 0.4]
                
                im1.set_array(display1)
                im2.set_array(display2)
                
                display3 = base_maze.copy()
                phase3_idx = frame - phase1_frames - phase2_frames
                current_idx = min(phase3_idx * skip, len(speed_path) - 1)
                
                max_speed_val = max(speeds) if speeds else 1
                for i in range(0, current_idx, max(1, skip // 2)):
                    py, px = speed_path[i]
                    if i < len(speeds):
                        speed_ratio = speeds[i] / max_speed_val
                        display3[py, px] = [1 - speed_ratio, speed_ratio, 0]
                
                if current_idx < len(speed_path):
                    ry, rx = speed_path[current_idx]
                    display3[max(0, ry-3):ry+4, max(0, rx-3):rx+4] = [0, 1, 1]
                
                im3.set_array(display3)
                axes[2].set_title(f'Phase 3: Optimized Run [{current_idx}/{len(speed_path)}] ⚡', 
                                 fontsize=14, fontweight='bold', color='green')
                
                return [im1, im2, im3]
        
        anim = FuncAnimation(fig, animate, frames=total_frames, 
                           interval=30, blit=True, repeat=True)
        
        plt.tight_layout()
        print("Displaying animation... (close window to continue)")
        plt.show()
        
        return anim
    
    def visualize_results(self, path_to_goal, path_return, speed_path, speeds, explored):
        """Save final visualization."""
        print("\nCreating final visualization...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Maze', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        skeleton_display = np.zeros((*self.skeleton.shape, 3))
        skeleton_display[self.binary_maze == 1] = [1, 1, 1]
        skeleton_display[self.skeleton == 1] = [0, 1, 1]
        ax2.imshow(skeleton_display)
        ax2.set_title('Skeleton', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        distance_display = np.zeros((*self.distance_transform.shape, 3))
        distance_display[self.binary_maze == 1] = [0.3, 0.3, 0.3]
        max_dist = np.max(self.distance_transform)
        for y in range(self.distance_transform.shape[0]):
            for x in range(self.distance_transform.shape[1]):
                if self.binary_maze[y, x] == 1:
                    dist_ratio = self.distance_transform[y, x] / max_dist
                    distance_display[y, x] = [dist_ratio, 0, 1 - dist_ratio]
        ax3.imshow(distance_display)
        ax3.set_title('Distance Transform', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 0])
        exploration_display = np.zeros((*self.skeleton.shape, 3))
        exploration_display[self.binary_maze == 1] = [0.9, 0.9, 0.9]
        exploration_display[self.binary_maze == 0] = [0, 0, 0]
        
        if explored:
            for pixel in explored:
                exploration_display[pixel] = [0.7, 0.8, 1.0]
        
        if path_to_goal:
            for pixel in path_to_goal:
                exploration_display[pixel] = [0, 0, 1]
        
        if self.start_pixel:
            y, x = self.start_pixel
            exploration_display[max(0, y-2):y+3, max(0, x-2):x+3] = [0, 1, 0]
        if self.goal_pixel:
            y, x = self.goal_pixel
            exploration_display[max(0, y-3):y+4, max(0, x-3):x+4] = [1, 1, 0]
        
        ax4.imshow(exploration_display)
        ax4.set_title(f'Exploration ({len(path_to_goal)} px)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        return_display = exploration_display.copy()
        if path_return:
            for pixel in path_return:
                return_display[pixel] = [1, 0.5, 0]
        ax5.imshow(return_display)
        ax5.set_title(f'Return ({len(path_return)} px)', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        speed_display = np.zeros((*self.skeleton.shape, 3))
        speed_display[self.binary_maze == 1] = [0.9, 0.9, 0.9]
        speed_display[self.binary_maze == 0] = [0, 0, 0]
        
        if speed_path and speeds:
            max_speed_val = max(speeds) if speeds else 1
            for i, pixel in enumerate(speed_path[:-1]):
                if i < len(speeds):
                    speed_ratio = speeds[i] / max_speed_val
                    speed_display[pixel] = [1 - speed_ratio, speed_ratio, 0]
        
        if self.start_pixel:
            y, x = self.start_pixel
            speed_display[max(0, y-2):y+3, max(0, x-2):x+3] = [0, 1, 1]
        if self.goal_pixel:
            y, x = self.goal_pixel
            speed_display[max(0, y-3):y+4, max(0, x-3):x+4] = [1, 0, 1]
        
        ax6.imshow(speed_display)
        ax6.set_title(f'Optimized Run ({len(speed_path)} px)', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        improvement = len(path_to_goal) - len(speed_path)
        efficiency = len(speed_path)/len(path_to_goal)*100
        turn_analysis = self.analyze_turns(speed_path)
        
        stats_text = f"""
OPTIMIZED MAZE SOLVER RESULTS
{'='*80}

EXPLORATION                          SPEED RUN (Angle-Optimized)
Path to Goal: {len(path_to_goal):,} pixels              Optimal Path: {len(speed_path):,} pixels
Return Path: {len(path_return):,} pixels               Improvement: {improvement} pixels shorter
Total Explored: {len(explored):,} pixels             Efficiency: {efficiency:.1f}% of exploration

TURN OPTIMIZATION ANALYSIS
{'='*80}
Straight Sections: {turn_analysis['straight']} (100% speed)
45° Diagonal Turns: {turn_analysis['diagonal']} (85% speed - SMOOTH!)
90° Right Angles: {turn_analysis['right_angle']} (50% speed)
Sharp Turns (>90°): {turn_analysis['sharp']} (30% speed)

✓ Diagonal movements (45°) preferred for smoother, faster navigation
✓ Turn angle penalties applied: straight=0, 45°=0.1, 90°=0.5, >90°=1.0
✓ Speed optimization considers turn sharpness and wall proximity
        """
        
        ax7.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax7.transAxes)
        
        plt.savefig('optimized_maze_solution.png', dpi=150, bbox_inches='tight')
        print("✓ Solution saved as 'optimized_maze_solution.png'")
        plt.show()
    
    def solve(self, show_animation=True):
        """Complete solving process."""
        print("\n" + "="*60)
        print("OPTIMIZED MICROMOUSE MAZE SOLVER")
        print("With Angle Optimization (45°, 90° turns)")
        print("="*60)
        
        start_time = time.time()
        
        self.load_and_process_image()
        self.extract_path_centerline()
        self.find_start_position()
        self.find_goal_region()
        
        path_to_goal, path_return, explored = self.exploration_phase()
        
        if path_to_goal is None:
            print("\n❌ Exploration failed!")
            return
        
        speed_path, speeds = self.speed_run_phase()
        
        if speed_path is None:
            print("\n❌ Speed run failed!")
            return
        
        total_time = time.time() - start_time
        
        if show_animation:
            self.create_animation(path_to_goal, path_return, speed_path, speeds)
        
        self.visualize_results(path_to_goal, path_return, speed_path, speeds, explored)
        
        print("\n" + "="*60)
        print("✓ COMPLETE")
        print("="*60)
        print(f"Processing time: {total_time:.2f} seconds")
        
        return path_to_goal, path_return, speed_path


def main():
    """Main function."""
    print("OPTIMIZED MICROMOUSE MAZE SOLVER")
    print("="*60)
    print("Features:")
    print("  ✓ Angle-optimized pathfinding")
    print("  ✓ Supports 45° diagonal turns (smoother, faster)")
    print("  ✓ Turn penalties: 0° < 45° < 90° < 180°")
    print("  ✓ Live animation with speed visualization")
    print()
    
    image_path = input("Enter maze image path: ").strip()
    if not image_path:
        print("No image path provided!")
        return
    
    show_anim = input("Show live animation? (y/n, default=y): ").strip().lower()
    show_animation = show_anim != 'n'
    
    try:
        solver = OptimizedMazeSolver(image_path)
        solver.solve(show_animation=show_animation)
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find '{image_path}'")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
