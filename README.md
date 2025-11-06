# Micromouse Maze Solver - README

## Overview

This Python program simulates a micromouse competition solver using the A* pathfinding algorithm. It processes maze images (white maze with black walls) and performs two solving phases:

1. **Exploration Run (First Try)**: Discovers the maze structure while finding a path to the goal
2. **Speed Run (Second Try)**: Uses complete maze knowledge to find the optimal path quickly

## Features

- **Image Processing**: Converts maze images to grid representation using OpenCV
- **A* Algorithm**: Implements A* pathfinding with Manhattan distance heuristic
- **Two-Phase Solving**: 
  - First run simulates real exploration (discovering walls)
  - Second run uses complete knowledge for optimal speed
- **Visualization**: 
  - Side-by-side comparison of both runs
  - Animated path progression
  - Color-coded paths and explored areas
- **Performance Metrics**: Compares path lengths and execution times

## Requirements

```bash
pip install opencv-python numpy matplotlib
```

Or install all at once:
```bash
pip install opencv-python numpy matplotlib
```

## Usage

### Basic Usage

```python
from micromouse_solver import MazeSolver

# Initialize solver with your maze image
solver = MazeSolver("maze.png")

# Process the image (adjust cell_size based on your maze resolution)
solver.load_and_process_image(cell_size=20)

# Set start and goal (auto-detected by default)
solver.set_start_and_goal()

# Solve the maze with visualization
exploration_path, speed_path = solver.solve(visualize=True, animate=True)
```

### Interactive Mode

Run the program directly:
```bash
python micromouse_solver.py
```

You'll be prompted to:
1. Enter the maze image path
2. Specify cell size (pixels per grid cell)
3. Confirm or override start/goal positions
4. Choose whether to show animation

## Maze Image Requirements

- **Format**: PNG, JPG, or any OpenCV-supported format
- **Colors**: 
  - White background (paths) - RGB(255, 255, 255)
  - Black walls - RGB(0, 0, 0)
- **Structure**: Clear boundaries between walls and paths
- **Resolution**: Higher resolution = more accurate grid conversion

### Example Maze Structure
```
████████████████████
█     █       █    █
█ ███ █ █████ █ ██ █
█   █   █   █ █  █ █
███ █████ █ █ ██ █ █
█   █     █ █    █ █
█ ███████ █ ████ █ █
█               █   █
████████████████████
```

## How It Works

### 1. Image Processing
- Loads maze image using OpenCV
- Converts to grayscale and applies binary threshold
- Creates grid representation by sampling cell centers
- Each cell is classified as wall (1) or path (0)

### 2. Exploration Run (First Try)
- Simulates a real micromouse discovering the maze
- Uses A* algorithm with gradual wall discovery
- Tracks explored cells and discovered walls
- May take longer path due to incomplete knowledge
- **Purpose**: Learn the maze structure

### 3. Speed Run (Second Try)
- Uses complete maze knowledge from exploration
- Applies A* algorithm with full wall information
- Finds truly optimal path
- Executes much faster
- **Purpose**: Achieve best time to goal

### 4. A* Algorithm Details

**Components:**
- `g(n)`: Actual cost from start to current node
- `h(n)`: Heuristic (Manhattan distance to goal)
- `f(n) = g(n) + h(n)`: Total estimated cost

**Manhattan Distance Heuristic:**
```
h(pos1, pos2) = |row1 - row2| + |col1 - col2|
```

This heuristic is:
- **Admissible**: Never overestimates actual cost
- **Consistent**: Satisfies triangle inequality
- **Optimal for grid-based movement**: Perfect for micromouse mazes

## Visualization

### Static Visualization (3 panels)
1. **Original Image**: Your input maze
2. **Exploration Run**: Blue path with light blue explored area
3. **Speed Run**: Green optimal path

### Color Legend
- **Green** (start in first run): Starting position
- **Red** (goal in first run): Goal position
- **Blue**: Exploration path
- **Light Blue**: Explored cells during first run
- **Bright Green**: Optimal speed run path
- **Cyan** (start in second run): Starting position
- **Magenta** (goal in second run): Goal position
- **Black**: Walls

### Animated Visualization
- Shows both runs side-by-side
- Animates path progression frame by frame
- Helps understand the exploration vs speed run difference

## Example Output

```
==================================================
MICROMOUSE MAZE SOLVER
==================================================

=== EXPLORATION RUN (First Try) ===
Discovering maze structure...
Goal reached! Explored 245 cells
Path length: 156 cells
Exploration time: 0.023 seconds

=== SPEED RUN (Second Try) ===
Using discovered maze knowledge for optimal path...
Optimal path found! Length: 89 cells
Cells visited: 112 (much faster!)
Speed run time: 0.008 seconds

==================================================
COMPARISON
==================================================
Exploration run: 156 steps
Speed run: 89 steps
Improvement: 67 steps faster
Time improvement: 2.9x faster

Visualization saved as 'maze_solution.png'
```

## Customization

### Adjust Cell Size
```python
# For high-resolution images
solver.load_and_process_image(cell_size=30)

# For low-resolution images
solver.load_and_process_image(cell_size=10)
```

### Manual Start/Goal Setting
```python
# Set custom start and goal positions
solver.set_start_and_goal(
    start=(0, 0),      # Top-left corner
    goal=(15, 15)      # Bottom-right corner
)
```

### Disable Visualization
```python
# Run without showing visualization (faster)
exploration_path, speed_path = solver.solve(visualize=False, animate=False)
```

## Algorithm Complexity

### Time Complexity
- **A* Algorithm**: O(E log V) where:
  - V = number of vertices (grid cells)
  - E = number of edges (cell connections)
  - For grid: E ≈ 4V (each cell has ~4 neighbors)
  - Overall: O(V log V)

### Space Complexity
- **Grid Storage**: O(rows × cols)
- **Open Set**: O(V) in worst case
- **Path Storage**: O(V) for longest path
- **Overall**: O(V)

## Micromouse Competition Notes

This implementation follows micromouse competition principles:

1. **Two-phase strategy**: Exploration then speed run
2. **Wall discovery**: Simulates sensor-based wall detection
3. **Optimal pathfinding**: Uses efficient A* algorithm
4. **Performance metrics**: Tracks both time and path length

### Real Competition Differences
- Real robots use sensors (IR, ultrasonic) instead of image processing
- Physical constraints: acceleration, turning radius, momentum
- Diagonal movement possible in some competitions
- Multiple speed runs allowed after exploration
- Time limits on exploration phase

## Troubleshooting

### Image Not Loading
- Check file path is correct
- Ensure image format is supported (PNG, JPG recommended)
- Verify file permissions

### Poor Maze Detection
- Increase image resolution
- Adjust `cell_size` parameter
- Ensure clear contrast between walls and paths
- Remove noise from image

### No Path Found
- Verify start and goal are not inside walls
- Check if maze has a valid solution
- Ensure walls form proper boundaries

### Slow Performance
- Reduce maze resolution
- Increase cell_size for coarser grid
- Disable animation for faster execution

## Future Enhancements

Possible improvements:
- Support for diagonal movement
- Flood fill algorithm implementation
- Real-time sensor simulation
- 3D visualization
- Multiple goal positions (center square)
- Turn optimization (minimize turns for faster time)
- Export path as robot commands

## License

Free to use for educational and competition purposes.

## References

- Micromouse Competition: https://en.wikipedia.org/wiki/Micromouse
- A* Algorithm: https://en.wikipedia.org/wiki/A*_search_algorithm
- Flood Fill for Micromouse: Common alternative algorithm
- OpenCV Documentation: https://docs.opencv.org/

## Contact

For issues or improvements, feel free to modify and enhance this code for your competition needs!

Good luck with your micromouse competition! 🐭🏆
