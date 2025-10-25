# Valo-Algorithm
Optimization Algorithm Inspired from Valorant
a# ValoEnhanced Optimization Algorithm

A bio-inspired metaheuristic optimization algorithm implementing role-based agent collaboration with adaptive search strategies.

## Overview

ValoEnhanced is a swarm intelligence algorithm that uses multiple agent roles to efficiently explore and exploit the search space. The algorithm is inspired by team-based coordination strategies and implements four distinct agent roles that work collaboratively to find optimal solutions.

## Algorithm Architecture

### Agent Roles

The algorithm employs four specialized agent types, each with unique behaviors:

1. **Initiators (30%)** - Exploration agents that discover new regions of the search space
   - Use adaptive Lévy flights for long-range exploration
   - Generate hotspots (promising regions) for other agents
   - Exploration intensity decreases adaptively as optimization progresses

2. **Duelists (35%)** - Exploitation agents that intensify search around hotspots
   - Target the best hotspots identified by Initiators
   - Use directed movement with Lévy flight perturbations
   - Balance between targeted exploitation and local exploration

3. **Controllers (20%)** - Coordination agents that maintain population cohesion
   - Calculate centroids of active agents (Initiators and Duelists)
   - Pull the swarm toward high-quality regions
   - Prevent premature convergence through strategic positioning

4. **Sentinels (15%)** - Convergence agents that refine solutions
   - Focus on the global best solution
   - Incorporate Controller positions for balanced refinement
   - Perform occasional random perturbations to escape local optima

### Key Features

- **Adaptive Step Sizing**: Movement magnitudes adjust based on population diversity and optimization progress
- **Lévy Flight Integration**: Power-law step distributions enable efficient exploration across multiple scales
- **Elite Archive**: Maintains top solutions to preserve high-quality candidates
- **Progressive Focus**: Automatically transitions from exploration to exploitation over iterations

### Operating Modes

The algorithm supports three operational modes:

- **Conservative**: Cautious exploration with strong convergence (`mode="conservative"`)
- **Balanced**: Equal emphasis on exploration and exploitation (`mode="balanced"`)
- **Aggressive**: Intensive exploration with delayed convergence (`mode="aggressive"`)

## Usage

### Basic Optimization

```python
from models.Enhanced_Algo import ValoEnhanced
import numpy as np

# Define your objective function
def sphere_function(x):
    return np.sum(x**2)

# Initialize optimizer
optimizer = ValoEnhanced(
    func=sphere_function,
    lb=-5.12,          # Lower bound
    ub=5.12,           # Upper bound
    dim=30,            # Problem dimension
    n_agents=48,       # Population size
    seed=12345         # Random seed
)

# Run optimization
positions_history, best_history, best_score = optimizer.run(iterations=100)

# Get final solution
solution = optimizer.get_best_solution()
print(f"Best score: {solution['score']:.6e}")
print(f"Best position: {solution['position']}")
```

### 2D Visualization

Use `testing_plot.py` to visualize algorithm behavior and generate animated GIFs:

```python
python testing_plot.py
```

This will:
- Run the optimization on a 2D test function
- Generate an animated GIF showing agent movements
- Display convergence behavior and role dynamics

### High-Dimensional Testing

Use `testing_function.py` for benchmarking on higher-dimensional problems:

```python
python testing_function.py
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 40 | Number of agents in the swarm |
| `dim` | - | Problem dimensionality |
| `mode` | "balanced" | Operating mode (conservative/balanced/aggressive) |
| `sigma1` | 0.8 | Initiator step size coefficient |
| `sigma2` | 0.12 | Duelist attraction coefficient |
| `beta` | 1.5 | Lévy flight exponent |
| `eta` | 0.25 | Controller convergence rate |
| `lmbda` | 2 | Sentinel convergence coefficient |
| `top_k` | auto | Number of top hotspots to track |
| `seed` | None | Random seed for reproducibility |

## Algorithm Workflow

1. **Initialization**: Agents are randomly distributed across the search space
2. **Role Assignment**: Agents are assigned roles based on specified fractions
3. **Iteration Loop**:
   - Initiators explore and create hotspots
   - Duelists exploit promising hotspots
   - Controllers maintain swarm cohesion
   - Sentinels refine the best solution
4. **Evaluation**: All positions are evaluated and the global best is updated
5. **Archive Update**: Elite solutions are preserved
6. **Convergence**: Process repeats until maximum iterations reached

## Performance Examples

Below are convergence plots demonstrating the algorithm's performance on standard benchmark functions:

| Function | Visualization | Convergence Plot |
|----------|---------------|------------------|
| **Rastrigin** (highly multimodal) | <img width="561" height="422" alt="image" src="https://github.com/user-attachments/assets/e4e500ad-0c01-44c4-8b46-f2fd7a97bdb4" /> |![listen_simulation_rastrigin](https://github.com/user-attachments/assets/486faab7-c8cb-4457-a870-52397582a90b)|
| **Griewank** (unimodal convex) |<img width="636" height="410" alt="image" src="https://github.com/user-attachments/assets/fcc5cd99-822d-408a-ae92-3d235e399df4" />| ![listen_simulation_griewank](https://github.com/user-attachments/assets/cd3d1feb-17c1-4d4b-b4a2-3df7886869a0)|
| **RosenBrock** (multimodal with global structure) | <img width="370" height="280" alt="image" src="https://github.com/user-attachments/assets/7c92a314-acab-445b-a72f-694b73721b64" />| ![listen_simulation_rosenbrock](https://github.com/user-attachments/assets/9f4099ff-cfe0-4643-abf7-851518d142fc)|

*Drag and drop your images into the empty parentheses above*

## Project Structure

```
.
├── models/
│   └── Enhanced_Algo.py      # ValoEnhanced algorithm implementation
|   └── Base_Algo.py          # Base Algorithm
├── utils/
│   ├── test_functions.py     # Benchmark optimization functions
│   └── visualize.py          # Visualization utilities
├── testing_plot.py           # 2D visualization script
├── testing_function.py       # High-dimensional testing script
└── README.md                 # This file
```

## Customization

### Custom Role Distribution

```python
optimizer = ValoEnhanced(
    func=your_function,
    lb=lower_bound,
    ub=upper_bound,
    dim=dimensions,
    roles_frac={
        "initiator": 0.25,
        "duelist": 0.40,
        "controller": 0.20,
        "sentinel": 0.15
    }
)
```

### Aggressive Exploration

```python
optimizer = ValoEnhanced(
    func=your_function,
    lb=lower_bound,
    ub=upper_bound,
    dim=dimensions,
    mode="aggressive"  # More exploration, less convergence
)
```
Made with ❤️ from NiceGuy
