from utils.test_functions import FUNCTIONS
import numpy as np
from models.Enhanced_Algo import ValoEnhanced
from utils.visualize import visualize_simulation


chooser = 2 
FUNC = FUNCTIONS[chooser] 
FUNC_NAME = FUNC.name     
LB = -5.12
UB = 5.12                   

N_AGENTS = 48
ITER = 100
SEED = 12345

opt = ValoEnhanced(
    func=FUNC.func,          # Important: pass the actual function, not the TestFunction object
    lb=LB,
    ub=UB,
    n_agents=N_AGENTS,
    dim=2,
    seed=SEED
)

positions_history, best_history, best_score = opt.run(iterations=ITER)
print(f"Final best score: {best_score:.6e} at {opt.best}")

view_size = 5.12
LB_arr = np.array([-1*view_size, -1*view_size])
UB_arr = np.array([view_size,view_size])

visualize_simulation(
    positions_history,
    best_history,
    FUNC.func,
    FUNC_NAME,
    LB_arr,
    UB_arr,
    opt.roles,
    N_AGENTS,
    ITER,
    output_gif=True
)
