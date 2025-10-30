from utils.test_functions import FUNCTIONS
import numpy as np
from models.Enhanced_Algo import ValoEnhanced


chooser = 6
FUNC = FUNCTIONS[chooser] 
FUNC_NAME = FUNC.name     
LB = -5.12
UB = 5.12                   

N_AGENTS = 60
ITER = 100
SEED = 12345
dim = 2

opt = ValoEnhanced(
    func=FUNC.func,          # Important: pass the actual function, not the TestFunction object
    lb=LB,
    ub=UB,
    n_agents=N_AGENTS,
    dim=dim,
    seed=SEED
)

positions_history, best_history, best_score = opt.run(iterations=ITER)
print(f"Final best score: {FUNC_NAME} , {best_score:.6e} at {opt.best}")
