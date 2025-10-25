import numpy as np
import math
from tqdm import tqdm

def rastrigin_2d(x):
    A = 10
    return A * 2 + (x[0]**2 - A * math.cos(2*math.pi*x[0])) + (x[1]**2 - A * math.cos(2*math.pi*x[1]))

def sphere_2d(x):
    return x[0]**2 + x[1]**2

class LISTEN2D:
    def __init__(self, func, lb, ub, n_agents=40, roles_frac=None, seed=None,
                 sigma1=0.8, sigma2=0.12, beta=0.6, eta=0.25, lmbda=0.12, top_k=3):
        self.func = func
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.dim = 2
        self.n_agents = n_agents
        self.rng = np.random.RandomState(seed)
        
        if roles_frac is None:
            roles_frac = {"initiator": 0.35, "duelist": 0.30, "controller": 0.20, "sentinel": 0.15}
        self.roles_frac = roles_frac
        
        n_init = int(np.round(self.n_agents * roles_frac["initiator"]))
        n_duel = int(np.round(self.n_agents * roles_frac["duelist"]))
        n_ctrl = int(np.round(self.n_agents * roles_frac["controller"]))
        n_sent = self.n_agents - (n_init + n_duel + n_ctrl)
        self.roles = (["initiator"]*n_init + ["duelist"]*n_duel + 
                      ["controller"]*n_ctrl + ["sentinel"]*n_sent)
        # fix length
        while len(self.roles) > self.n_agents:
            self.roles.pop()
        while len(self.roles) < self.n_agents:
            self.roles.append("initiator")
        
        # params
        self.p = {"sigma1": sigma1, "sigma2": sigma2, "beta": beta, "eta": eta, "lmbda": lmbda, "top_k": top_k}
        self.X = self.rng.uniform(self.lb, self.ub, (self.n_agents, 2))
        self.fitness = np.full(self.n_agents, np.inf)
        self.best = None
        self.best_score = np.inf
        self.archive = []
        # track history
        self.history_positions = [self.X.copy()]
        self.history_best = [None]
    
    def levy(self, dim=2):
        beta = 1.5
        sigma_u = (math.gamma(1+beta) * math.sin(math.pi*beta/2) / 
                   (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
        u = self.rng.normal(0, sigma_u, size=dim)
        v = self.rng.normal(0, 1, size=dim)
        step = u / (np.abs(v)**(1/beta))
        return step
    
    def clip(self, x):
        return np.minimum(np.maximum(x, self.lb), self.ub)
    
    def evaluate(self):
        for i in range(self.n_agents):
            self.fitness[i] = self.func(self.X[i])
            if self.fitness[i] < self.best_score:
                self.best_score = self.fitness[i]
                self.best = self.X[i].copy()
    
    def step(self):
        initiator_idx = [i for i, r in enumerate(self.roles) if r=="initiator"]
        duelists_idx = [i for i, r in enumerate(self.roles) if r=="duelist"]
        controllers_idx = [i for i, r in enumerate(self.roles) if r=="controller"]
        sentinels_idx = [i for i, r in enumerate(self.roles) if r=="sentinel"]

        # ----------------------- INITIATORS -----------------------
        hotspots = []
        pop_std = np.std(self.X, axis=0)
        step_scale = np.mean(pop_std)

        for i in initiator_idx:
            sigma_adapt = self.p["sigma1"] * step_scale * (1 - self.iter / self.max_iter)
            if self.rng.rand() < 0.05:
                step = 0.5 * self.levy(self.dim)
            else:
                step = sigma_adapt * self.rng.randn(self.dim)
            self.X[i] = self.clip(self.X[i] + step)
            hotspots.append(self.X[i].copy())

        # ----------------------- HOTSPOT SELECTION -----------------------
        if len(hotspots) > 0:
            hotspot_scores = np.array([self.func(h) for h in hotspots])
            k = min(self.p["top_k"], len(hotspots))
            top_idx = np.argsort(hotspot_scores)[:k]
            targets = np.array(hotspots)[top_idx]
        else:
            targets = self.X[self.rng.choice(self.n_agents, size=min(3,self.n_agents), replace=False)]

        # ----------------------- DUELISTS -----------------------
        for d in duelists_idx:
            target = targets[self.rng.randint(0, targets.shape[0])]
            step = self.p["sigma2"] * (target - self.X[d]) + self.p["beta"] * self.levy(self.dim)
            self.X[d] = self.clip(self.X[d] + step)

        # ----------------------- CONTROLLERS -----------------------
        for c in controllers_idx:
            active_idx = duelists_idx + initiator_idx
            if len(active_idx) > 0:
                k = min(self.p["top_k"], len(active_idx))
                chosen = self.rng.choice(active_idx, size=k, replace=False)
                centroid = np.mean(self.X[chosen], axis=0)

                # Weight duelists more (stronger influence)
                if len(duelists_idx) > 0:
                    duel_centroid = np.mean(self.X[duelists_idx], axis=0)
                    centroid = 0.7 * duel_centroid + 0.3 * centroid

                # Add small jitter to preserve diversity
                centroid += 0.01 * self.rng.randn(self.dim)
                step = self.p["eta"] * (centroid - self.X[c])
                self.X[c] = self.clip(self.X[c] + step)
            else:
                rand = self.X[self.rng.randint(0, self.n_agents)]
                step = self.p["eta"] * (rand - self.X[c])
                self.X[c] = self.clip(self.X[c] + step)

        # ----------------------- SENTINELS -----------------------
        for s in sentinels_idx:
            if self.best is not None:
                step = self.p["lmbda"] * (self.best - self.X[s])
                self.X[s] = self.clip(self.X[s] + step)
            if self.rng.rand() < 0.05:
                self.X[s] = self.clip(self.X[s] + 0.02 * self.rng.randn(self.dim))

        # ----------------------- RANDOM PERTURBATION -----------------------
        for i in range(self.n_agents):
            if self.rng.rand() < 0.02:
                self.X[i] = self.clip(self.X[i] + 0.01 * self.rng.randn(self.dim))

        # ----------------------- EVALUATION -----------------------
        self.evaluate()
        combined = list(zip(self.fitness, [x.copy() for x in self.X]))
        combined_sorted = sorted(combined, key=lambda z: z[0])
        elites = [pos for (score,pos) in combined_sorted[:min(5, len(combined_sorted))]]
        self.archive = elites

        self.history_positions.append(self.X.copy())
        self.history_best.append(self.best.copy() if self.best is not None else None)
        self.iter += 1

    
    def run(self, iterations=100):
        self.max_iter = iterations
        self.iter = 0
        self.evaluate()
        for _ in tqdm(range(iterations)):
            self.step()
        return self.history_positions, self.history_best, self.best_score


ziser = 100
LB = np.array([-1*ziser, -1*ziser])
UB = np.array([ziser,ziser])
N_AGENTS = 100
ITER = 100
SEED = 12345


FUNC = rastrigin_2d 
FUNC_NAME = "Rastrigin (2D)" if FUNC is rastrigin_2d else "Sphere (2D)"


opt = LISTEN2D(func=FUNC, lb=LB, ub=UB, n_agents=N_AGENTS, seed=SEED)

positions_history, best_history, best_score = opt.run(iterations=ITER)
print(f"Final best score: {best_score:.6e} at {opt.best}")