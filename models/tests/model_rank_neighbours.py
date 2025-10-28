import numpy as np
import math
from tqdm import tqdm


class ValoEnhancedKNN:
    def __init__(self, func, lb, ub, dim, n_agents=40, roles_frac=None, seed=None,
                 sigma1=0.8, sigma2=0.12, beta=1.5, eta=0.25, lmbda=2, top_k=None,
                 mode="balanced"):

        self.func = func
        self.lb, self.ub, self.dim = np.array(lb), np.array(ub), dim
        self.n_agents = n_agents
        self.rng = np.random.RandomState(seed)
        self.mode = mode.lower()

        if top_k is None:
            self.top_k = max(1, int(0.02 * n_agents * np.log2(dim + 1)))
        else:
            self.top_k = top_k

        if roles_frac is None:
            roles_frac = {"initiator": 0.30, "duelist": 0.35, "controller": 0.20, "sentinel": 0.15}
        self.roles_frac = roles_frac

        n_init = int(np.round(n_agents * roles_frac["initiator"]))
        n_duel = int(np.round(n_agents * roles_frac["duelist"]))
        n_ctrl = int(np.round(n_agents * roles_frac["controller"]))
        n_sent = n_agents - (n_init + n_duel + n_ctrl)
        self.roles = (["initiator"] * n_init + ["duelist"] * n_duel +
                      ["controller"] * n_ctrl + ["sentinel"] * n_sent)

        while len(self.roles) > n_agents:
            self.roles.pop()
        while len(self.roles) < n_agents:
            self.roles.append("initiator")

        if self.mode == "conservative":
            self.p = {"sigma1": 0.8, "sigma2": sigma2, "beta": beta, "eta": 0.25, "lmbda": 0.2}
        elif self.mode == "balanced":
            self.p = {"sigma1": 1.0, "sigma2": sigma2, "beta": beta, "eta": 0.4, "lmbda": 1.5}
        elif self.mode == "aggressive":
            self.p = {"sigma1": 1.2, "sigma2": 0.2, "beta": beta, "eta": 0.55, "lmbda": 0.1}
        else:
            raise ValueError("Mode must be one of: conservative | balanced | aggressive")

        self.p["top_k"] = self.top_k

        self.positions = self.rng.uniform(lb, ub, (n_agents, dim))
        self.fitness = np.full(n_agents, np.inf)
        self.best = None
        self.best_score = np.inf
        self.best_idx = None
        self.archive = []

        self.history_positions = [self.positions.copy()]
        self.history_best = [None]
    
    def levy_flight(self, beta, dim, rng=np.random):
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                    (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = rng.normal(0, sigma_u, size=dim)
        v = rng.normal(0, 1, size=dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def _evaluate(self):
        for i in range(self.n_agents):
            f = self.func(self.positions[i])
            self.fitness[i] = f
            if f < self.best_score:
                self.best_score = f
                self.best = self.positions[i].copy()
                self.best_idx = i

    def _step(self):
        initiator_idx = [i for i, r in enumerate(self.roles) if r == "initiator"]
        duelists_idx = [i for i, r in enumerate(self.roles) if r == "duelist"]
        controllers_idx = [i for i, r in enumerate(self.roles) if r == "controller"]
        sentinels_idx = [i for i, r in enumerate(self.roles) if r == "sentinel"]

        hotspots = []
        pop_std = np.std(self.positions, axis=0)
        step_scale = np.mean(pop_std)
        progress = self.iter / self.max_iter

        levy_prob = 0.2 - 0.1 * progress
        for i in initiator_idx:
            sigma_adapt = self.p["sigma1"] * step_scale * (0.3 + 0.7 * (1 - progress))
            if self.rng.rand() < levy_prob:
                step = 0.8 * self.levy_flight(self.p["beta"], self.dim, self.rng)
            else:
                step = sigma_adapt * self.rng.randn(self.dim)
            self.positions[i] = self._clip(self.positions[i] + step)
            hotspots.append(self.positions[i].copy())

        if len(hotspots) > 0:
            hotspot_scores = np.array([self.func(h) for h in hotspots])
            k = min(self.p["top_k"], len(hotspots))
            top_idx = np.argsort(hotspot_scores)[:k]
            targets = np.array(hotspots)[top_idx]
        else:
            targets = self.positions[self.rng.choice(self.n_agents, size=min(3, self.n_agents), replace=False)]

        for d in duelists_idx:
            target = targets[self.rng.randint(0, len(targets))]
            lf = self.levy_flight(self.p["beta"], self.dim, self.rng)
            step = (self.p["sigma2"] * (target - self.positions[d]) +
                    0.5 * self.p["beta"] * lf)
            self.positions[d] = self._clip(self.positions[d] + step)

        for c in controllers_idx:
            active_idx = duelists_idx + initiator_idx
            if len(active_idx) > 0:
                # Compute distances from controller c to all active agents
                distances = np.linalg.norm(self.positions[active_idx] - self.positions[c], axis=1)
                
                # Compute k dynamically same as before
                k_max = min(self.p["top_k"], len(active_idx))
                k = max(5, int(k_max * (1 - progress) + 0.5))
                
                # Find k nearest neighbors (indices relative to active_idx)
                nearest_indices = np.argsort(distances)[:k]
                nearest_agents = [active_idx[i] for i in nearest_indices]
                
                # Compute centroid of nearest neighbors
                centroid = np.mean(self.positions[nearest_agents], axis=0)

                step = self.p["eta"] * (centroid - self.positions[c]) + 0.1 * self.rng.randn(self.dim)
                self.positions[c] = self._clip(self.positions[c] + step)
            else:
                rand = self.positions[self.rng.randint(0, self.n_agents)]
                step = self.p["eta"] * (rand - self.positions[c]) + 0.1 * self.rng.randn(self.dim)
                self.positions[c] = self._clip(self.positions[c] + step)



        controllers_exist = len(controllers_idx) > 0
        controller_centroid = (np.mean(self.positions[controllers_idx], axis=0)
                            if controllers_exist else None)

        for s in sentinels_idx:
            if self.best is not None:
                step = self.p["lmbda"] * (self.best - self.positions[s]) + 0.2*(controller_centroid - self.positions[s])
                self.positions[s] = self._clip(self.positions[s] + step)
            if self.rng.rand() < 0.05:
                self.positions[s] = self._clip(self.positions[s] + 0.02 * self.rng.randn(self.dim))


        self._evaluate()
        combined = sorted(zip(self.fitness, [x.copy() for x in self.positions]), key=lambda z: z[0])
        elites = [pos for (score, pos) in combined[:min(5, len(combined))]]
        self.archive = elites

        self.history_positions.append(self.positions.copy())
        self.history_best.append(self.best.copy() if self.best is not None else None)
        self.iter += 1

    def run(self, iterations=100):
        self.max_iter = iterations
        self.iter = 0
        self._evaluate()

        desc = f"Enhanced with KNN ({self.mode.title()} Mode)"
        with tqdm(total=iterations, desc=desc, ncols=100) as pbar:
            for _ in range(iterations):
                self._step()
                pbar.set_postfix({"Best Score": f"{self.best_score:.6f}"})
                pbar.update(1)

        return self.history_positions, self.history_best, self.best_score
    
    def get_best_solution(self):
        return {
            "position": self.best.copy() if self.best is not None else None,
            "score": self.best_score,
            "index": self.best_idx
        }
    

def rastrigin(x):
    x = np.asarray(x)
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

dim = 100
lb = -100 * np.ones(dim)
ub = 100 * np.ones(dim)

opt = ValoEnhancedKNN(rastrigin, lb, ub, 100, n_agents=600, seed=42)

hist_pos, hist_best, best_score = opt.run(iterations=200)
print("\nBest score:", best_score)
print("Best position (first 6 dims):", opt.get_best_solution()["position"][:6])
