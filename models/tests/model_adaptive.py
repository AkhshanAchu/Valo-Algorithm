import numpy as np
import math
from tqdm import tqdm


class ValoEnhancedKNNAdaptiveOptimized:
    """
    Balanced+Safe adaptive Valo optimizer.
    Key features:
      - Distance-weighted KNN controller centroid
      - Velocity + momentum for stable moves
      - Robust softmax sampling with deterministic fallback (fixes ValueError)
      - Probabilistic promotion/demotion with elite protection
      - Adaptive conv_period and conv_strength based on progress/stagnation
      - Role rebalance to respect original distribution (unless allow_fractional_adjust=True)
      - Minor auto-boost nudges when stagnation detected
    """
    def __init__(
        self, func, lb, ub, dim,
        n_agents=40, roles_frac=None, seed=None,
        sigma1=0.9, sigma2=0.12, beta=1.5, eta=0.32, lmbda=2.0, top_k=None,
        mode="balanced",
        conv_period=10, top_promote_frac=0.08, bottom_demote_frac=0.10,
        allow_fractional_adjust=False,
        aggressiveness=0.5, conv_strength=0.8,
        n_elite=5, vel_momentum=0.65, min_k_neighbors=5,
        temp_start=1.0, temp_end=0.1,
        stagnation_threshold=20, auto_boost_factor=1.3
    ):
        self.func = func
        self.lb, self.ub, self.dim = np.array(lb), np.array(ub), dim
        self.n_agents = int(n_agents)
        self.rng = np.random.RandomState(seed)
        self.mode = mode.lower()

        if top_k is None:
            self.top_k = max(1, int(0.02 * self.n_agents * np.log2(dim + 1)))
        else:
            self.top_k = top_k

        if roles_frac is None:
            roles_frac = {"initiator": 0.30, "duelist": 0.35, "controller": 0.20, "sentinel": 0.15}
        self.roles_frac = roles_frac.copy()

        # initialize roles
        n_init = int(np.round(self.n_agents * roles_frac["initiator"]))
        n_duel = int(np.round(self.n_agents * roles_frac["duelist"]))
        n_ctrl = int(np.round(self.n_agents * roles_frac["controller"]))
        n_sent = self.n_agents - (n_init + n_duel + n_ctrl)
        self.roles = (["initiator"] * n_init + ["duelist"] * n_duel +
                      ["controller"] * n_ctrl + ["sentinel"] * n_sent)
        # pad/cut exactly
        while len(self.roles) > self.n_agents:
            self.roles.pop()
        while len(self.roles) < self.n_agents:
            self.roles.append("initiator")

        # mode presets (small base differences)
        if self.mode == "conservative":
            p_base = {"sigma1": 0.75, "sigma2": sigma2, "beta": beta, "eta": 0.26, "lmbda": 0.22}
        elif self.mode == "balanced":
            p_base = {"sigma1": sigma1, "sigma2": sigma2, "beta": beta, "eta": eta, "lmbda": lmbda}
        elif self.mode == "aggressive":
            p_base = {"sigma1": 1.2 * sigma1, "sigma2": 1.1 * sigma2, "beta": beta, "eta": 1.2 * eta, "lmbda": 0.9 * lmbda}
        else:
            raise ValueError("Mode must be conservative | balanced | aggressive")

        # tune by aggressiveness parameter
        self.aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))
        self.conv_strength = float(np.clip(conv_strength, 0.0, 1.0))
        self.n_elite = int(max(0, n_elite))
        self.vel_momentum = float(np.clip(vel_momentum, 0.0, 0.95))
        self.min_k_neighbors = max(1, int(min_k_neighbors))

        # temperature anneal
        self.temp_start = float(temp_start)
        self.temp_end = float(temp_end)

        # base param set (adapted)
        self.p = p_base.copy()
        # slight scaling by aggressiveness to allow more aggressive steps when requested
        self.p["sigma1"] *= 1.0 + 0.4 * self.aggressiveness
        self.p["eta"] *= 1.0 + 0.5 * self.aggressiveness
        self.p["lmbda"] *= 1.0 + 0.35 * self.aggressiveness
        self.p["top_k"] = self.top_k

        # conversion params
        self.conv_period = max(1, int(conv_period))
        self.top_promote_frac = float(np.clip(top_promote_frac, 0.0, 0.5))
        self.bottom_demote_frac = float(np.clip(bottom_demote_frac, 0.0, 0.5))
        self.allow_fractional_adjust = bool(allow_fractional_adjust)

        # state
        self.positions = self.rng.uniform(self.lb, self.ub, (self.n_agents, self.dim))
        self.velocities = np.zeros((self.n_agents, self.dim))
        self.fitness = np.full(self.n_agents, np.inf)
        self.best = None
        self.best_score = np.inf
        self.best_idx = None
        self.archive = []

        # histories
        self.history_positions = [self.positions.copy()]
        self.history_best = [None]
        self.role_history = [self.roles.copy()]
        self.diversity_history = []
        self.temp_history = []
        self.best_history = []

        # target counts from fractions (for rebalance)
        self.target_role_counts = {
            r: int(np.round(self.roles_frac.get(r, 0) * self.n_agents))
            for r in ("initiator", "duelist", "controller", "sentinel")
        }
        # fix rounding
        total = sum(self.target_role_counts.values())
        if total != self.n_agents:
            self.target_role_counts["initiator"] += (self.n_agents - total)

        # stagnation / auto boost parameters
        self.stagnation_threshold = int(max(1, stagnation_threshold))
        self.auto_boost_factor = max(1.0, float(auto_boost_factor))
        self.iter = 0
        self.max_iter = None
        self.last_improvement_iter = 0
        self.best_history = []

    # ---- utilities ----
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

    def _population_diversity(self):
        # mean per-dimension std normalized by domain
        std = np.mean(np.std(self.positions, axis=0))
        domain = np.mean(self.ub - self.lb) + 1e-12
        return float(std / domain)

    # ---- robust softmax sampling helper with fallback ----
    def _sample_with_probs_safe(self, pool_indices, scores, size, replace=False):
        """
        pool_indices: list/array of indices
        scores: array of raw scores (higher -> better)
        size: desired sample size
        replace: whether to sample with replacement
        Returns chosen indices (list)
        """
        if len(pool_indices) == 0:
            return []

        # extract pool scores
        pool_scores = np.asarray(scores)[pool_indices].astype(float)

        # compute softmax probabilities (stable)
        maxs = np.max(pool_scores)
        exps = np.exp(pool_scores - maxs)
        probs = exps / (np.sum(exps) + 1e-12)

        # filter nonzero entries
        nz_mask = probs > 1e-12
        if np.sum(nz_mask) == 0:
            # fallback deterministic top-k from pool by score
            sorted_pool = sorted(pool_indices, key=lambda i: scores[i], reverse=True)
            return sorted_pool[:min(size, len(sorted_pool))]

        pool_nz = np.array(pool_indices)[nz_mask]
        probs_nz = probs[nz_mask]
        # re-normalize
        probs_nz = probs_nz / (np.sum(probs_nz) + 1e-12)

        if len(pool_nz) < size and not replace:
            # fallback: take all pool_nz
            chosen = pool_nz.tolist()
            # if we still need more, fill deterministically from remaining best
            remaining = [i for i in pool_indices if i not in chosen]
            remaining_sorted = sorted(remaining, key=lambda i: scores[i], reverse=True)
            needed = size - len(chosen)
            chosen += remaining_sorted[:needed]
            return chosen
        try:
            chosen = self.rng.choice(pool_nz, size=size, replace=replace, p=probs_nz)
            return list(np.array(chosen, dtype=int))
        except Exception:
            # robust fallback deterministic top
            sorted_pool = sorted(pool_indices, key=lambda i: scores[i], reverse=True)
            return sorted_pool[:min(size, len(sorted_pool))]

    # ---- role rebalance ----
    def _rebalance_roles(self):
        if self.allow_fractional_adjust:
            return
        counts = {"initiator": 0, "duelist": 0, "controller": 0, "sentinel": 0}
        for r in self.roles:
            counts[r] += 1
        need = {r: self.target_role_counts[r] - counts[r] for r in counts}
        surplus = [r for r, v in need.items() if v < 0]
        deficit = [r for r, v in need.items() if v > 0]
        if not surplus and not deficit:
            return

        # move worst from surplus roles
        for role in surplus:
            how_many = -need[role]
            role_idx = [i for i, rr in enumerate(self.roles) if rr == role]
            role_idx_sorted = sorted(role_idx, key=lambda i: self.fitness[i], reverse=True)
            to_move = role_idx_sorted[:how_many]
            for i in to_move:
                self.roles[i] = None

        unassigned = [i for i, rr in enumerate(self.roles) if rr is None]
        unassigned_sorted = sorted(unassigned, key=lambda i: self.fitness[i])

        ptr = 0
        for role in deficit:
            cnt = need[role]
            for _ in range(cnt):
                if ptr >= len(unassigned_sorted):
                    break
                idx = unassigned_sorted[ptr]
                self.roles[idx] = role
                ptr += 1

        # fill any remaining None with initiators
        for i in range(len(self.roles)):
            if self.roles[i] is None:
                self.roles[i] = "initiator"

    # ---- adaptive conversion (balanced safe) ----
    def _adaptive_conversion(self):
        # temperature anneal based on progress
        tfrac = min(1.0, self.iter / max(1, self.max_iter))
        temp = self.temp_start * (1 - tfrac) + self.temp_end * tfrac
        self.temp_history.append(temp)

        # goodness = negative fitness (higher better)
        goodness = -self.fitness.copy()
        # select promotable pool excluding elites
        ordered = np.argsort(self.fitness)
        elite_set = set(ordered[:self.n_elite]) if self.n_elite > 0 else set()
        promotable_pool = [i for i in range(self.n_agents) if i not in elite_set]
        if len(promotable_pool) == 0:
            return

        # size parameters
        top_count = max(1, int(np.floor(self.top_promote_frac * self.n_agents)))
        bottom_count = max(1, int(np.floor(self.bottom_demote_frac * self.n_agents)))

        # sample promoters safely
        top_indices = self._sample_with_probs_safe(promotable_pool, goodness, size=min(top_count, len(promotable_pool)), replace=False)

        # demotion: pick worst performers probabilistically (by weakness)
        weakness = self.fitness.copy()
        demotable_pool = [i for i in range(self.n_agents) if i not in elite_set]
        bottom_indices = self._sample_with_probs_safe(demotable_pool, -weakness, size=min(bottom_count, len(demotable_pool)), replace=False)

        # soft promotions/demotions with conv_strength probability
        half = int(np.ceil(len(top_indices) / 2.0))
        promoters_ctrl = top_indices[:half]
        promoters_sent = top_indices[half:]

        for i in promoters_ctrl:
            if self.rng.rand() < self.conv_strength:
                self.roles[i] = "controller"
            else:
                self.roles[i] = "duelist"

        for i in promoters_sent:
            if self.rng.rand() < (0.5 * self.conv_strength):
                self.roles[i] = "sentinel"
            else:
                self.roles[i] = "controller"

        for i in bottom_indices:
            if self.rng.rand() < (0.9 * self.conv_strength):
                self.roles[i] = "initiator"
            else:
                self.roles[i] = "duelist"

        # small random smoothing to preserve diversity
        mid_pool = [i for i in range(self.n_agents) if i not in top_indices and i not in bottom_indices and i not in elite_set]
        if len(mid_pool) > 0:
            nm = max(1, int(0.01 * self.n_agents))
            chosen = self.rng.choice(mid_pool, size=min(nm, len(mid_pool)), replace=False)
            for i in chosen:
                if self.rng.rand() < 0.5:
                    self.roles[i] = "duelist"

        # detect stagnation and decide adaptive conv_period / auto-boost
        if len(self.best_history) >= 2:
            if self.best_history[-1] < self.best_history[-2]:
                self.last_improvement_iter = self.iter

        stagnation = self.iter - self.last_improvement_iter
        if stagnation > self.stagnation_threshold:
            # accelerate conversions and optionally boost sigma for more exploration
            self.conv_period_adaptive = max(1, int(self.conv_period * 0.5))
            # temporarily boost sigma1 for initiators
            self.p["sigma1"] *= self.auto_boost_factor
        else:
            self.conv_period_adaptive = self.conv_period

        # adjust target_role_counts mildly based on diversity
        div = self._population_diversity()
        if div < 0.015:
            # low diversity -> encourage initiators (explore)
            self.target_role_counts["initiator"] = min(self.n_agents, self.target_role_counts["initiator"] + 1)
            self.target_role_counts["controller"] = max(1, self.target_role_counts["controller"] - 1)
        elif div > 0.06:
            # high diversity -> encourage controllers
            self.target_role_counts["controller"] = min(self.n_agents, self.target_role_counts["controller"] + 1)
            self.target_role_counts["initiator"] = max(1, self.target_role_counts["initiator"] - 1)

        # re-balance counts to stay near original distribution (unless fractional adjust allowed)
        self._rebalance_roles()

    # ---- main step ----
    def _step(self):
        initiator_idx = [i for i, r in enumerate(self.roles) if r == "initiator"]
        duelists_idx = [i for i, r in enumerate(self.roles) if r == "duelist"]
        controllers_idx = [i for i, r in enumerate(self.roles) if r == "controller"]
        sentinels_idx = [i for i, r in enumerate(self.roles) if r == "sentinel"]

        hotspots = []
        pop_std = np.std(self.positions, axis=0)
        step_scale = float(np.mean(pop_std) + 1e-12)
        progress = self.iter / max(1, self.max_iter)

        # dynamic levy prob: decreases with progress, increases with aggressiveness
        levy_prob = float(np.clip(0.22 - 0.11 * progress + 0.12 * self.aggressiveness, 0.0, 0.65))

        # initiators: exploration (levy or gaussian based on levy_prob)
        for i in initiator_idx:
            sigma_adapt = self.p["sigma1"] * step_scale * (0.3 + 0.7 * (1.0 - progress)) * (1.0 + 0.6 * self.aggressiveness)
            if self.rng.rand() < levy_prob:
                step = 0.85 * self.levy_flight(self.p["beta"], self.dim, self.rng)
            else:
                step = sigma_adapt * self.rng.randn(self.dim)
            vel = self.vel_momentum * self.velocities[i] + (1 - self.vel_momentum) * step
            self.velocities[i] = vel
            self.positions[i] = self._clip(self.positions[i] + vel)
            hotspots.append(self.positions[i].copy())

        # compute hotspot targets
        if len(hotspots) > 0:
            hotspot_scores = np.array([self.func(h) for h in hotspots])
            k = min(self.p["top_k"], len(hotspots))
            top_idx = np.argsort(hotspot_scores)[:k]
            targets = np.array(hotspots)[top_idx]
        else:
            targets = self.positions[self.rng.choice(self.n_agents, size=min(3, self.n_agents), replace=False)]

        # duelists: move toward random hotspot target with smaller steps + some levy
        for d in duelists_idx:
            target = targets[self.rng.randint(0, len(targets))]
            lf = self.levy_flight(self.p["beta"], self.dim, self.rng)
            step = (self.p["sigma2"] * (target - self.positions[d]) * (1.0 + 0.45 * self.aggressiveness) +
                    0.45 * self.p["beta"] * lf)
            vel = self.vel_momentum * self.velocities[d] + (1 - self.vel_momentum) * step
            self.velocities[d] = vel
            self.positions[d] = self._clip(self.positions[d] + vel)

        # controllers: distance-weighted centroid of k nearest active agents (initiators+duelists)
        active_idx = duelists_idx + initiator_idx
        for c in controllers_idx:
            if len(active_idx) > 0:
                distances = np.linalg.norm(self.positions[active_idx] - self.positions[c], axis=1)
                k_max = min(self.p["top_k"], len(active_idx))
                dynamic_k = max(self.min_k_neighbors, int(k_max * (1.0 - progress) + 0.5))
                nearest_indices = np.argsort(distances)[:dynamic_k]
                nearest_agents = [active_idx[i] for i in nearest_indices]
                dists = distances[nearest_indices].astype(float)
                inv = 1.0 / (dists + 1e-9)
                weights = inv / (np.sum(inv) + 1e-12)
                centroid = np.sum(self.positions[nearest_agents] * weights[:, None], axis=0)
                step = self.p["eta"] * (centroid - self.positions[c]) * (1.0 + 0.6 * self.aggressiveness)
                vel = self.vel_momentum * self.velocities[c] + (1 - self.vel_momentum) * step
                self.velocities[c] = vel
                self.positions[c] = self._clip(self.positions[c] + vel)
            else:
                rand = self.positions[self.rng.randint(0, self.n_agents)]
                step = self.p["eta"] * (rand - self.positions[c]) + 0.1 * self.rng.randn(self.dim)
                vel = self.vel_momentum * self.velocities[c] + (1 - self.vel_momentum) * step
                self.velocities[c] = vel
                self.positions[c] = self._clip(self.positions[c] + vel)

        # sentinel behaviour: guard best + account for controller centroid
        controllers_exist = len(controllers_idx) > 0
        controller_centroid = (np.mean(self.positions[controllers_idx], axis=0) if controllers_exist else np.zeros(self.dim))
        for s in sentinels_idx:
            if self.best is not None:
                step = self.p["lmbda"] * (self.best - self.positions[s]) * (1.0 + 0.45 * self.aggressiveness) + 0.25 * (controller_centroid - self.positions[s])
                vel = self.vel_momentum * self.velocities[s] + (1 - self.vel_momentum) * step
                self.velocities[s] = vel
                self.positions[s] = self._clip(self.positions[s] + vel)
            if self.rng.rand() < 0.06:
                self.positions[s] = self._clip(self.positions[s] + 0.02 * self.rng.randn(self.dim))

        # evaluate
        self._evaluate()

        # update histories & metrics
        self.history_positions.append(self.positions.copy())
        self.history_best.append(self.best.copy() if self.best is not None else None)
        self.role_history.append(self.roles.copy())
        div = self._population_diversity()
        self.diversity_history.append(div)
        self.best_history.append(self.best_score)

        # track improvement
        if len(self.best_history) >= 2 and self.best_history[-1] < self.best_history[-2]:
            self.last_improvement_iter = self.iter

        # use adaptive conv_period if set by previous conversion, else default
        current_conv_period = getattr(self, "conv_period_adaptive", self.conv_period)
        if (self.conv_period > 0) and (self.iter % max(1, current_conv_period) == 0) and (self.iter > 0):
            self._adaptive_conversion()

        # if sigma1 was auto-boosted previously, slowly decay it back to baseline
        # simple decay mechanism:
        base_sigma1 = (0.75 if self.mode == "conservative" else (1.0 if self.mode == "balanced" else 1.2 * sigma1))
        self.p["sigma1"] = 0.97 * self.p["sigma1"] + 0.03 * (base_sigma1 * (1.0 + 0.4 * self.aggressiveness))

        self.iter += 1

    # ---- run / API ----
    def run(self, iterations=100):
        self.max_iter = int(iterations)
        self.iter = 0
        self._evaluate()
        self.best_history = [self.best_score]

        desc = f"ValoOptimized (Balanced+Safe) mode={self.mode}"
        with tqdm(total=iterations, desc=desc, ncols=100) as pbar:
            for _ in range(iterations):
                self._step()
                pbar.set_postfix({"Best": f"{self.best_score:.6g}", "Div": f"{self.diversity_history[-1]:.4g}"})
                pbar.update(1)

        return {
            "history_positions": self.history_positions,
            "history_best": self.history_best,
            "role_history": self.role_history,
            "diversity_history": self.diversity_history,
            "temp_history": self.temp_history,
            "best_score": self.best_score
        }

    def get_best_solution(self):
        return {
            "position": self.best.copy() if self.best is not None else None,
            "score": self.best_score,
            "index": self.best_idx
        }


# -------------------------
# Example usage (Rastrigin)
# -------------------------
if __name__ == "__main__":
    def rastrigin(x):
        x = np.asarray(x)
        n = len(x)
        return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

    dim = 100
    lb = -100 * np.ones(dim)
    ub = 100 * np.ones(dim)

    opt = ValoEnhancedKNNAdaptiveOptimized(
        rastrigin, lb, ub, dim,
        n_agents=300, seed=42,
        mode="balanced",
        conv_period=12,
        top_promote_frac=0.06,
        bottom_demote_frac=0.08,
        allow_fractional_adjust=False,
        aggressiveness=0.7,
        conv_strength=0.9,
        n_elite=6,
        vel_momentum=0.7,
        min_k_neighbors=8,
        temp_start=1.0, temp_end=0.05,
        stagnation_threshold=18,
        auto_boost_factor=1.25
    )

    results = opt.run(iterations=200)
    print("\nBest score:", results["best_score"])
    best = opt.get_best_solution()
    print("Best position (first 6 dims):", best["position"][:6])
    from collections import Counter
    final_roles = Counter(results["role_history"][-1])
    print("Final role counts:", final_roles)
