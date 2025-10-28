import numpy as np
import math
from tqdm import tqdm



class ValoBellmanRL:
    """
    ValoBellmanRL: Controlled LISTEN style swarm with a Bellman/Q-learning
    policy layer for course-correction and adaptive movement selection.
    """

    def __init__(self, func, lb, ub, dim, n_agents=40, roles_frac=None, seed=None,
                 sigma1=0.8, sigma2=0.12, beta=1.5, eta=0.25, lmbda=2, top_k=None,
                 mode="balanced",
                 # RL hyperparams (Bellman/Q-learning)
                 rl_alpha=0.2, rl_gamma=0.9, rl_epsilon=0.3,
                 rl_min_epsilon=0.02, rl_epsilon_decay=0.995,
                 rl_progress_bins=10, rl_rank_bins=10):
        # objective and domain
        self.func = func
        self.lb, self.ub, self.dim = np.array(lb), np.array(ub), dim

        # population and RNG
        self.n_agents = n_agents
        self.rng = np.random.RandomState(seed)
        self.mode = mode.lower()

        # top_k heuristic
        if top_k is None:
            self.top_k = max(1, int(0.02 * n_agents * np.log2(dim + 1)))
        else:
            self.top_k = top_k

        # roles
        if roles_frac is None:
            roles_frac = {"initiator": 0.30, "duelist": 0.35, "controller": 0.20, "sentinel": 0.15}
        self.roles_frac = roles_frac

        n_init = int(np.round(n_agents * roles_frac["initiator"]))
        n_duel = int(np.round(n_agents * roles_frac["duelist"]))
        n_ctrl = int(np.round(n_agents * roles_frac["controller"]))
        n_sent = n_agents - (n_init + n_duel + n_ctrl)
        self.roles = (["initiator"] * n_init + ["duelist"] * n_duel +
                      ["controller"] * n_ctrl + ["sentinel"] * n_sent)
        # adjust if rounding issues
        while len(self.roles) > n_agents:
            self.roles.pop()
        while len(self.roles) < n_agents:
            self.roles.append("initiator")

        # behavior parameters (mode presets)
        if self.mode == "conservative":
            self.p = {"sigma1": 0.8, "sigma2": sigma2, "beta": beta, "eta": 0.25, "lmbda": 0.2}
        elif self.mode == "balanced":
            self.p = {"sigma1": 1.0, "sigma2": sigma2, "beta": beta, "eta": 0.4, "lmbda": 1.5}
        elif self.mode == "aggressive":
            self.p = {"sigma1": 1.2, "sigma2": 0.2, "beta": beta, "eta": 0.55, "lmbda": 0.1}
        else:
            raise ValueError("Mode must be one of: conservative | balanced | aggressive")
        self.p["top_k"] = self.top_k
        self.p["sigma2"] = sigma2

        # population state
        self.positions = self.rng.uniform(lb, ub, (n_agents, dim))
        self.fitness = np.full(n_agents, np.inf)
        self.prev_fitness = np.full(n_agents, np.inf)  # used for reward calc
        self.best = None
        self.best_score = np.inf
        self.best_idx = None
        self.archive = []

        self.history_positions = [self.positions.copy()]
        self.history_best = [None]

        # RL / Bellman-Q components
        self.actions = ["levy", "towards_best", "centroid_follow", "exploit_target", "random_walk"]
        self.n_actions = len(self.actions)
        # Q table: dict key (role_id, progress_bin, rank_bin) -> np.array(n_actions)
        self.Q = {}
        self.rl_alpha = rl_alpha
        self.rl_gamma = rl_gamma
        self.rl_epsilon = rl_epsilon
        self.rl_min_epsilon = rl_min_epsilon
        self.rl_epsilon_decay = rl_epsilon_decay
        self.rl_progress_bins = rl_progress_bins
        self.rl_rank_bins = rl_rank_bins
        self.role_to_id = {"initiator": 0, "duelist": 1, "controller": 2, "sentinel": 3}

        # storage for per-step chosen actions to update Q after seeing reward
        self._recent_actions = {}

    # ---- utility / movement ----
    def levy_flight(self, beta, dim, rng):
        # Mantegna's algorithm like step
        sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                   (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = rng.normal(0, sigma_u, size=dim)
        v = rng.normal(0, 1, size=dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    # ---- evaluation ----
    def _evaluate(self):
        """Evaluate positions and update population fitness and best."""
        for i in range(self.n_agents):
            f = self.func(self.positions[i])
            self.prev_fitness[i] = self.fitness[i] if np.isfinite(self.fitness[i]) else f + 0.0
            self.fitness[i] = f
            if f < self.best_score:
                self.best_score = f
                self.best = self.positions[i].copy()
                self.best_idx = i

    # ---- RL (Bellman/Q) helpers ----
    def _get_state(self, agent_idx, fitness_snapshot, progress):
        """Discrete state: (role_id, progress_bin, rank_bin)."""
        role = self.roles[agent_idx]
        role_id = self.role_to_id.get(role, 0)
        progress_bin = int(np.floor(progress * (self.rl_progress_bins - 1e-9)))
        ranks = np.argsort(np.argsort(fitness_snapshot))  # 0 is best
        rank_normalized = ranks[agent_idx] / max(1, (self.n_agents - 1))
        rank_bin = int(np.floor(rank_normalized * (self.rl_rank_bins - 1e-9)))
        return (role_id, progress_bin, rank_bin)

    def _ensure_q(self, state):
        if state not in self.Q:
            # initialize to zeros (could use small random noise)
            self.Q[state] = np.zeros(self.n_actions)

    def _select_action(self, state):
        self._ensure_q(state)
        if self.rng.rand() < self.rl_epsilon:
            return int(self.rng.randint(0, self.n_actions))
        q = self.Q[state]
        # deterministic tie-breaker random
        maxv = q.max()
        choices = np.flatnonzero(q == maxv)
        return int(self.rng.choice(choices))

    def _update_q(self, state, action_idx, reward, next_state):
        self._ensure_q(state)
        self._ensure_q(next_state)
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.rl_gamma * best_next
        td_error = td_target - self.Q[state][action_idx]
        self.Q[state][action_idx] += self.rl_alpha * td_error

    # ---- action implementations ----
    def _apply_action(self, i, action_idx, targets, controller_centroid, step_scale, progress):
        pos = self.positions[i]
        action = self.actions[action_idx]
        if action == "levy":
            lf = self.levy_flight(self.p["beta"], self.dim, self.rng)
            step = 0.7 * lf * (0.6 + (1 - progress))
            self.positions[i] = self._clip(pos + step)
        elif action == "towards_best":
            if self.best is not None:
                step = self.p["lmbda"] * (self.best - pos) + 0.02 * self.rng.randn(self.dim)
                self.positions[i] = self._clip(pos + step)
            else:
                self.positions[i] = self._clip(pos + 0.02 * self.rng.randn(self.dim))
        elif action == "centroid_follow":
            if controller_centroid is not None:
                step = self.p["eta"] * (controller_centroid - pos) + 0.08 * self.rng.randn(self.dim)
                self.positions[i] = self._clip(pos + step)
            else:
                # fallback to small move to target
                tgt = targets[self.rng.randint(0, len(targets))]
                step = self.p["eta"] * (tgt - pos) + 0.05 * self.rng.randn(self.dim)
                self.positions[i] = self._clip(pos + step)
        elif action == "exploit_target":
            tgt = targets[self.rng.randint(0, len(targets))]
            step = self.p["sigma2"] * (tgt - pos) + 0.25 * self.rng.randn(self.dim)
            self.positions[i] = self._clip(pos + step)
        elif action == "random_walk":
            self.positions[i] = self._clip(pos + 0.015 * self.rng.randn(self.dim))
        else:
            self.positions[i] = self._clip(pos + 0.01 * self.rng.randn(self.dim))

    # ---- main optimizer step ----
    def _step(self):
        # role indices
        initiator_idx = [i for i, r in enumerate(self.roles) if r == "initiator"]
        duelists_idx = [i for i, r in enumerate(self.roles) if r == "duelist"]
        controllers_idx = [i for i, r in enumerate(self.roles) if r == "controller"]
        sentinels_idx = [i for i, r in enumerate(self.roles) if r == "sentinel"]

        hotspots = []
        pop_std = np.std(self.positions, axis=0)
        step_scale = float(np.mean(pop_std))
        progress = self.iter / self.max_iter

        # snapshot of old fitness for reward calc
        old_fitness = self.fitness.copy()

        # Initiator exploratory moves (seed hotspots)
        levy_prob = max(0.0, 0.2 - 0.15 * progress)
        for i in initiator_idx:
            sigma_adapt = self.p["sigma1"] * step_scale * (0.3 + 0.7 * (1 - progress))
            if self.rng.rand() < levy_prob:
                step = 0.8 * self.levy_flight(self.p["beta"], self.dim, self.rng)
            else:
                step = sigma_adapt * self.rng.randn(self.dim)
            self.positions[i] = self._clip(self.positions[i] + step)
            hotspots.append(self.positions[i].copy())

        # determine targets
        if len(hotspots) > 0:
            hotspot_scores = np.array([self.func(h) for h in hotspots])
            k = min(self.p["top_k"], len(hotspots))
            top_idx = np.argsort(hotspot_scores)[:k]
            targets = np.array(hotspots)[top_idx]
        else:
            targets = self.positions[self.rng.choice(self.n_agents, size=min(3, self.n_agents), replace=False)]

        # controller centroid
        controllers_exist = len(controllers_idx) > 0
        controller_centroid = (np.mean(self.positions[controllers_idx], axis=0)
                               if controllers_exist else None)

        # choose & apply action for each agent via Q-policy
        snapshot_fitness = self.fitness.copy()  # used to compute rank in state
        all_indices = list(range(self.n_agents))
        for i in all_indices:
            state = self._get_state(i, snapshot_fitness, progress)
            action_idx = self._select_action(state)
            # apply action (immediately modifies positions)
            self._apply_action(i, action_idx, targets, controller_centroid, step_scale, progress)
            # store chosen action for later Bellman update
            self._recent_actions[i] = (state, action_idx)

        # optional lightweight role-driven nudges (hybrid)
        for c in controllers_idx:
            active_idx = duelists_idx + initiator_idx
            if len(active_idx) > 0:
                k_max = min(self.p["top_k"], len(active_idx))
                k = max(3, int(k_max * (1 - progress) + 0.5))
                chosen = self.rng.choice(active_idx, size=k, replace=False)
                centroid = np.mean(self.positions[chosen], axis=0)
                step = self.p["eta"] * (centroid - self.positions[c]) + 0.04 * self.rng.randn(self.dim)
                self.positions[c] = self._clip(self.positions[c] + step)

        for s in sentinels_idx:
            if self.best is not None:
                ctrl_cent = controller_centroid if controller_centroid is not None else np.zeros(self.dim)
                step = self.p["lmbda"] * (self.best - self.positions[s]) + 0.2 * (ctrl_cent - self.positions[s])
                self.positions[s] = self._clip(self.positions[s] + step)
            if self.rng.rand() < 0.05:
                self.positions[s] = self._clip(self.positions[s] + 0.02 * self.rng.randn(self.dim))

        # evaluate after movement
        self._evaluate()

        # rewards: positive = improvement (old - new)
        rewards = old_fitness - self.fitness

        # Bellman/Q updates for each agent
        for i in range(self.n_agents):
            if i not in self._recent_actions:
                continue
            state, action_idx = self._recent_actions[i]
            next_state = self._get_state(i, self.fitness, progress)
            reward = rewards[i]
            self._update_q(state, action_idx, reward, next_state)

        # clear recent actions store
        self._recent_actions.clear()

        # update archive and history
        combined = sorted(zip(self.fitness, [x.copy() for x in self.positions]), key=lambda z: z[0])
        elites = [pos for (score, pos) in combined[:min(5, len(combined))]]
        self.archive = elites
        self.history_positions.append(self.positions.copy())
        self.history_best.append(self.best.copy() if self.best is not None else None)

        # iteration increment and epsilon annealing
        self.iter += 1
        # anneal epsilon but not below min
        self.rl_epsilon = max(self.rl_min_epsilon, self.rl_epsilon * self.rl_epsilon_decay)

    # ---- run + utilities ----
    def run(self, iterations=100, verbose=True):
        self.max_iter = iterations
        self.iter = 0
        self._evaluate()  # initial eval

        desc = f"ValoBellmanRL ({self.mode.title()} + Bellman-Q)"
        with tqdm(total=iterations, desc=desc, ncols=100, disable=(not verbose)) as pbar:
            for _ in range(iterations):
                self._step()
                pbar.set_postfix({"Best": f"{self.best_score:.6e}", "eps": f"{self.rl_epsilon:.3f}"})
                pbar.update(1)
        return self.history_positions, self.history_best, self.best_score

    def get_best_solution(self):
        return {"position": self.best.copy() if self.best is not None else None,
                "score": self.best_score, "index": self.best_idx}


def rastrigin(x):
    x = np.asarray(x)
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

dim = 100
lb = -100 * np.ones(dim)
ub = 100 * np.ones(dim)

opt = ValoBellmanRL(func=rastrigin, lb=lb, ub=ub, dim=dim, n_agents=600, seed=123,
                    mode="balanced", rl_alpha=0.25, rl_gamma=0.95,
                    rl_epsilon=0.4, rl_epsilon_decay=0.990,
                    rl_progress_bins=4, rl_rank_bins=8)

hist_pos, hist_best, best_score = opt.run(iterations=200, verbose=True)
print("\nBest score:", best_score)
print("Best position (first 6 dims):", opt.get_best_solution()["position"][:6])
