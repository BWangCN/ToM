"""Gym-style 1x1 continuous arena with a lag-pursuing opponent (CLAUDE.md §3.2).

Shared by the toy-data generator and the closed-loop ablation eval — this is the
single implementation of dynamics, rendering, scripted policies, and the
outcome_deceived rule.

Dynamics (10 Hz control):
  - ego action a = (dx, dy) in [-1,1]^2; ego moves a * EGO_STEP per step.
  - opponent pursues using a 3-STEP-LAGGED heading estimate of ego motion:
    at step t it aims at  ego_pos[t-3] + heading(ego, t-4 -> t-3) * LEAD.
    The lag is what makes feints work.

outcome_deceived (single rule, used for data labels and rollout eval):
  k_pass    = first step >= kf where ego crosses the opponent's y-level
              (fallback kf+H)                     # the actual passing moment
  pass_side = sign(ego.x[k_pass] - opp.x[k_pass]) # flank of the guard ego takes
  opp_dx    = opp.x[kf+5] - opp.x[kf]             # opponent lateral commitment
  deceived  = (opp_dx * pass_side) < COMMIT_THR   # failed to commit toward truth
"""

import numpy as np
from PIL import Image, ImageDraw

EGO_STEP = 0.035
OPP_STEP = 0.026
LAG = 3
LEAD = 0.12
COMMIT_THR = 0.004
EPISODE_LEN = 40
CHUNK_LEN = 8
GOAL = np.array([0.5, 0.88])
GOAL_TOL = 0.06
ARENA_MIN, ARENA_MAX = 0.01, 0.99

DECISIONS = ["HONEST_DIRECT", "FEINT_LEFT_GO_RIGHT", "FEINT_RIGHT_GO_LEFT", "HESITATION_FAKE"]

FALSE_BELIEFS = {
    "HONEST_DIRECT": "none",
    "FEINT_LEFT_GO_RIGHT": "B believes A will take the left corridor",
    "FEINT_RIGHT_GO_LEFT": "B believes A will take the right corridor",
    "HESITATION_FAKE": "B believes A has not committed to a side yet",
}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


class ToyEnv:
    """reset/step/render. Opponent is internal; ego action is external."""

    def __init__(self, ego_start=None, opp_start=None, image_size=256):
        self.image_size = image_size
        self._ego_start = np.array(ego_start if ego_start is not None else [0.5, 0.12])
        self._opp_start = np.array(opp_start if opp_start is not None else [0.55, 0.62])

    def reset(self):
        self.ego = self._ego_start.astype(np.float64).copy()
        self.opp = self._opp_start.astype(np.float64).copy()
        self.t = 0
        self.ego_hist = [self.ego.copy()]  # position at end of step t (index t)
        self.opp_hist = [self.opp.copy()]
        return self._obs()

    def _obs(self):
        return {"ego": self.ego.copy(), "opp": self.opp.copy(), "goal": GOAL.copy(), "t": self.t}

    def _opp_move(self):
        # 3-step-lagged heading estimate: needs history up to t-3 and t-4.
        if len(self.ego_hist) > LAG + 1:
            p_lag = self.ego_hist[-1 - LAG]
            p_lag_prev = self.ego_hist[-2 - LAG]
            heading = _unit(p_lag - p_lag_prev)
            target = p_lag + heading * LEAD
            step = _unit(target - self.opp) * OPP_STEP
            self.opp = np.clip(self.opp + step, ARENA_MIN, ARENA_MAX)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.ego = np.clip(self.ego + action * EGO_STEP, ARENA_MIN, ARENA_MAX)
        self._opp_move()
        self.t += 1
        self.ego_hist.append(self.ego.copy())
        self.opp_hist.append(self.opp.copy())
        done = self.t >= EPISODE_LEN
        return self._obs(), 0.0, done, {}

    def render(self):
        return render_state(self.ego, self.opp, self.image_size)


def render_state(ego, opp, image_size=256):
    """Render arbitrary (ego, opp) positions — lets callers re-render from history."""
    s = image_size
    img = Image.new("RGB", (s, s), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def px(p):  # world -> pixel, y up
        return (p[0] * s, (1.0 - p[1]) * s)

    gx, gy = px(GOAL)
    half = 12
    d.rectangle([gx - half, gy - half, gx + half, gy + half], fill=(40, 180, 70))
    r = 9
    ox, oy = px(opp)
    d.ellipse([ox - r, oy - r, ox + r, oy + r], fill=(220, 50, 50))
    ex, ey = px(ego)
    d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(40, 90, 220))
    return img


# ---------------------------------------------------------------- outcomes

def outcome_deceived(ego_traj, opp_traj, keyframe, chunk_len=CHUNK_LEN):
    """The single deception rule (see module docstring). (N,2) position arrays."""
    ego = np.asarray(ego_traj)
    opp = np.asarray(opp_traj)
    T = len(ego) - 1
    kf = keyframe
    k5 = min(kf + 5, T)
    k_pass = T  # fallback: end of episode
    for t in range(kf, T + 1):
        if ego[t, 1] >= opp[t, 1] + 0.04:  # clearly past the guard vertically
            k_pass = t
            break
    pass_side = np.sign(ego[k_pass, 0] - opp[k_pass, 0])
    if pass_side == 0:
        pass_side = 1.0
    opp_dx = opp[k5, 0] - opp[kf, 0]
    return bool(opp_dx * pass_side < COMMIT_THR)


def detect_commit_keyframe(ego_xs, lo=8, hi=24, w=3):
    """Rollout eval: keyframe = step in [lo,hi] with max mean |x-velocity| over next w."""
    ego_xs = np.asarray(ego_xs)
    hi = min(hi, len(ego_xs) - 1 - w)
    best_kf, best_v = lo, -1.0
    for kf in range(lo, hi + 1):
        v = abs(ego_xs[kf + w] - ego_xs[kf]) / w
        if v > best_v:
            best_v, best_kf = v, kf
    return best_kf


def rollout_metrics(ego_hist, opp_hist):
    """Closed-loop metrics: deceived flag (at detected commit keyframe) + goal reached."""
    ego = np.asarray(ego_hist)
    opp = np.asarray(opp_hist)
    kf = detect_commit_keyframe(ego[:, 0])
    deceived = outcome_deceived(ego, opp, kf)
    dists = np.linalg.norm(ego - GOAL[None, :], axis=1)
    return {"deceived": deceived, "goal_reached": bool((dists <= GOAL_TOL).any()),
            "detected_keyframe": int(kf)}


# ---------------------------------------------------------------- scripted egos

class ScriptedEgo:
    """Four decision policies. Phases: APPROACH -> MANEUVER (8 steps, = GT chunk) -> HOME.

    The keyframe is the first MANEUVER step; the 8-step maneuver is the action chunk.
    """

    def __init__(self, decision, rng):
        assert decision in DECISIONS
        self.decision = decision
        self.trigger_y_gap = 0.24 + rng.uniform(-0.03, 0.03)
        self.trigger_dist = 0.30 + rng.uniform(-0.03, 0.03)
        self.force_kf = int(rng.integers(20, 25))
        self.jit = rng.uniform(0.9, 1.0)  # maneuver magnitude jitter
        self.phase = "APPROACH"
        self.m_step = 0
        self.keyframe = None
        self.open_side = None   # flank the ego intends to pass on (+1 right, -1 left)
        self.opp_kf = None      # opponent position snapshot at the keyframe
        self._opp_xs = []       # observed opponent x history (state <= t only)

    def _maneuver_action(self, obs):
        """Opponent-anchored 8-step maneuver = the GT action chunk."""
        i, j = self.m_step, self.jit
        d = self.decision
        ego = obs["ego"]
        wx = self.opp_kf[0] + 0.22 * self.open_side  # corridor waypoint x
        cut = np.array([np.clip(6.0 * (wx - ego[0]), -1, 1) * j, 0.45])
        if d == "FEINT_LEFT_GO_RIGHT" or d == "FEINT_RIGHT_GO_LEFT":
            feint = np.array([-0.9 * j * self.open_side, 0.25])
            return feint if i < 4 else cut
        if d == "HESITATION_FAKE":
            if i < 3:
                return np.array([0.12 * (-1) ** i, 0.04])
            return np.array([np.clip(6.0 * (wx - ego[0]), -1, 1) * j, 0.50])
        # HONEST_DIRECT: commit immediately to the open corridor
        return cut

    def act(self, obs):
        ego, opp = obs["ego"], obs["opp"]
        self._opp_xs.append(float(opp[0]))
        if self.phase == "APPROACH":
            trigger = (ego[1] >= opp[1] - self.trigger_y_gap
                       or np.linalg.norm(ego - opp) <= self.trigger_dist
                       or obs["t"] >= self.force_kf)
            if trigger and obs["t"] >= 8:
                self.phase = "MANEUVER"
                self.keyframe = obs["t"]
                self.opp_kf = opp.copy()
                if self.decision == "FEINT_LEFT_GO_RIGHT":
                    self.open_side = 1.0    # feint left, pass right
                elif self.decision == "FEINT_RIGHT_GO_LEFT":
                    self.open_side = -1.0   # feint right, pass left
                elif self.decision == "HESITATION_FAKE":
                    # burst against the opponent's current lateral momentum
                    drift = (self._opp_xs[-1] - self._opp_xs[-3]
                             if len(self._opp_xs) >= 3 else 0.0)
                    if abs(drift) > 1e-4:
                        self.open_side = float(-np.sign(drift))
                    else:
                        dx = ego[0] - opp[0]
                        self.open_side = (float(np.sign(dx)) if abs(dx) > 0.02
                                          else float(-np.sign(opp[0] - 0.5) or 1.0))
                else:
                    dx = ego[0] - opp[0]
                    self.open_side = (float(np.sign(dx)) if abs(dx) > 0.02
                                      else float(-np.sign(opp[0] - 0.5) or 1.0))
        if self.phase == "APPROACH":
            a = _unit(GOAL - ego) + np.array([0.2 * (0.5 - ego[0]), 0.0])
            return np.clip(a, -1, 1)
        if self.phase == "MANEUVER":
            a = self._maneuver_action(obs)
            self.m_step += 1
            if self.m_step >= CHUNK_LEN:
                self.phase = "HOME"
            return np.clip(a, -1, 1)
        # HOME: thread the chosen corridor past the guard, then head for the goal
        if ego[1] < opp[1] + 0.03:
            target = np.array([opp[0] + 0.20 * self.open_side, opp[1] + 0.12])
        else:
            target = GOAL
        a = _unit(target - ego)
        if np.linalg.norm(ego - opp) < 0.10:
            a = a + np.array([0.8 * float(np.sign(ego[0] - opp[0]) or self.open_side), 0.0])
        return np.clip(a, -1, 1)


def make_episode_env(episode_idx, seed=0, image_size=256):
    """Deterministic per-episode layout + decision (round-robin).

    For feint episodes the opponent starts on the side the feint targets — you
    feint into his bias to pull him further across, then cut through the side
    he vacates (this matches the reasoning templates: "sits right-of-center;
    a left step makes him commit left, opening the right").
    """
    rng = np.random.default_rng(seed * 100003 + episode_idx)
    decision = DECISIONS[episode_idx % len(DECISIONS)]
    ego_start = [0.5 + rng.uniform(-0.05, 0.05), 0.12 + rng.uniform(-0.02, 0.02)]
    if decision == "FEINT_LEFT_GO_RIGHT":
        side = 1.0    # guard biased right-of-center; feint left, pass right
    elif decision == "FEINT_RIGHT_GO_LEFT":
        side = -1.0   # guard biased left-of-center; feint right, pass left
    else:
        side = float(rng.choice([-1.0, 1.0]))
    opp_start = [0.5 + side * rng.uniform(0.03, 0.13), 0.62 + rng.uniform(-0.04, 0.04)]
    env = ToyEnv(ego_start=ego_start, opp_start=opp_start, image_size=image_size)
    policy = ScriptedEgo(decision, rng)
    return env, policy, decision, rng
