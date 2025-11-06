
from __future__ import annotations

"""
BrachistochroneScene.py — ManimCE
---------------------------------
Illustrates the brachistochrone problem with three candidate paths between two points:
1) Straight line (shortest distance but not fastest).
2) Steep dip curve (large initial acceleration but longer total path).
3) Cycloid (optimal time under gravity), with rolling without slipping included.

Features:
- Physics is computed separately from the animation.
- A ValueTracker drives time so animation time == physical time (1:1).
- Rolling without slipping: v(y) = sqrt(2 g Δy / (1 + k)), k = I/(m r^2). For a solid sphere, k = 2/5.

Usage (from terminal):
    manim -pqh BrachistochroneScene.py BrachistochroneComparison
    manim -p BrachistochroneScene.py BrachistochroneComparison

Requires: Manim Community Edition (pip install manim==0.*), NumPy.
"""

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


from manim import config
# ---- Render & animation knobs ----
DISCRETE_MODE = True      # if True, write exactly ANIM_STEPS+1 frames (no long plays)
DISPLAY_SAMPLES = 600      # path polyline points for drawing only (faster per-frame)
# Physics resolution (for accurate timing integrals)
PHYS_SAMPLES = 4001        # increase for more accuracy, cost is minimal (NumPy only)
# Animation sampling (number of time steps we visibly update on screen)
ANIM_STEPS = 180           # reduce to cut visible steps (e.g., 90, 120); increases speed
# Speed-up factor: physical seconds divided by this factor becomes on-screen run_time
TIME_SPEED = 4.0           # e.g. 4.0 → 1 sec phys = 0.25 sec animation
# Lower frame rate to reduce total frames rendered
config.frame_rate = 12     # e.g., 12 fps renders 4x fewer frames than 48 fps
# Optionally scale down resolution for even faster renders:
# config.pixel_width = 1280
# config.pixel_height = 720
from manim import (
    Axes,
    BLUE,
    GREEN,
    GREY_B,
    RED,
    Scene,
    Tex,
    VGroup,
    VMobject,
    ValueTracker,
    Dot,
    Line,
    always_redraw,
    rate_functions,
    DOWN, UP, LEFT, RIGHT, UL, UR, DL, DR, ORIGIN,
)


def _downsample_poly(P: np.ndarray, n: int) -> np.ndarray:
    if n >= len(P):
        return P
    idx = np.linspace(0, len(P)-1, n).round().astype(int)
    return P[idx]

# ------------------------------


# ------------------------------
# Physics helpers (no Manim here)
# ------------------------------

@dataclass
class PathParam:
    """Container for a geometric path parameterization r(u) over u ∈ [0,1]."""
    r_of_u: Callable[[np.ndarray], np.ndarray]  # (N,) -> (N, 2) array of (x,y)
    # Note: r_of_u should be vectorized over u.

def arclength_and_time(
    r_of_u: Callable[[np.ndarray], np.ndarray],
    u_grid: np.ndarray,
    g: float,
    y0: float,
    rolling_k: float = 0.0,  # k = I/(m r^2), solid sphere default
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cumulative arclength s(u) and time t(u) along a path r(u) with gravity only.
    Speed v depends only on vertical drop Δy = (y0 - y). For rolling without slipping:
        v = sqrt(2 g Δy / (1 + k))
    where k = I/(m r^2). For a sliding point mass, set k = 0.

    Returns:
        u_grid: same as input
        s_cum: cumulative arc length at grid points
        t_cum: cumulative time at grid points
        y_vals: y(u) at grid points
    """
    pts = r_of_u(u_grid)  # shape (N, 2)
    dx = np.gradient(pts[:, 0], u_grid, edge_order=2)
    dy = np.gradient(pts[:, 1], u_grid, edge_order=2)
    ds_du = np.sqrt(dx * dx + dy * dy)

    s_cum = np.cumsum(0.5 * (ds_du[1:] + ds_du[:-1]) * np.diff(u_grid))
    s_cum = np.insert(s_cum, 0, 0.0)

    y_vals = pts[:, 1]
    # Clamp Δy >= 0 for numeric safety (no imaginary speeds if y rises slightly due to discretization).
    dy_drop = np.maximum(0.0, y0 - y_vals)
    v = np.sqrt(2.0 * g * dy_drop / (1.0 + rolling_k))

    # Avoid division by zero at the very start where v=0: regularize with tiny epsilon for integration
    v = np.maximum(v, 1e-9)

    # Time integral: t = ∫ ds / v
    t_integrand = ds_du / v
    t_cum = np.cumsum(0.5 * (t_integrand[1:] + t_integrand[:-1]) * np.diff(u_grid))
    t_cum = np.insert(t_cum, 0, 0.0)

    return u_grid, s_cum, t_cum, y_vals


def invert_time(t_grid: np.ndarray, u_grid: np.ndarray) -> Callable[[float], float]:
    """
    Return a function u(t) by monotonic interpolation of t(u).
    Assumes t_grid is increasing w.r.t u_grid (no stalling upward).
    """
    # Ensure monotonicity: enforce strictly increasing with small eps if needed.
    t_grid = np.maximum.accumulate(t_grid)
    def u_of_t(t: float) -> float:
        if t <= t_grid[0]:
            return u_grid[0]
        if t >= t_grid[-1]:
            return u_grid[-1]
        idx = np.searchsorted(t_grid, t, side="right") - 1
        # Linear interpolation
        t0, t1 = t_grid[idx], t_grid[idx + 1]
        u0, u1 = u_grid[idx], u_grid[idx + 1]
        w = (t - t0) / max(t1 - t0, 1e-12)
        return (1 - w) * u0 + w * u1
    return u_of_t


def solve_cycloid_theta1(L: float, H: float, tol: float = 1e-12, maxiter: int = 200) -> float:
    """
    Solve for θ1 that connects (0,0) to (L, -H) by a cycloid:
        x = R(θ - sin θ), y = -R(1 - cos θ), θ ∈ [0, θ1]
    where R = H / (1 - cos θ1). Then L must satisfy:
        L = R(θ1 - sin θ1) = H * (θ1 - sin θ1) / (1 - cos θ1)

    We solve f(θ1) = H*(θ1 - sin θ1)/(1 - cos θ1) - L = 0 for θ1 in (0, 2π).
    The optimal cycloid typically has θ1 ∈ (0, 2π); we search in (0, 2π).

    Returns θ1.
    """
    def f(theta):
        if np.isclose(1 - math.cos(theta), 0.0):
            return 1e9
        return H * (theta - math.sin(theta)) / (1 - math.cos(theta)) - L

    a, b = 1e-6, 2.0 * math.pi - 1e-6
    fa, fb = f(a), f(b)
    # Ensure opposite signs for bisection; if not, try widening very slightly
    if fa * fb > 0:
        # Try a fallback bracket scan
        thetas = np.linspace(1e-6, 2.0 * math.pi - 1e-6, 2000)
        vals = [f(th) for th in thetas]
        sign_changes = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
        if len(sign_changes) == 0:
            # As a last resort, just return something near π
            return math.pi + 1e-3
        i0 = sign_changes[0]
        a, b = thetas[i0], thetas[i0 + 1]
        fa, fb = vals[i0], vals[i0 + 1]

    # Bisection
    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def make_cycloid_between(
    A: Tuple[float, float],
    B: Tuple[float, float],
    n_samples: int = 1001,
) -> np.ndarray:
    """
    Return (N,2) array of cycloid points from A to B (top to bottom/right typically).
    A = (x0, y0), B = (x1, y1), with y1 < y0 expected for brachistochrone.
    """
    x0, y0 = A
    x1, y1 = B
    L = x1 - x0
    H = y0 - y1
    if L <= 0 or H <= 0:
        raise ValueError("Cycloid expects B to the right and below A (L>0, H>0).")
    theta1 = solve_cycloid_theta1(L, H)
    R = H / (1.0 - math.cos(theta1))

    thetas = np.linspace(0.0, theta1, n_samples)
    x = R * (thetas - np.sin(thetas)) + x0
    y = -R * (1.0 - np.cos(thetas)) + y0
    return np.column_stack([x, y])


def make_poly_bezier(
    A: Tuple[float, float],
    B: Tuple[float, float],
    ctrl1: Tuple[float, float],
    ctrl2: Tuple[float, float],
    n_samples: int = 1001,
) -> np.ndarray:
    """
    Cubic Bézier from A to B with two control points.
    P(u) = (1-u)^3 A + 3(1-u)^2 u C1 + 3(1-u) u^2 C2 + u^3 B
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C1 = np.array(ctrl1, dtype=float)
    C2 = np.array(ctrl2, dtype=float)
    u = np.linspace(0.0, 1.0, n_samples)[:, None]
    P = ((1 - u) ** 3) * A + 3 * ((1 - u) ** 2) * u * C1 + 3 * (1 - u) * (u ** 2) * C2 + (u ** 3) * B
    return P


def make_straight(
    A: Tuple[float, float],
    B: Tuple[float, float],
    n_samples: int = 1001,
) -> np.ndarray:
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    u = np.linspace(0.0, 1.0, n_samples)[:, None]
    P = (1 - u) * A + u * B
    return P


def prepare_motion_functions(
    P: np.ndarray,
    g: float = 9.81,
    rolling_k: float = 0.0,
) -> Tuple[Callable[[float], np.ndarray], float, Callable[[float], float]]:
    """
    Given a polyline P (N,2) from A to B (monotonic in x, decreasing y overall),
    build:
      - pos(t): returns (x,y) at time t by inverting t(u) along the path
      - total_time: final time to reach B
      - u_of_t: the parameter mapping (for debug/inspection)

    The parameterization uses u ∈ [0,1] mapped to the polyline with linear interpolation.
    """
    # Create a C1-ish parameterization r(u) by linear interpolation on P
    def r_of_u(u_in: np.ndarray) -> np.ndarray:
        u = np.clip(u_in, 0.0, 1.0)
        # fractional index along polyline
        s = u * (P.shape[0] - 1)
        i0 = np.floor(s).astype(int)
        i1 = np.clip(i0 + 1, 0, P.shape[0] - 1)
        w = (s - i0).reshape(-1, 1)
        return (1 - w) * P[i0] + w * P[i1]

    u_grid = np.linspace(0.0, 1.0, 2001)
    y0 = P[0, 1]
    _, _, t_cum, _ = arclength_and_time(r_of_u, u_grid, g=g, y0=y0, rolling_k=rolling_k)
    u_of_t = invert_time(t_cum, u_grid)

    def pos_at_time(t: float) -> np.ndarray:
        u = u_of_t(t)
        return r_of_u(np.array([u]))[0]

    return pos_at_time, float(t_cum[-1]), u_of_t


# ------------------------------
# Manim scene (uses the physics)
# ------------------------------

class BrachistochroneComparison(Scene):
    def construct(self):
        # Configuration: start (top-left) and end (bottom-right) in scene coords
        A = (-5.0, 2.5)
        B = (5.0, -1.0)

        # Build axes for context (not strictly required for physics)
        axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[-3.5, 3.5, 1],
            x_length=11.0,
            y_length=6.0,
            axis_config={"include_numbers": True, "stroke_width": 2},
            tips=False,
        )  # centered
        self.add(axes)

        # Visual markers for A and B
        A_dot = Dot(axes.c2p(*A), color=GREY_B)
        B_dot = Dot(axes.c2p(*B), color=GREY_B)
        self.add(A_dot, B_dot)

        # --- Define the three candidate paths (pure geometry) ---
        # 1) Straight line
        P_straight = make_straight(A, B, n_samples=PHYS_SAMPLES)

        # 2) "Steep dip" cubic Bézier: go down quickly then across
        #    Control points chosen to create a deep initial drop and longer path.
        ctrl1 = (A[0] + 0.5 * (B[0] - A[0]) * 0.3, A[1] - 3.0)
        ctrl2 = (A[0] + 0.7 * (B[0] - A[0]), B[1] - 0.2)
        P_steep = make_poly_bezier(A, B, ctrl1, ctrl2, n_samples=PHYS_SAMPLES)

        # 3) Cycloid (optimal for minimal time under gravity; rolling only rescales speed by constant factor)
        P_cycloid = make_cycloid_between(A, B, n_samples=PHYS_SAMPLES)

        # --- Prepare motion (physics) for each path ---
        g = 9.81
        k = 0.0  # frictionless bead on a wire
        pos_straight, T_straight, _ = prepare_motion_functions(P_straight, g=g, rolling_k=k)
        pos_steep, T_steep, _ = prepare_motion_functions(P_steep, g=g, rolling_k=k)
        pos_cycloid, T_cycloid, _ = prepare_motion_functions(P_cycloid, g=g, rolling_k=k)

        # Keep total window so that all paths finish within the same animation (1:1 time -> run_time = max T)
        
        # --- Downsample the motion to ANIM_STEPS steps for fast rendering ---
        def sample_motion(pos_func, T, steps):
            times = np.linspace(0.0, T, steps + 1)
            pts = np.array([pos_func(t) for t in times])
            return times, pts  # shapes: (steps+1,), (steps+1, 2)

        times_straight, pts_straight = sample_motion(pos_straight, T_straight, ANIM_STEPS)
        times_steep,    pts_steep    = sample_motion(pos_steep,    T_steep,    ANIM_STEPS)
        times_cycloid,  pts_cyc      = sample_motion(pos_cycloid,  T_cycloid,  ANIM_STEPS)

        # We animate for scaled duration so total frames are limited by frame_rate * run_time
        T_max = max(T_straight, T_steep, T_cycloid)
        RUN_TIME = T_max / max(TIME_SPEED, 1e-6)


        # --- Convert polylines to VMobjects for display ---
        def poly_to_vmobject(P, color):
            vm = VMobject(stroke_color=color, stroke_width=6)
            Pd = _downsample_poly(P, DISPLAY_SAMPLES)
            vm.set_points_as_corners([axes.c2p(*xy) for xy in Pd])
            return vm

        path_straight = poly_to_vmobject(P_straight, color=RED)
        path_steep = poly_to_vmobject(P_steep, color=BLUE)
        path_cycloid = poly_to_vmobject(P_cycloid, color=GREEN)

        # Add paths
        self.add(path_straight, path_steep, path_cycloid)

        # Labels for paths and times (to be updated after we compute numeric values)
        label_group = VGroup()
        label_straight = Tex(r"\textbf{Straight} (min distance)", color=RED).scale(0.7)
        label_steep = Tex(r"\textbf{Steep dip} (big initial accel.)", color=BLUE).scale(0.7)
        label_cycloid = Tex(r"\textbf{Cycloid} (min time)", color=GREEN).scale(0.7)

        # Position labels near the midpoints of the paths
        mid_idx = lambda P: len(P) // 2
        for label, P in [(label_straight, P_straight), (label_steep, P_steep), (label_cycloid, P_cycloid)]:
            idx = mid_idx(P)
            label.move_to(axes.c2p(*P[idx]) + np.array([0.0, 0.6, 0.0]))
            label_group.add(label)

        self.add(*label_group)

        # --- Dots with step-snapping updater ---
        t_tracker = ValueTracker(0.0)

        def make_step_dot(times_arr, pts_arr, color):
            dot = Dot(color=color, radius=0.08)
            def updater(mobj):
                t = t_tracker.get_value()
                # snap to nearest precomputed time index based on the path's own duration
                idx = int(round((len(times_arr) - 1) * t / max(T_max, 1e-12)))
                idx = max(0, min(len(times_arr) - 1, idx))
                x, y = pts_arr[min(idx, len(pts_arr)-1)]
                mobj.move_to(axes.c2p(x, y))
            dot.add_updater(updater)
            # Initialize at start
            x0, y0 = pts_arr[0]
            dot.move_to(axes.c2p(x0, y0))
            return dot

        dot_straight = make_step_dot(times_straight, pts_straight, RED)
        dot_steep    = make_step_dot(times_steep,    pts_steep,    BLUE)
        dot_cycloid  = make_step_dot(times_cycloid,  pts_cyc,      GREEN)

        self.add(dot_straight, dot_steep, dot_cycloid)

        # --- Display numeric arrival times and ranking ---
        times_text = VGroup(
            Tex(rf"$T_{{\text{{straight}}}} = {T_straight:.3f}\,\text{{s}}$", color=RED).scale(0.7),
            Tex(rf"$T_{{\text{{steep}}}} = {T_steep:.3f}\,\text{{s}}$", color=BLUE).scale(0.7),
            Tex(rf"$T_{{\text{{cycloid}}}} = {T_cycloid:.3f}\,\text{{s}}$", color=GREEN).scale(0.7),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(UL)

        # Legend line
        legend_line = Line(axes.c2p(A[0], A[1] + 0.3), axes.c2p(B[0], A[1] + 0.3), stroke_width=1.5, color=GREY_B)

        self.add(times_text, legend_line)

        # --- Animate: advance t_tracker linearly so run_time == physical time (1:1) ---
        # Start from t=0
        t_tracker.set_value(0.0)
        self.wait(0)

        # Play the motion
        
        if DISCRETE_MODE:
            # Emit exactly ANIM_STEPS+1 frames; advance physical time uniformly to T_max
            for i in range(ANIM_STEPS + 1):
                t_tracker.set_value(T_max * i / max(ANIM_STEPS, 1))
                self.wait(0)  # write a single frame
        else:
            self.play(t_tracker.animate.set_value(T_max), run_time=RUN_TIME, rate_func=rate_functions.linear)


        self.wait(0)

        # --- Final comparative statement ---
        order = sorted(
            [("Straight", T_straight), ("Steep", T_steep), ("Cycloid", T_cycloid)],
            key=lambda kv: kv[1],
        )
        result_text = Tex(
            rf"\textbf{{Fastest}}: {order[0][0]} \quad < \quad {order[1][0]} \quad < \quad {order[2][0]}",
        ).scale(0.9).to_corner(DR)
        self.add(result_text)
        self.wait(0)
