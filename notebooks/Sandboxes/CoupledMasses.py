# coupled_masses_manim.py
#
# Requirements:
#   - manim (manim-community)
#   - numpy, scipy
#
# Render (preview, high quality):
#   manim -pqh coupled_masses_manim.py CoupledMassesScene
#
# Render (production):
#   manim -p -qk coupled_masses_manim.py CoupledMassesScene

from manim import *
import numpy as np
from scipy.integrate import solve_ivp

# -------------------------------
# Physics / Numerical solution
# -------------------------------
def coupled_ode(t, y, m1, m2, k1, k2, k3, c1=0.0, c2=0.0):
    """
    y = [x1, v1, x2, v2] are displacements from equilibrium.
    Equations (displacements about equilibrium):
        m1 x1'' = -k1 x1 - k2 (x1 - x2) - c1 v1
        m2 x2'' = -k3 x2 - k2 (x2 - x1) - c2 v2
    """
    x1, v1, x2, v2 = y
    a1 = (-k1*x1 - k2*(x1 - x2) - c1*v1) / m1
    a2 = (-k3*x2 - k2*(x2 - x1) - c2*v2) / m2
    return [v1, a1, v2, a2]


def simulate_system(
    m1=1.0, m2=1.0, k1=2.0, k2=4.0, k3=3.0,
    c1=0.0, c2=0.0,
    T=10.0, fps_sample=240,
    y0=(0.3, 0.0, -0.1, 0.0)
):
    """
    Run a numerical solution y(t) = [x1, v1, x2, v2] for t in [0, T].
    Returns (t_grid, x1_grid, x2_grid) sampled at uniform grid for easy interpolation.
    """
    t_eval = np.linspace(0.0, T, int(T*fps_sample)+1)
    sol = solve_ivp(
        coupled_ode, (0.0, T), y0,
        t_eval=t_eval, args=(m1, m2, k1, k2, k3, c1, c2),
        rtol=1e-9, atol=1e-12, method="DOP853"
    )
    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    x1 = sol.y[0]
    x2 = sol.y[2]
    return sol.t, x1, x2


# -------------------------------
# Simple spring helper (zig-zag)
# -------------------------------
def spring_polyline(start, end, coils=6, amplitude=0.25, inset=0.35):
    """
    Returns a VMobject shaped like a planar coil spring from start -> end.
    Uses set_points_as_corners for a crisp zig-zag. Compatible with manim v0.19.
    """
    start = np.array(start, dtype=float)
    end   = np.array(end, dtype=float)
    vec = end - start
    L = np.linalg.norm(vec)
    if L < 1e-6:
        return Line(start, end, stroke_width=6)

    # Local frame
    xhat = vec / L
    up = np.array([0.0, 1.0, 0.0])
    yhat = up - np.dot(up, xhat) * xhat
    ny = np.linalg.norm(yhat)
    if ny < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
        yhat = right - np.dot(right, xhat) * xhat
        yhat /= np.linalg.norm(yhat)
    else:
        yhat /= ny

    # Straight end segments + zig-zag body
    Lz = max(L - 2 * inset, 0.0)
    n_verts = 2 * coils + 1
    xs = np.linspace(inset, inset + Lz, n_verts)

    ys = np.zeros_like(xs)
    ys[1::2] =  amplitude
    ys[2::2] = -amplitude
    # Ensure the last zig-zag point is on the center line
    if n_verts > 0:
        ys[-1] = 0

    pts = [start, start + xhat * inset]
    for xi, yi in zip(xs, ys):
        pts.append(start + xhat * xi + yhat * yi)
    pts += [end - xhat * inset, end]

    pts = np.array(pts, dtype=float)

    spring = VMobject()
    spring.set_points_as_corners(pts)
    spring.set_stroke(width=6)
    spring.set_fill(opacity=0)
    return spring

# make these "global" so that we can define various scenes with different initial conditions
x_0 = [0.4, 0.0]
x_dot_0 = [0.0, 0.0]
M1 = 1.0
M2 = 1.0
K1 = 10.0
K2 = 1.0
K3 = 10.0
FIG = 11.2

# -------------------------------
# Manim Scene
# -------------------------------
class CoupledMassesScene(Scene):
    def construct(self):
        # ---- Parameters ----
        # Physics
        m1, m2 = M1, M2
        k1, k2, k3 = K1, K2, K3
        c1, c2 = 0.00, 0.00     # light damping for nice visuals (set to 0 for ideal)
        T_total = 30.0          # seconds of simulated (and animated) time
        fps_sample = 240    # samples per second for ODE solution and interpolation

        # Initial conditions: (x1, v1, x2, v2)
        y0 = (x_0[0], x_dot_0[0], x_0[1], x_dot_0[1])

        # Create the main title
        title = Text(f"Coupled Oscillators: Taylor Fig. {FIG:0.1f}", font_size=38).to_edge(UP)

        # Create the subtitle with spring constants
        subtitle = Text(
            f"k₁={k1}, k₂={k2}, k₃={k3}",
            font_size=32,
            color=WHITE
        ).next_to(title, DOWN)

        subtitle2 = Text(
            f"m₁={m1}, m₂={m2}",
            font_size=32,
            color=WHITE
        ).next_to(subtitle, DOWN)

        self.add(title, subtitle, subtitle2)

        # Solve once up-front
        t_grid, x1_grid, x2_grid = simulate_system(
            m1, m2, k1, k2, k3, c1, c2, T_total, fps_sample, y0
        )

        # Interpolators for animation time -> displacement
        def x1_of(t):
            return np.interp(t, t_grid, x1_grid)

        def x2_of(t):
            return np.interp(t, t_grid, x2_grid)

        # ---- Scene layout ----
        # Shift the spring-mass system down to make room for the titles
        system_shift = UP * 0.5
        
        # Horizontal rail (just a guide)
        rail = Line(LEFT*6, RIGHT*6, stroke_opacity=0.25).shift(system_shift)

        # "Walls"
        left_wall  = Rectangle(width=0.25, height=3.5).move_to(LEFT*6 + system_shift)
        right_wall = Rectangle(width=0.25, height=3.5).move_to(RIGHT*6 + system_shift)
        left_wall.set_fill(GREY, opacity=1.0).set_stroke(width=0)
        right_wall.set_fill(GREY, opacity=1.0).set_stroke(width=0)

        # Mass blocks (centered vertically, move along x only)
        mass_w, mass_h = 1.0, 0.6
        mass1 = Rectangle(width=mass_w, height=mass_h, color=BLUE, fill_opacity=1.0)
        mass2 = Rectangle(width=mass_w, height=mass_h, color=GREEN, fill_opacity=1.0)

        # Equilibrium positions for the two masses in scene coords
        # (These are *centers* where x1=x2=0.)
        x1_eq = -3.0
        x2_eq = +3.0

        # Scale factor to map physical displacement (in "meters") to scene units
        scale = 2.0  # feel free to adjust

        # Initialize positions at t=0
        mass1.move_to([x1_eq + scale*x1_of(0.0), 0.0, 0.0] + system_shift)
        mass2.move_to([x2_eq + scale*x2_of(0.0), 0.0, 0.0] + system_shift)

        # Labels (optional)
        label1 = Text("m₁", font_size=28).move_to(mass1.get_center())
        label2 = Text("m₂", font_size=28).move_to(mass2.get_center())

        # Add static elements
        self.add(rail, left_wall, right_wall, mass1, mass2, label1, label2)

        # Time tracker (drives the animation 1:1 with real time)
        t_tracker = ValueTracker(0.0)

        # ---- Dynamic Axes and Traces ----
        # This VGroup will hold the axes and its labels. We need a stable reference
        # for the trace updaters, so we'll update this group's contents.
        axes_group = VGroup()
        
        # We store the axes object itself as an attribute of the VGroup
        # so that the trace updaters can access its coordinate conversion methods.
        def axes_updater(mob):
            t_now = t_tracker.get_value()
            # Determine the x-axis range, growing in steps of 10
            t_max_axis = max(10, np.ceil(t_now / 10) * 10)
            # t_max_axis = t_now
            # if (t_max_axis < 10):
            #     t_max_axis = 10

            new_axes = Axes(
                x_range=[0, t_max_axis, t_max_axis / 5],
                y_range=[-0.6, 0.6, 0.2],
                x_length=10,
                y_length=2.5,
                axis_config={"include_numbers": True, "font_size": 24},
                tips=False,
            ).shift(DOWN * 2.2)
            
            new_labels = new_axes.get_axis_labels(x_label="t", y_label="x")
            
            # Store the new axes object on the container
            mob.axes = new_axes
            
            # Use .become() to transform the old axes group into the new one
            mob.become(VGroup(new_axes, new_labels))

        axes_group.add_updater(axes_updater)
        axes_updater(axes_group) # Initial call to create the first axes
        self.add(axes_group)

        # Updaters for the masses (position vs. t)
        def mass1_updater(mob):
            t = t_tracker.get_value()
            x = x1_eq + scale*x1_of(t)
            mob.move_to([x, 0.0, 0.0] + system_shift)

        def mass2_updater(mob):
            t = t_tracker.get_value()
            x = x2_eq + scale*x2_of(t)
            mob.move_to([x, 0.0, 0.0] + system_shift)

        mass1.add_updater(mass1_updater)
        mass2.add_updater(mass2_updater)

        label1.add_updater(lambda m: m.move_to(mass1.get_center()))
        label2.add_updater(lambda m: m.move_to(mass2.get_center()))

        # Springs: left (wall <-> m1), middle (m1 <-> m2), right (m2 <-> wall)
        # Use always_redraw so geometry refreshes as the masses move.
        spring_left = always_redraw(
            lambda: spring_polyline(
                left_wall.get_right(), mass1.get_left(), coils=6, amplitude=0.25, inset=0.25
            ).set_color(WHITE)
        )
        spring_middle = always_redraw(
            lambda: spring_polyline(
                mass1.get_right(), mass2.get_left(), coils=12, amplitude=0.25, inset=0.25
            ).set_color(WHITE)
        )
        spring_right = always_redraw(
            lambda: spring_polyline(
                mass2.get_right(), right_wall.get_left(), coils=6, amplitude=0.25, inset=0.25
            ).set_color(WHITE)
        )

        self.add(spring_left, spring_middle, spring_right)

        # ---- Create the time-series plot traces ----
        # VMobjects that will hold the traced curves
        trace1 = VMobject(color=BLUE, stroke_width=3)
        trace2 = VMobject(color=GREEN, stroke_width=3)
        
        # Start with a single point at t=0
        trace1.set_points_as_corners([axes_group.axes.c2p(0, x1_of(0))])
        trace2.set_points_as_corners([axes_group.axes.c2p(0, x2_of(0))])
        
        self.add(trace1, trace2)
        
        # Updater to extend the traces as time progresses
        def trace_updater(mob, x_func):
            t = t_tracker.get_value()
            current_axes = axes_group.axes
            if t > 0:
                # Create points from t=0 to current t
                num_points = max(2, int(t * fps_sample / 10))  # sample for smoothness
                t_vals = np.linspace(0, t, num_points)
                x_vals = [x_func(ti) for ti in t_vals]
                points = [current_axes.c2p(ti, xi) for ti, xi in zip(t_vals, x_vals)]
                mob.set_points_as_corners(points)
        
        trace1.add_updater(lambda mob: trace_updater(mob, x1_of))
        trace2.add_updater(lambda mob: trace_updater(mob, x2_of))

        # Optional: show a running time readout
        time_readout = DecimalNumber(
            number=0.0, num_decimal_places=2, include_sign=False
        ).set_font_size(28).to_corner(UR).shift(LEFT*1.1 + DOWN*1.5)
        time_label = Text("t (s) =", font_size=28).next_to(time_readout, LEFT, buff=0.2)

        def time_updater(mob):
            mob.set_value(t_tracker.get_value())

        time_readout.add_updater(time_updater)
        self.add(time_label, time_readout)

        # Animate: advance the tracker from 0 -> T_total in real time (rate_func=linear)
        self.play(t_tracker.animate.set_value(T_total), run_time=T_total, rate_func=linear)

        # Hold last frame briefly
        self.wait(0.5)

        # Clean up updaters (optional)
        mass1.clear_updaters()
        mass2.clear_updaters()
        time_readout.clear_updaters()
        trace1.clear_updaters()
        trace2.clear_updaters()


# -------------------------------
# Notes on the physics
# -------------------------------
# For small oscillations and taking x1, x2 as displacements from equilibrium,
# the undamped equations are:
#   m1 x1'' = -k1 x1 - k2 (x1 - x2)
#   m2 x2'' = -k3 x2 - k2 (x2 - x1)
# which is a standard two-degree-of-freedom linear system.
# The script optionally includes small viscous damping (c1, c2) for nicer visuals.
#
# References (classic treatments):
# - H. Goldstein, C. Poole, J. Safko, "Classical Mechanics", 3rd ed., Pearson.
# - L. Meirovitch, "Elements of Vibration Analysis", McGraw-Hill.

class FirstNormalMode(CoupledMassesScene):
    def __init__(self, *args, **kwargs):
        global x_0, x_dot_0, K1, K2, K3, FIG
        K1, K2, K3 = 5.0, 5.0, 5.0
        FIG = 11.2
        x_0 = [0.4, 0.4]  # first normal mode, both move together
        x_dot_0 = [0.0, 0.0]  # starting from rest
        super().__init__(*args, **kwargs)

class SecondNormalMode(CoupledMassesScene):
    def __init__(self, *args, **kwargs):
        global x_0, x_dot_0, K1, K2, K3, FIG
        K1, K2, K3 = 1.0, 1.0, 1.0
        FIG = 11.4
        x_0 = [0.4, -0.4]  # second normal mode, opposite directions
        x_dot_0 = [0.0, 0.0]  # starting from rest
        super().__init__(*args, **kwargs)

class GeneralMotion(CoupledMassesScene):
    def __init__(self, *args, **kwargs):
        global x_0, x_dot_0, K1, K2, K3, FIG
        K1, K2, K3 = 1.0, 1.0, 1.0
        FIG = 11.6
        x_0 = [0.3, 0.2]  # second normal mode, opposite directions
        x_dot_0 = [0.75, -0.5]  
        super().__init__(*args, **kwargs)

class WeakCoupling(CoupledMassesScene):
    def __init__(self, *args, **kwargs):
        global x_0, x_dot_0, K1, K2, K3, FIG
        K1, K2, K3 = 10.0, 1.0, 10.0
        FIG = 11.8
        x_0 = [0.4, 0.]  # only mass 1 displaced initially to make beats visible
        x_dot_0 = [0.0, 0.0]  # starting from rest
        super().__init__(*args, **kwargs)