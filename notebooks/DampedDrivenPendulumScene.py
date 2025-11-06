# manim -pql DampedDrivenPendulumScene.py DampedDrivenPendulumScene
# Requires: manim (community), numpy, scipy

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from manim import *

# -----------------------------
# Numerical model (separate)
# -----------------------------
@dataclass
class PendulumParams:
    # phi'' + 2*beta*phi' + omega0^2*sin(phi) = gamma*omega0^2*cos(omega_ext*t)
    omega0: float = 2.0 * np.pi * 1.5      # natural frequency (rad/s)
    beta: float = (2.0 * np.pi * 1.5) / 4  # damping coeff (s^-1)
    gamma: float = 1.06                    # driving amplitude (scaled) [rad/s^2] = gamma*omega0^2
    omega_ext: float = 2 * np.pi * 1.0     # driving frequency (rad/s)
    phi0: float = 0.0                      # initial angle (rad)
    omega_init: float = 0.0                # initial angular velocity
    Ti: float = 0.0                        # start time
    Tf: float = 3.0                        # end time
    delta_T: float = 0.01                  # step for sampling (s)
    first_periods_skip: int = 10           # skip initial periods in Poincare section
    figure_num: str = "12.5"               # for labeling output files, if any
    phi_axis_step: float = 2* np.pi         # for phase space plot
    phi_axis_limits: tuple = None  # for phase space plot

def simulate(params: PendulumParams):
    def rhs(t, y):
        phi, omega = y
        dphi = omega
        drive_force = params.gamma*(params.omega0**2)*np.cos(params.omega_ext*t)
        domega = -2*params.beta*omega - (params.omega0**2)*np.sin(phi) + drive_force
        return [dphi, domega]

    t_eval = np.arange(params.Ti, params.Tf, params.delta_T, dtype=float)
    sol = solve_ivp(
        rhs, (params.Ti, params.Tf),
        [params.phi0, params.omega_init],
        t_eval=t_eval, rtol=1e-8, atol=1e-10, method="RK45"
    )
    # add the drive force for plotting
    drive_force = params.gamma*(params.omega0**2)*np.cos(params.omega_ext*sol.t)
    return {"t": sol.t, "phi": sol.y[0], "omega": sol.y[1], "drive": drive_force}

# Choose parameters here

# 12.2
# PARAMS = PendulumParams( gamma = 0.2, Tf = 6.0)
# 12.3
# PARAMS = PendulumParams( gamma = 0.9, Tf = 6.0)
# 12.4
# PARAMS = PendulumParams( gamma = 1.06, TF = 30.0, first_periods_skip=10 )
# 12.5
# PARAMS = PendulumParams( gamma = 1.073, Tf = 30.0, first_periods_skip=21, figure_num="12.5" )
# 12.6
# PARAMS = PendulumParams( gamma = 1.077, Tf = 15.0, first_periods_skip=4, figure_num="12.6" )
# 12.7a : TODO in the future add two pendulums simultaneously with slightly different initial conditions
#  this is the same as 12.6 actually
# PARAMS = PendulumParams( gamma = 1.077, Tf = 15.0, phi0 = 0.0, first_periods_skip=4, figure_num="12.7a" )
# 12.7b 
# PARAMS = PendulumParams( gamma = 1.077, Tf = 15.0, phi0 = -np.pi/2.0, first_periods_skip=4, figure_num="12.7b" )
# 12.8 : several to show period doubling
# PARAMS = PendulumParams( gamma = 1.06, Tf = 40.0, phi0 = -np.pi/2.0, first_periods_skip=5, figure_num="12.8a", phi_axis_limits=(-3*np.pi, 3*np.pi) )
# PARAMS = PendulumParams( gamma = 1.078, Tf = 40.0, phi0 = -np.pi/2.0, first_periods_skip=12, figure_num="12.8b", phi_axis_limits=(-3*np.pi, 3*np.pi) )
# PARAMS = PendulumParams( gamma = 1.081, Tf = 40.0, phi0 = -np.pi/2.0, first_periods_skip=8, figure_num="12.8c", phi_axis_limits=(-3*np.pi, 3*np.pi) )
# PARAMS = PendulumParams( gamma = 1.0826, Tf = 40.0, phi0 = -np.pi/2.0, first_periods_skip=12, figure_num="12.8d", phi_axis_limits=(-3*np.pi, 3*np.pi) )
# 12.9 is not a DDP
# 12.10
PARAMS = PendulumParams( gamma = 1.105, Tf = 30.0, phi0 = -np.pi/2.0, first_periods_skip=3, figure_num="12.10", phi_axis_limits=(-2*np.pi, 2*np.pi) )
DATA = simulate(PARAMS)


from fractions import Fraction

def format_pi_fraction(val, denom=2):
    """
    Format a value (float) as a LaTeX multiple of π with denominator up to `denom`.
    Example: -4.71238898 -> r"$-frac{3}{2}pi$"
    """
    # Express as a rational multiple of pi
    multiple = val / np.pi
    frac = Fraction(multiple).limit_denominator(denom)  # denominator up to denom

    num, den = frac.numerator, frac.denominator

    if num == 0:
        return r"$0$"
    elif den == 1:  # integer multiple
        if num == 1:
            return r"$\pi$"
        elif num == -1:
            return r"$-\pi$"
        else:
            return rf"${num}\pi$"
    else:
        if num == 1:
            return rf"$\tfrac{{\pi}}{{{den}}}$"
        elif num == -1:
            return rf"$\tfrac{{-\pi}}{{{den}}}$"
        else:
            if ( num/den ) < 0:
                num = abs(num)
                return rf"$-\tfrac{{{num}}}{{{den}}}\pi$"
            else:
                return rf"$\tfrac{{{num}}}{{{den}}}\pi$"
    
def make_pi_labels(limits, step):
    start, stop = limits
    values = np.arange(start, stop, step)  # include endpoint
    print(values)
    labels = []
    for v in values:        
        labels.append((v, format_pi_fraction(v, denom=4)))
    labels.append( (0.0, "0") )
    return labels


# -----------------------------
# Manim visualization (uses pre-solved arrays)
# -----------------------------
class DampedDrivenPendulumScene(Scene):
    def construct(self):
        t_arr = DATA["t"]
        th_arr = DATA["phi"]
        om_arr = DATA["omega"]

        t0 = float(t_arr[0])
        T_total = float(t_arr[-1] - t0)
        playback_speed = 1.0
        run_time = T_total / playback_speed

        # Interpolants
        def interp(arr, t):
            return np.interp(t, t_arr, arr)

        get_phi = lambda t: interp(th_arr, t)
        get_omega = lambda t: interp(om_arr, t)
        current_t = lambda: t0 + t_tracker.get_value()

        # Title
        title = Text( f"Damped-Driven Pendulum : Taylor Fig. {PARAMS.figure_num}", font_size=24)
        initial_conditions = [
            rf"\phi_0 = {PARAMS.phi0:.3f}\ \mathrm{{rad}},",
            rf"\dot{{\phi}}_0 = {PARAMS.omega_init:.3f}\ \mathrm{{rad/s}}",
        ]
        initial_conditions_tex = VGroup(*[MathTex(s).scale(0.6) for s in initial_conditions]).arrange(RIGHT, aligned_edge=DOWN, buff=0.15)
        initial_conditions_bg = RoundedRectangle(
            corner_radius=0.15,
            width=initial_conditions_tex.width + 0.4,
            height=initial_conditions_tex.height + 0.3,
            stroke_color=GREY_B,
            stroke_width=1,
            fill_color=BLACK,
            fill_opacity=0.15,
        ).move_to(initial_conditions_tex)
        initial_conditions_group = VGroup(initial_conditions_bg, initial_conditions_tex)
        # subtitle = Text("Physical (left)  |  State Space (right)", font_size=24).next_to(title, DOWN, buff=0.2)
        title_group = VGroup(title, initial_conditions_group).arrange(DOWN, aligned_edge=LEFT, buff=0.06).to_edge(UP)

        # Parameter panel (top-left)
        param_items = [
            rf"\omega_0 = {PARAMS.omega0:.3f}\ \mathrm{{rad/s}}",
            rf"\beta = {PARAMS.beta:.3f}\ \mathrm{{s^{{-1}}}}",
            rf"\gamma = {PARAMS.gamma:.3f}",
            rf"\omega_d = {PARAMS.omega_ext:.3f}\ \mathrm{{rad/s}}",
        ]

        param_tex = VGroup(*[MathTex(s).scale(0.6) for s in param_items]).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        panel_bg = RoundedRectangle(
            corner_radius=0.15,
            width=param_tex.width + 0.4,
            height=param_tex.height + 0.3,
            stroke_color=GREY_B,
            stroke_width=1,
            fill_color=BLACK,
            fill_opacity=0.15,
        ).move_to(param_tex)

        params_panel = VGroup(panel_bg, param_tex).to_corner(UL).shift(RIGHT*0.15)

        


        # Time tracker
        t_tracker = ValueTracker(0.0)

        # ---------------- Physical representation (left, moved up) ----------------
        pivot = LEFT*3.5 + UP*0.75
        L_vis = 1.0

        def bob_pos_abs(t):
            th = get_phi(t)
            return pivot + L_vis * np.array([np.sin(th), -np.cos(th), 0.0])

        rod = always_redraw(lambda: Line(pivot, bob_pos_abs(current_t()), stroke_width=6))
        bob = always_redraw(lambda: Circle(radius=0.12, color=YELLOW, fill_opacity=0.75).move_to(bob_pos_abs(current_t())))
        trace_pendulum = TracedPath(
            lambda: bob.get_center(), 
            stroke_width=3, 
            stroke_opacity=0.75, 
            color=YELLOW_B,
            dissipating_time=1.0)

        # ---------------- Phase space (right): omega vs phi ----------------
        phi_max = 1.1*np.max(np.abs(th_arr)) #max(1.1*np.max(np.abs(th_arr)), 1.5)
        phi_min = 1.1*np.min(th_arr) #max(1.1*np.max(np.abs(th_arr)), 1.5)
        omega_max = max(1.1*np.max(np.abs(om_arr)), 1.5)

        axes = Axes(
            x_range=[phi_min, phi_max, max(phi_max/4, 0.5)],
            y_range=[-omega_max, omega_max, max(omega_max/4, 0.5)],
            x_length=6,
            y_length=4,
            tips=False,
        ).move_to(RIGHT*3.5)

        x_left = axes.x_axis.get_start()
        y_bottom = axes.y_axis.get_start()
        xlab = MathTex(r"\phi~[\mathrm{rad}]").scale(0.6).next_to(x_left, LEFT, buff=0.2)
        ylab = MathTex(r"\dot{\phi}~[\mathrm{rad/s}]").scale(0.6).next_to(y_bottom, DOWN, buff=0.2)

        phase_point = always_redraw(
            lambda: Circle(radius=0.06, color=TEAL, fill_opacity=0.75).move_to( axes.coords_to_point(get_phi(current_t()), get_omega(current_t())) )
        )

        
        # Phase space trace

        trace_phase = TracedPath(lambda: phase_point.get_center(), stroke_width=3, stroke_opacity=0.6, color=[TEAL_B, RED_B], dissipating_time=10.0)
        trace_phase_2 = TracedPath(lambda: phase_point.get_center(), stroke_width=2, stroke_opacity=0.3, color=RED_A)

        # ----------- Poincaré-like markers at multiples of drive period -----------
        Td = 2*np.pi / PARAMS.omega_ext
        SKIP_PERIODS = 0  # set >0 to ignore initial transients
        k_max = int(np.floor((t_arr[-1] - (t0 + SKIP_PERIODS*Td)) / Td))
        mark_times = [t0 + (SKIP_PERIODS + i + 1)*Td for i in range(max(0, k_max))]

        poincare_dots = VGroup()
        for tm in mark_times:
            # skip the first few if desired according to PARAMS.first_periods_skip
            if tm < t0 + PARAMS.first_periods_skip * Td:
                continue
            p = axes.coords_to_point(get_phi(tm), get_omega(tm))
            d = Dot(p, radius=0.05, color=RED).set_opacity(0.0)
            d.add_updater(lambda m, tt=tm: m.set_opacity(1.0 if current_t() >= tt else 0.0))
            poincare_dots.add(d)

        # ---------------- phi(t) plot below the pendulum ----------------
        y_step = np.pi / 2
        y_min_raw = 1.1 * phi_min
        y_max_raw = 1.1 * phi_max
        y_min = y_step * np.floor(y_min_raw / y_step)
        y_max = y_step * np.ceil(y_max_raw / y_step)
        if np.isclose(y_min, y_max):
            y_max = y_min + y_step

        axes_t = Axes(
            x_range=[PARAMS.Ti, PARAMS.Tf+1, 5],
            y_range=[y_min, y_max, y_step],
            x_length=6,
            y_length=2.2,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": True},
            y_axis_config={
                "include_numbers": False,
            },
        ).move_to(LEFT*3.1 + DOWN*2.0)

        tlabel = MathTex(r"t~[\mathrm{s}]").scale(0.6).next_to(axes_t.x_axis, RIGHT, buff=0.15)
        philabel = MathTex(r"\phi~[\mathrm{rad}]").scale(0.6).next_to(axes_t.y_axis, UP, buff=0.15)

        
        label_limits = (y_min, y_max)
        if PARAMS.phi_axis_limits is not None:
            label_limits = PARAMS.phi_axis_limits
        axis_values_labels = make_pi_labels( label_limits, PARAMS.phi_axis_step)
        y_axis_labels = VGroup()  # Create a group named x_axis_labels
        #   pos.   tex.
        for x_val, x_tex in axis_values_labels:
            tex = Tex(x_tex)  # Convert string to tex
            tex.font_size = 24
            tex.next_to(axes_t.coords_to_point(0, x_val), LEFT)  # Put tex on the position
            y_axis_labels.add(tex)  # Add tex in graph

        axes_t.add(y_axis_labels)

        phi_curve = VMobject(stroke_color=YELLOW, stroke_width=2)
        def update_phi_curve(m):
            t_now = current_t()
            mask = t_arr <= t_now
            if not np.any(mask):
                pts = []
            else:
                pts = [axes_t.coords_to_point(t, th) for t, th in zip(t_arr[mask], th_arr[mask])]
            if len(pts) >= 2:
                m.set_points_as_corners(pts)
            elif len(pts) == 1:
                m.set_points_as_corners([pts[0], pts[0]])
            return m
        phi_curve.add_updater(update_phi_curve)

        phi_dot = always_redraw(
            lambda: Dot(axes_t.coords_to_point(current_t(), get_phi(current_t())),
                        radius=0.06, color=YELLOW)
        )

        # draw and update the driving force curve if desired
        drive_curve = VMobject(stroke_color=RED, stroke_width=2, stroke_opacity=0.5)
        def update_drive_curve(m):
            # shrink the curve to fit on the axis (y_min, y_max)
            drive_shrink = (np.max(np.abs(DATA["drive"])) or 1.0)  # avoid divide-by-zero
            # data = (y_max - y_min) / 4.0  # scale to 1/4 of the vertical range
            data = (DATA["drive"] / drive_shrink) * ( 1.1*y_max_raw )
            # * ( y_max_raw - y_min_raw )

            t_now = current_t()
            mask = t_arr <= t_now
            if not np.any(mask):
                pts = []
            else:
                pts = [axes_t.coords_to_point(t, df) for t, df in zip(t_arr[mask], data[mask])]
            if len(pts) >= 2:
                m.set_points_as_corners(pts)
            elif len(pts) == 1:
                m.set_points_as_corners([pts[0], pts[0]])
            return m
        drive_curve.add_updater(update_drive_curve)
        self.add(drive_curve)



        # Driving-force arrow at the bob (tangential to the arc)
        A_max = abs(PARAMS.gamma * (PARAMS.omega0**2)) or 1.0   # avoid divide-by-zero
        drive_scale = 1.0  # visual units for |drive| = A_max (tune to taste)

        def tangent_hat(th):  # unit vector for increasing theta (perpendicular to rod)
            return np.array([np.cos(th), np.sin(th), 0.0])

        def drive_vector(t):
            th = get_phi(t)
            val = PARAMS.gamma * (PARAMS.omega0**2) * np.cos(PARAMS.omega_ext * t)  # angular accel contribution
            length = drive_scale * (abs(val) / A_max)
            direction = np.sign(val) or 1.0
            return direction * length * tangent_hat(th)

        def drive_color(val):
            # map negative→BLUE, zero→GREY_B, positive→RED
            alpha = 0.5 * (val / A_max + 1.0)
            return interpolate_color(BLUE, RED, alpha)

        drive_arrow = always_redraw(
            lambda: Arrow(
                start=bob_pos_abs(current_t()),
                end=bob_pos_abs(current_t()) + drive_vector(current_t()),
                buff=0.0,
                max_tip_length_to_length_ratio=0.18,
                stroke_width=6,
                color=drive_color(PARAMS.gamma * (PARAMS.omega0**2) * np.cos(PARAMS.omega_ext * current_t())),
            )
        )

        # ---------------- Stable time display ----------------
        time_num = DecimalNumber(0, num_decimal_places=2).scale(0.6).set_color(GREY_B)
        time_num.add_updater(lambda m: m.set_value(current_t()))
        time_s = always_redraw(
            lambda: Text("s", font_size=24).next_to(time_num, RIGHT, buff=0.1)
        )
        time_label = VGroup(Text("t =", font_size=24), time_num, time_s).arrange(RIGHT, buff=0.08)
        time_label.to_corner(UR).shift(LEFT*0.2 + DOWN*0.2)

        # Build scene
        self.add(title_group)
        self.add(params_panel)
        self.add(rod, bob, trace_pendulum)
        self.add(axes, xlab, ylab, phase_point, trace_phase, trace_phase_2, poincare_dots)
        self.add(axes_t, tlabel, philabel, phi_curve, phi_dot)
        self.add(time_label)
        self.add(drive_arrow)

        self.wait(0.3)
        

        self.play(t_tracker.animate.set_value(T_total), run_time=run_time, rate_func=linear)

class Figure12_2(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 0.2, Tf = 6.0, phi0 = 0.0, first_periods_skip=3, figure_num="12.2", phi_axis_limits=(-1*np.pi, 1*np.pi), phi_axis_step=np.pi/2 )
        DATA = simulate(PARAMS)
        super().construct()

class Figure12_3(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 0.9, Tf = 6.0, phi0 = 0.0, first_periods_skip=4, figure_num="12.3", phi_axis_limits=(-1*np.pi, 1*np.pi), phi_axis_step=np.pi/2 )
        DATA = simulate(PARAMS)
        super().construct()

class Figure12_4(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.06, Tf = 30.0, phi0 = 0.0, first_periods_skip=10, figure_num="12.4", phi_axis_limits=(-1*np.pi, 5*np.pi), phi_axis_step=2*np.pi )
        DATA = simulate(PARAMS)
        super().construct()

# PARAMS = PendulumParams( gamma = 1.073, Tf = 30.0, first_periods_skip=21, figure_num="12.5" )
class Figure12_5(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.073, Tf = 30.0, phi0 = 0.0, first_periods_skip=21, figure_num="12.5", phi_axis_step=2*np.pi )
        DATA = simulate(PARAMS)
        super().construct()

# PARAMS = PendulumParams( gamma = 1.077, Tf = 15.0, first_periods_skip=4, figure_num="12.6" )
class Figure12_6(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.077, Tf = 30.0, phi0 = 0.0, first_periods_skip=4, figure_num="12.6", phi_axis_step=2*np.pi )
        DATA = simulate(PARAMS)
        super().construct()

# PARAMS = PendulumParams( gamma = 1.077, Tf = 15.0, phi0 = -np.pi/2.0, first_periods_skip=4, figure_num="12.7b" )
class Figure12_7(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.077, Tf = 30.0, phi0 = -np.pi/2.0, first_periods_skip=4, figure_num="12.7", phi_axis_step=2*np.pi )
        DATA = simulate(PARAMS)
        super().construct()

class Figure12_8a(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.06, Tf = 40.0, phi0 = -np.pi/2.0, first_periods_skip=5, figure_num="12.8a", phi_axis_limits=(-3*np.pi, 3*np.pi) )
        DATA = simulate(PARAMS)
        super().construct()

class Figure12_8b(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.078, Tf = 40.0, phi0 = -np.pi/2.0, first_periods_skip=5, figure_num="12.8b", phi_axis_limits=(-3*np.pi, 3*np.pi) )
        DATA = simulate(PARAMS)
        super().construct()

class Figure12_8c(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.081, Tf = 40.0, phi0 = -np.pi/2.0, first_periods_skip=5, figure_num="12.8c", phi_axis_limits=(-3*np.pi, 3*np.pi) )
        DATA = simulate(PARAMS)
        super().construct()

class Figure12_8d(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.0826, Tf = 40.0, phi0 = -np.pi/2.0, first_periods_skip=5, figure_num="12.8d", phi_axis_limits=(-3*np.pi, 3*np.pi) )
        DATA = simulate(PARAMS)
        super().construct()

class Figure12_10(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.105, Tf = 30.0, phi0 = -np.pi/2.0, first_periods_skip=3, figure_num="12.10", phi_axis_limits=(-2*np.pi, 2*np.pi) )
        DATA = simulate(PARAMS)
        super().construct()

class Figure12_10Extended(DampedDrivenPendulumScene):
    def construct(self):
        global PARAMS, DATA
        PARAMS = PendulumParams( gamma = 1.105, Tf = 300.0, phi0 = -np.pi/2.0, first_periods_skip=0, figure_num="12.10", phi_axis_limits=(-2*np.pi, 2*np.pi) )
        DATA = simulate(PARAMS)
        super().construct()
