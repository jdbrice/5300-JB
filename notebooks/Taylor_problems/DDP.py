import numpy as np
from scipy.integrate import solve_ivp
from manim import *

class Pendulum():
    """
    Pendulum class implements the parameters and differential equation for 
     a pendulum using the notation from Taylor.
     
    Parameters
    ----------
    omega_0 : float
        natural frequency of the pendulum (\sqrt{g/l} where l is the 
        pendulum length) 
    beta : float
        coefficient of friction 
    gamma_ext : float
        amplitude of external force is gamma * omega_0**2 
    omega_ext : float
        frequency of external force 
    phi_ext : float
        phase angle for external force 

    Methods
    -------
    dy_dt(y, t)
        Returns the right side of the differential equation in vector y, 
        given time t and the corresponding value of y.
    driving_force(t) 
        Returns the value of the external driving force at time t.
    """
    def __init__(self, omega_0=1., beta=0.2,
                 gamma_ext=0.2, omega_ext=0.689, phi_ext=0.
                ):
        self.omega_0 = omega_0
        self.beta = beta
        self.gamma_ext = gamma_ext
        self.omega_ext = omega_ext
        self.phi_ext = phi_ext
    
    def dy_dt(self, t, y):
        """
        This function returns the right-hand side of the diffeq: 
        [dphi/dt d^2phi/dt^2]
        
        Parameters
        ----------
        y : float
            A 2-component vector with y[0] = phi(t) and y[1] = dphi/dt
        t : float
            time 
            
        Returns
        -------
        
        """
        F_ext = self.driving_force(t)
        return [y[1], -self.omega_0**2 * np.sin(y[0]) - 2.*self.beta * y[1] \
                       + F_ext]
    
    def driving_force(self, t):
        """
        This function returns the value of the driving force at time t.
        """
        return self.gamma_ext * self.omega_0**2 \
                              * np.cos(self.omega_ext*t + self.phi_ext)  
    
    def solve_ode(self, y0, time, dt=0.01 ):
        print("Solving ODE")
        print("t span=", (0, time))
        t_eval = np.arange(0, time, dt)
        t_eval = np.append(t_eval, time) # add the last time point
        print("t_eval=", (t_eval[0], t_eval[-1]))
        solution = solve_ivp(self.dy_dt, 
                             t_span=(0, time), 
                             y0=y0,
                             method='RK23', 
                             t_eval=t_eval)
        return solution.y

class DDPChaosDemo(Scene):
    def construct(self):
        # Add latex of the damped driven pendulum
        ddp = MathTex(r"\ddot{\phi} + 2\beta \dot{\phi} + \omega_0^2 \sin(\phi) = \gamma \omega_0^2 \cos(\omega_{\text{ext}} t + \phi_{\text{ext}})")
        ddp.to_corner(UL)
        self.add(ddp)

        # Common pendulum parameters
        gamma_ext = 1.105
        omega_ext = 2.*np.pi
        phi_ext = 0.

        omega_0 = 1.5*omega_ext
        beta = omega_0/4.

        # Make Latex for the parameters
        omega_0_tex = MathTex(r"\omega_0 = 1.5 \omega_{\text{ext}}, ")
        beta_tex = MathTex(r"\beta = \frac{\omega_0}{4},")
        gamma_ext_tex = MathTex(r"\gamma = 1.105,")
        omega_ext_tex = MathTex(r"\omega_{\text{ext}} = 2 \pi")
        phi_ext_tex = MathTex(r"\phi_{\text{ext}} = 0")
        omega_0_tex.next_to(ddp, DOWN, buff=0.4).shift(3.5*LEFT)
        beta_tex.next_to(omega_0_tex, RIGHT, buff=0.2)
        gamma_ext_tex.next_to(beta_tex, RIGHT, buff=0.2)
        omega_ext_tex.next_to(gamma_ext_tex, RIGHT, buff=0.2)
        phi_ext_tex.next_to(omega_0_tex, DOWN, buff=0.2).shift( 0.5 * LEFT)
        self.add(omega_0_tex, beta_tex, gamma_ext_tex, omega_ext_tex, phi_ext_tex)

        p1 = Pendulum(omega_0=omega_0, beta=beta,
                           gamma_ext=gamma_ext, omega_ext=omega_ext, phi_ext=phi_ext)
        
        p2 = Pendulum(omega_0=omega_0, beta=beta,
                           gamma_ext=gamma_ext, omega_ext=omega_ext, phi_ext=phi_ext)
        
        # Set up the time points for the solution
        t_start = 0
        t_end = 90
        duration = t_end - t_start
        
        # set delta_t according to the frame rate
        delta_t = 1/config.frame_rate
        t_pts = np.arange(0, t_end, delta_t)
        t_pts = np.append(t_pts, t_end) # add the last time point
        print("t_pts=", (t_pts.shape))

        # Set up the initial conditions for the solution
        x_0 = -np.pi / 2.
        x_dot_0 = 0.
        points1 = p1.solve_ode([x_0, x_dot_0], t_end, delta_t)
        points2 = p2.solve_ode([x_0+np.pi/500, x_dot_0], t_end, delta_t)
        
        # Create the axes for state space representation
        min_x = np.min( points1[0] )
        max_x = np.max( points1[0] )
        max_xdot = np.max( points1[1] )
        min_xdot = np.min( points1[1] )
        axis_ss = Axes(
            x_range=[min_x, max_x, max_x/2],
            y_range=[min_xdot, max_xdot, max_xdot/2],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "include_ticks": True, "decimal_number_config": {"num_decimal_places": 1}, 'tip_shape': StealthTip},
        )
        axis_ss.to_corner(DR, buff=0.2)
        self.add(axis_ss)

        axis_pen = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "include_ticks": True, "decimal_number_config": {"num_decimal_places": 1}, 'tip_shape': StealthTip},
        )
        axis_pen.to_corner(DL, buff=0.2)
        # self.add(axis_pen)

        

        

        print("points1=", points1.shape)
        curve_points1 = np.column_stack((t_pts, points1[0], np.zeros_like(t_pts)))
        curve_points2 = np.column_stack((t_pts, points2[0], np.zeros_like(t_pts)))
        
    
        curve_ss1 = np.column_stack( ( (points1[0]), points1[1], np.zeros_like(points1[0])) )
        curve_ss2 = np.column_stack( (points2[0], points2[1], np.zeros_like(points2[0])) )

        # apply the axis_ss.c2p to the curve_ss points
        curve_ss1 = np.array([axis_ss.c2p(*point) for point in curve_ss1])
        curve_ss2 = np.array([axis_ss.c2p(*point) for point in curve_ss2])        


        # curve1 = VMobject().set_points_smoothly(curve_points1)
        # curve2 = VMobject().set_points_smoothly(curve_points2)
        # curve2.set_color(RED)

        vmcurve_ss1 = VMobject().set_points_smoothly(curve_ss1)
        vmcurve_ss1.set_color(BLUE_B)

        vmcurve_ss2 = VMobject().set_points_smoothly(curve_ss2)
        vmcurve_ss2.set_color(RED)

        L = 4
        dot1 = Dot(radius=0.2, color=RED)
        dot1.add_updater(lambda m: m.move_to(axis_pen.c2p( L * np.sin( axis_ss.p2c(vmcurve_ss1.get_end())[0] ), -L * np.cos( axis_ss.p2c(vmcurve_ss1.get_end())[0] ), 0)))
        self.add(dot1)

        dot1_trace = TracedPath(dot1.get_center, stroke_color=RED, stroke_width=3, dissipating_time=0.5, stroke_opacity=[0, 1] )
        self.add(dot1_trace)

        rod1 = Line(axis_pen.c2p(0,0,0), axis_pen.c2p(0, 0, 0), stroke_width=4)
        rod1.add_updater(lambda m: m.put_start_and_end_on(axis_pen.c2p(0,0,0), axis_pen.c2p( L * np.sin( axis_ss.p2c(vmcurve_ss1.get_end())[0] ), -L * np.cos( axis_ss.p2c(vmcurve_ss1.get_end())[0] ), 0)))
        self.add(rod1)

        dot2 = Dot(radius=0.2, color=BLUE)
        dot2.add_updater(lambda m: m.move_to(axis_pen.c2p( L * np.sin( axis_ss.p2c(vmcurve_ss2.get_end())[0] ), -L * np.cos( axis_ss.p2c(vmcurve_ss2.get_end())[0] ), 0)))
        self.add(dot2)

        dot2_trace = TracedPath(dot2.get_center, stroke_color=WHITE, stroke_width=3, dissipating_time=0.5, stroke_opacity=[0, 1] )
        self.add(dot2_trace)

        rod2 = Line(axis_pen.c2p(0,0,0), axis_pen.c2p(0, 0, 0), stroke_width=4)
        rod2.add_updater(lambda m: m.put_start_and_end_on(axis_pen.c2p(0,0,0), axis_pen.c2p( L * np.sin( axis_ss.p2c(vmcurve_ss2.get_end())[0] ), -L * np.cos( axis_ss.p2c(vmcurve_ss2.get_end())[0] ), 0)))
        self.add(rod2)

        

        # ss1_dot = Dot(radius=0.1, color=RED)
        # ss1_dot.move_to(vmcurve_ss1.points[0])
        # ss1_dot.add_updater(lambda m: m.move_to(vmcurve_ss1.get_end()))
        # self.add(ss1_dot)

        

        # axis_ss.c2p(0, 0, 0)

        self.play(
            *(
                # Create( curve1, rate_func=linear, duration=duration ),
                # Create( curve2, rate_func=linear, duration=duration ),
                Create( vmcurve_ss1, rate_func=linear, duration=duration ),
                Create( vmcurve_ss2, rate_func=linear, duration=duration ),
            ),
            run_time=duration
        )