from manim import *
import numpy as np

# === Parameters ===
# Lengths of the rods
L1 = 2
L2 = 1.25
# Masses of the bobs
m1 = 1
m2 = 1
# Gravitational acceleration
g = 9.8
duration = 100

# === Initial Conditions ===
# Angles measured from vertical (in radians)
theta1_0 = PI / 3.001   # initial angle of the first pendulum
theta2_0 = PI / 2.5   # initial angle of the second pendulum
# Initial angular velocities
omega1_0 = 0.27
omega2_0 = 0.5

# Damping (friction) coefficients (in 1/s)
damping1 = 0.03   # Damping for the first pendulum
damping2 = 0.15   # Damping for the second pendulum

# Driving force parameters (applied to the first pendulum)
driving_amplitude = 0.1   # f in the driving force f*cos(omega_drive*t)
driving_frequency = 2.1   # omega_drive (radians per second)


class CustomTracedStateSpace(VMobject):
    def __init__(self, get_point, initial_pos = ORIGIN, threshold=0.5, **kwargs):
        """
        get_point : a function that returns the current point (as a np.array) in axes coordinates.
        threshold : maximum allowed distance between consecutive points.
        """
        super().__init__(**kwargs)
        self.get_point = get_point
        self.threshold = threshold
        self.last_point = None
        # Start with an empty set of points.
        self.set_points_as_corners([ initial_pos, initial_pos ])

    def update_trace(self, dt):
        new_point = self.get_point()
        # If we already have a point, check the distance:
        if self.last_point is not None:
            if np.linalg.norm(new_point - self.last_point) > self.threshold:
                # Start a new segment to avoid drawing a long line.
                self.start_new_path(new_point)
        self.add_line_to(new_point)
        self.last_point = new_point

    def updater_func(self, mob, dt):
        self.update_trace(dt)
        return mob

class DoublePendulumScene(Scene):
    def construct(self):


        """
        \mathcal{L} &=\frac{1}{2}(m_1 + m_2) l_1^2 \dot{\theta}_1^2 + \frac{1}{2}m_2 l_2^2 \dot{\theta}_2^2 + \\& m_2 l_1 l_2 \dot{\theta}_1\dot{\theta}_2 \cos(\theta_1 - \theta_2) + (m_1 + m_2) g l_1 \cos\theta_1 + m_2 g l_2\cos\theta_2 
        """

        L = MathTex(R"""
                \begin{aligned}
                    \mathcal{L} = &\frac{1}{2}(m_1 + m_2) l_1^2 \dot{\theta}_1^2 + \frac{1}{2}m_2 l_2^2 \dot{\theta}_2^2 +\\ & m_2 l_1 l_2 \dot{\theta}_1\dot{\theta}_2 \cos(\theta_1 - \theta_2) + (m_1 + m_2) g l_1 \cos\theta_1 + m_2 g l_2\cos\theta_2 
                \end{aligned}
                """,
                font_size = 30
                )
        L.to_corner(UL, buff=0.5)
        self.add(L)
        
        # The state vector: [theta1, theta2, omega1, omega2]
        self.state = np.array([theta1_0, theta2_0, omega1_0, omega2_0], dtype=float)

        # Set up a time variable so that the driving force can be time-dependent.
        self.t = 0.0

        # Fixed pivot point (at the origin)
        # self.graph_origin = 1 * RIGHT + 0.3 * DOWN
        pivot = [-3, 0, 0]

        # === Helper Function for Positions ===
        def get_positions(state):
            """
            Given the state [theta1, theta2, omega1, omega2],
            compute the positions of the first and second bobs.
            Both angles are measured (in radians) from the vertical.
            """
            theta1, theta2, _, _ = state
            # Position of the first bob
            bob1_pos = pivot + L1 * np.array([np.sin(theta1), -np.cos(theta1), 0])
            # Position of the second bob (attached to bob1)
            bob2_pos = bob1_pos + L2 * np.array([np.sin(theta2), -np.cos(theta2), 0])
            return bob1_pos, bob2_pos

        # Get the initial positions of the two bobs
        bob1_initial, bob2_initial = get_positions(self.state)

        ss_scale = 1
        ssv_scale = 1
        bob1_ss_initial = [theta1_0 * ss_scale, omega1_0 * ssv_scale, 0]
        bob2_ss_initial = [theta2_0 * ss_scale, omega2_0 * ssv_scale, 0]

        # === Create the Mobjects ===
        # First rod: from the pivot to the first bob
        rod1 = Line(pivot, bob1_initial, stroke_width=4)
        # Second rod: from the first bob to the second bob
        rod2 = Line(bob1_initial, bob2_initial, stroke_width=4)
        # Bobs
        bob1 = Circle(radius=0.12, color=RED, fill_opacity=0.75).move_to(bob1_initial)
        bob2 = Circle(radius=0.12, color=BLUE, fill_opacity=0.75).move_to(bob2_initial)
        # A dot at the pivot
        pivot_dot = Dot(point=pivot, color=WHITE)

        # Add the basic elements to the scene
        self.add(rod1, rod2, bob1, bob2, pivot_dot)

        # === Add a Traced Path for Bob1 ===
        trace1 = TracedPath(
            bob1.get_center,
            dissipating_time=2,  # show only the last 5 seconds of the trail
            stroke_color=RED,
            stroke_width=2,
            stroke_opacity=[0, 1]
        )
        self.add(trace1)

        # === Add a Traced Path for Bob2 ===
        trace = TracedPath(
            bob2.get_center,
            dissipating_time=15,  # show only the last 5 seconds of the trail
            stroke_color=YELLOW,
            stroke_width=2,
            stroke_opacity=[0, 1]
        )
        self.add(trace)

        # Axes 1: Upper Left
        axes1 = Axes(
            x_range=[-np.pi, np.pi, np.pi/2],
            y_range=[-10, 10, 4],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True, "include_ticks": True},
            x_axis_config={
                "include_numbers": False,
            },
        ).to_corner(UR, buff=0.5)
        
        values_x = [
            (-PI, r"$-\pi$"),
            (-PI / 2, r"$-\frac{\pi}{2}$"),
            (PI / 2, r"$\frac{\pi}{2}$"),
            (PI, r"$\pi$"),
        ]
        x_axis_labels = VGroup()  # Create a group named x_axis_labels
        #   pos.   tex.
        for x_val, x_tex in values_x:
            tex = Tex(x_tex)  # Convert string to tex
            tex.next_to(axes1.coords_to_point(x_val, 0), DOWN)  # Put tex on the position
            x_axis_labels.add(tex)  # Add tex in graph

        self.add(axes1)
        self.add(x_axis_labels)
        # self.wait()

        # Define a getter function that returns the current point in axes coordinates.
        def get_state_point_bob1():
            # Wrap theta1 into the interval [-π, π]
            wrapped_theta1 = ((self.state[0] + PI) % (2 * PI)) - PI
            # Use the axes' coordinate-to-point conversion.
            return axes1.c2p(wrapped_theta1, self.state[2])
        def get_state_point_bob2():
            # Wrap theta1 into the interval [-π, π]
            wrapped_theta1 = ((self.state[1] + PI) % (2 * PI)) - PI
            # Use the axes' coordinate-to-point conversion.
            return axes1.c2p(wrapped_theta1, self.state[3])
        # === Add a Traced Path for Bob1 State Space ===
        trace_ss2 = CustomTracedStateSpace(
            get_point=get_state_point_bob2,
            initial_pos=axes1.c2p(bob2_ss_initial[0], bob2_ss_initial[1]),
            # bob1_ss.get_center,
            threshold=1.5,
            # dissipating_time=60,  # show only the last 5 seconds of the trail
            stroke_color=BLUE,
            stroke_width=2,
        )
        self.add(trace_ss2)
        trace_ss2.add_updater(lambda m, dt: m.updater_func(m, dt))

        bob2_ss = Circle(radius=0.05, color=BLUE, fill_opacity=0.75).move_to( (bob2_ss_initial))
        # bob2_ss = Circle(radius=0.05, color=BLUE, fill_opacity=0.75).move_to(bob2_ss_initial)

        self.add(bob2_ss)


        # === Add a Traced Path for Bob1 State Space ===
        trace_ss1 = CustomTracedStateSpace(
            get_point=get_state_point_bob1,
            initial_pos=axes1.c2p(bob1_ss_initial[0], bob1_ss_initial[1]),
            # bob1_ss.get_center,
            threshold=1.5,
            # dissipating_time=60,  # show only the last 5 seconds of the trail
            stroke_color=RED,
            stroke_width=2,
        )
        self.add(trace_ss1)
        trace_ss1.add_updater(lambda m, dt: m.updater_func(m, dt))

        bob1_ss = Circle(radius=0.05, color=RED, fill_opacity=0.75).move_to( (bob1_ss_initial))
        # bob2_ss = Circle(radius=0.05, color=BLUE, fill_opacity=0.75).move_to(bob2_ss_initial)

        self.add(bob1_ss)


        # === Add a Driving Force Arrow ===
        # This arrow is anchored at the pivot. Its direction is set to be tangent
        # to the first pendulum’s circular path, i.e. in the (cos(theta1), sin(theta1))
        # direction. Its length is proportional to the instantaneous driving force.
        # You can adjust the scaling factor as needed.
        scaling_factor = 5  # Adjust to get a visually appealing arrow length
        driving_arrow = Arrow(
            start=pivot,
            end=pivot + scaling_factor * driving_amplitude * np.array([np.cos(theta1_0), np.sin(theta1_0), 0]),
            buff=0,
            color=GREEN,
            stroke_width=4,
        )

        def update_arrow(arrow):
            # Use the current value of theta1 and t
            theta1 = self.state[0]
            # Compute the instantaneous driving force (which may be negative)
            instantaneous_force = driving_amplitude * np.cos(driving_frequency * self.t)
            # The arrow vector points tangentially: (cos(theta1), sin(theta1), 0)
            arrow_vector = scaling_factor * instantaneous_force * np.array([np.cos(theta1), np.sin(theta1), 0])
            bob1_pos, _ = get_positions(self.state)
            # dont update if arrow vector is too small
            if np.linalg.norm(arrow_vector) < 0.1:
                arrow.set_opacity(0)
            else:
                arrow.set_opacity(1)
            arrow.put_start_and_end_on(pivot, pivot + arrow_vector)
            return arrow

        if driving_amplitude > 0:
            driving_arrow.add_updater(update_arrow)
            self.add(driving_arrow)

        # === Equations of Motion ===
        def derivatives(state):
            """
            Compute the derivatives [dtheta1/dt, dtheta2/dt, domega1/dt, domega2/dt]
            for the double pendulum.
            """
            theta1, theta2, omega1, omega2 = state
            delta = theta1 - theta2

            # Denominator for both accelerations
            denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)

            # Angular acceleration for the first pendulum
            domega1_u = (
                -g * (2 * m1 + m2) * np.sin(theta1)
                - m2 * g * np.sin(theta1 - 2 * theta2)
                - 2 * np.sin(delta) * m2 * (omega2**2 * L2 + omega1**2 * L1 * np.cos(delta))
            ) / (L1 * denom)

            # Angular acceleration for the second pendulum
            domega2_u = (
                2 * np.sin(delta) * (
                    omega1**2 * L1 * (m1 + m2)
                    + g * (m1 + m2) * np.cos(theta1)
                    + omega2**2 * L2 * m2 * np.cos(delta)
                )
            ) / (L2 * denom)

            # Apply damping terms (proportional to the angular velocities)
            # and add the driving force to the first pendulum.
            domega1 = domega1_u - damping1 * omega1 \
                      + driving_amplitude * np.cos(driving_frequency * self.t)
            domega2 = domega2_u - damping2 * omega2

            return np.array([omega1, omega2, domega1, domega2], dtype=float)

        # === Update the State via Euler Integration ===
        # We create a dummy mobject whose updater updates the state.
        def update_state(mob, dt):
            # Update the time variable
            self.t += dt
            # Compute derivatives and update the state vector.
            self.state += derivatives(self.state) * dt
            return mob

        updater_mobject = Mobject()
        updater_mobject.add_updater(update_state)
        self.add(updater_mobject)

        # === Updaters for the Graphical Elements ===
        # These updaters read the current state and reposition the rods and bobs.
        rod1.add_updater(lambda mob: mob.put_start_and_end_on(pivot, get_positions(self.state)[0]))
        rod2.add_updater(lambda mob: mob.put_start_and_end_on(get_positions(self.state)[0], get_positions(self.state)[1]))
        bob1.add_updater(lambda mob: mob.move_to(get_positions(self.state)[0]))
        bob2.add_updater(lambda mob: mob.move_to(get_positions(self.state)[1]))

        def one_loop(theta):
            # return theta
            return theta % 15 - 7.5
        def update_ss(mob, dt):
            bob2_ss.move_to(
                [get_state_point_bob2()]
                )
            bob1_ss.move_to(
                [get_state_point_bob1()]
                )
            #bob2_ss.move_to([one_loop(self.state[1]* ss_scale), self.state[3]* ssv_scale, 0])
            return mob
        bob2_ss.add_updater(update_ss)
        #bob2_ss.add_updater(update_ss)

        # === Run the Animation ===
        # Let the simulation run for 20 seconds.
        self.wait(duration)

        # (Optional) Remove the updaters if you plan to add further animations.
        updater_mobject.clear_updaters()
        rod1.clear_updaters()
        rod2.clear_updaters()
        bob1.clear_updaters()
        bob2.clear_updaters()


if __name__ == '__main__':
    from manim import config
    # Optionally, set configuration options if needed:
    config.pixel_width = 1280
    config.pixel_height = 720
    config.frame_rate = 60
    
    # Create and render the scene
    scene = DoublePendulumScene()
    scene.render()