import numpy as np

class FourWheelKinematic:
    def __init__(self, wheelbase=0.33, track_width=0.2, max_steer=np.radians(45)):
        self.L = wheelbase
        self.W = track_width
        self.max_steer = max_steer # Capped at 45 degrees
        self.x, self.y, self.theta = 0.0, 0.0, 0.0

    def calculate_commands(self, v_desired, steering_angle):
        """
        Returns the required angle and speed for each wheel 
        to satisfy Ackermann geometry.
        """
        # 1. HARD CONSTRAINT: Mechanical Steering Stops
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        
        if abs(steering_angle) < 0.001:
            return {
                "angles": {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0},
                "speeds": {"fl": v_desired, "fr": v_desired, "rl": v_desired, "rr": v_desired}
            }
            
        R = self.L / np.tan(steering_angle)
        
        # Ackermann Angles
        if steering_angle > 0: # Left Turn
            angle_fl = np.arctan(self.L / (R - self.W/2))
            angle_fr = np.arctan(self.L / (R + self.W/2))
        else: # Right Turn
            R_abs = abs(R)
            angle_fl = -np.arctan(self.L / (R_abs + self.W/2))
            angle_fr = -np.arctan(self.L / (R_abs - self.W/2))

        # Check: Even if the input steering is valid, the inner wheel might exceed limits 
        # because the inner wheel always turns MORE than the steering angle.
        # We clamp the output angles just to be safe visually.
        angle_fl = np.clip(angle_fl, -self.max_steer, self.max_steer)
        angle_fr = np.clip(angle_fr, -self.max_steer, self.max_steer)

        # Wheel Speeds (Electronic Differential)
        # v = omega * r
        omega_turn = v_desired / R # angular velocity of the turn
        
        r_rl = abs(R - self.W/2)
        r_rr = abs(R + self.W/2)
        r_fl = np.hypot(self.L, r_rl)
        r_fr = np.hypot(self.L, r_rr)
        
        # Ensure correct direction if reversing
        direction = np.sign(v_desired)
        
        return {
            "angles": {"fl": angle_fl, "fr": angle_fr, "rl": 0.0, "rr": 0.0},
            "speeds": {
                "fl": abs(omega_turn * r_fl) * direction,
                "fr": abs(omega_turn * r_fr) * direction,
                "rl": abs(omega_turn * r_rl) * direction,
                "rr": abs(omega_turn * r_rr) * direction
            }
        }
        
    def update_state(self, v, steering, dt):
        """Simple bicycle model integration for position tracking."""
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += (v / self.L) * np.tan(steering) * dt
        return self.x, self.y, self.theta


class DynamicRobot:
    """
    Euler-Lagrange Physics model (Bicycle Model).
    - Front Wheel Drive (FWD) Logic implemented.
    - Includes helper to map Bicycle State -> 4 Wheel Visuals.
    - Accounts for Mass, Inertia, and Tire Friction.
    - Simulates Drifting, Drag, and Actuator Limits.
    """
    def __init__(self, wheelbase=0.33, track_width=0.2, mass=5.0, I_z=0.5, C_alpha=50.0, max_steer=np.radians(45), max_force=20.0):
        self.L = wheelbase
        self.W = track_width 
        self.m = mass
        self.Iz = I_z
        self.Cf = C_alpha 
        self.Cr = C_alpha
        self.drag = 1.0
        self.max_steer = max_steer # Capped at 45 degrees
        self.max_force = max_force # Max motor force in Newtons
        
        self.lf = wheelbase / 2.0
        self.lr = wheelbase / 2.0
        
        # State Vector: [x, y, theta, vx, vy, omega]
        self.state = np.zeros(6)

    def update(self, throttle_force, steering_angle, dt):
        x, y, theta, vx, vy, omega = self.state
        
        # 1. HARD CONSTRAINTS: Steering AND Force
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        throttle_force = np.clip(throttle_force, -self.max_force, self.max_force)

        # 2. Calculate Slip Angles
        # (Where the tire is pointing) - (Where the tire is moving)
        if abs(vx) > 0.1:
            alpha_f = steering_angle - np.arctan2((vy + self.lf * omega), vx)
            alpha_r = -np.arctan2((vy - self.lr * omega), vx)
        else:
            alpha_f, alpha_r = 0.0, 0.0

        # 3. Calculate Lateral Tire Forces (Linear Tire Model)
        # Note: In a real implementation, we would saturate these at mu*N
        Fy_f = self.Cf * alpha_f
        Fy_r = self.Cr * alpha_r
        
        # 4. Solve Dynamics (Front Wheel Drive)
        # For FWD, the throttle force is applied at the steering angle
        Fx_fwd = throttle_force * np.cos(steering_angle) 
        Fy_fwd = throttle_force * np.sin(steering_angle) # Adds to lateral force!

        F_drag = self.drag * vx
        
        # Longitudinal Accel (Body Frame)
        # ax = (Sum Forces X + Centrifugal Force) / Mass
        ax = (Fx_fwd - F_drag - Fy_f * np.sin(steering_angle) + self.m * vy * omega) / self.m
        
        # Lateral Accel (Body Frame)
        # ay = (Sum Forces Y - Centrifugal Force) / Mass
        # Note: Fy_f acts perpendicular to the WHEEL, so we project it to the BODY
        ay = (Fy_f * np.cos(steering_angle) + Fy_r + Fy_fwd - self.m * vx * omega) / self.m
        
        # Rotational Accel
        # Torque = Force * Distance
        # Front lateral force provides torque, FWD pull provides torque if steered
        tau_f = (Fy_f * np.cos(steering_angle) + Fy_fwd) * self.lf
        tau_r = -Fy_r * self.lr
        alpha_z = (tau_f + tau_r) / self.Iz
        
        # 5. Integrate (Euler)
        vx += ax * dt
        vy += ay * dt
        omega += alpha_z * dt
        
        # Map to Global Coordinates
        v_global_x = vx * np.cos(theta) - vy * np.sin(theta)
        v_global_y = vx * np.sin(theta) + vy * np.cos(theta)
        
        x += v_global_x * dt
        y += v_global_y * dt
        theta += omega * dt
        
        self.state = np.array([x, y, theta, vx, vy, omega])
        
        return self.state

    def get_4_wheel_visuals(self, steering_angle):
        """
        Reverse-calculates the 4-wheel state from the bicycle model state
        so the visualizer can draw the robot correctly.
        """
        vx, vy, omega = self.state[3], self.state[4], self.state[5]
        
        # CONSTRAINT APPLIED HERE AS WELL
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        
        # Use the kinematic Ackermann math to get wheel ANGLES
        # (This assumes the wheels track the steering command instantly)
        if abs(steering_angle) < 0.001:
            fl_angle, fr_angle = 0.0, 0.0
            omega_turn = 0
            R = float('inf')
        else:
            R = self.L / np.tan(steering_angle)
            omega_turn = np.hypot(vx, vy) / R
            
            if steering_angle > 0:
                fl_angle = np.arctan(self.L / (R - self.W/2))
                fr_angle = np.arctan(self.L / (R + self.W/2))
            else:
                R_abs = abs(R)
                fl_angle = -np.arctan(self.L / (R_abs + self.W/2))
                fr_angle = -np.arctan(self.L / (R_abs - self.W/2))
        
        # Visual clamp to match mechanical limits
        fl_angle = np.clip(fl_angle, -self.max_steer, self.max_steer)
        fr_angle = np.clip(fr_angle, -self.max_steer, self.max_steer)

        # Wheel Speeds (Approximation for visuals)
        if R != float('inf') and R != 0:
             r_rl = abs(R - self.W/2)
             r_rr = abs(R + self.W/2)
             r_fl = np.hypot(self.L, r_rl)
             r_fr = np.hypot(self.L, r_rr)
             
             # Direction based on vx
             direction = np.sign(vx) if abs(vx) > 0.01 else 1.0
             
             fl_speed = abs(omega_turn * r_fl) * direction
             fr_speed = abs(omega_turn * r_fr) * direction
             rl_speed = abs(omega_turn * r_rl) * direction
             rr_speed = abs(omega_turn * r_rr) * direction
        else:
            speed = np.hypot(vx, vy)
            fl_speed, fr_speed, rl_speed, rr_speed = speed, speed, speed, speed

        return {
            "angles": {"fl": fl_angle, "fr": fr_angle, "rl": 0.0, "rr": 0.0},
            "speeds": {"fl": fl_speed, "fr": fr_speed, "rl": rl_speed, "rr": rr_speed}
        }