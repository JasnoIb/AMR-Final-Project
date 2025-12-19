import numpy as np

class LyapunovController:
    """
    Non-linear control law based on Lyapunov Stability analysis.
    Guarantees global asymptotic stability for the kinematic model.
    
    Includes automatic direction switching (Reverse) if the target 
    is behind the robot.
    """
    def __init__(self, k_rho=1.0, k_alpha=2.5, k_beta=-1.0, max_v=2.0, max_omega=np.radians(90)):
        # Tuning Gains (Standard starting values)
        # k_rho:   Pulls the robot like a magnet to the target position
        # k_alpha: Steers the robot to face the target
        # k_beta:  Adjusts final orientation to match target orientation
        self.k_rho = k_rho      
        self.k_alpha = k_alpha  
        self.k_beta = k_beta    
        
        self.max_v = max_v
        self.max_omega = max_omega

    def compute_commands(self, current_state, target_state, wheelbase=0.33):
        """
        Calculates control inputs (v, steering) and the Lyapunov Energy (V).
        
        Args:
            current_state: [x, y, theta] (from the robot model)
            target_state:  [x_ref, y_ref, theta_ref] (desired waypoint)
            wheelbase:     Length of robot (needed for Ackermann conversion)
            
        Returns:
            v_cmd:           Linear velocity command (m/s)
            steering_angle:  Steering angle command (radians)
            lyapunov_energy: The scalar value of the stability function V (should decay to 0)
        """
        x, y, theta = current_state
        x_g, y_g, theta_g = target_state # Goal state

        # 1. Coordinate Transformation
        # Calculate error in global frame
        dx = x_g - x
        dy = y_g - y
        
        # Distance to target (rho)
        rho = np.hypot(dx, dy)
        
        # Angle to target relative to global frame
        # We define alpha as the angle error between "where we are facing" and "where the target is"
        target_heading = np.arctan2(dy, dx)
        alpha = target_heading - theta
        
        # Normalize angle to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi 
        
        # Angle error between "where the target is" and "target final orientation"
        beta = theta_g - theta - alpha
        beta = (beta + np.pi) % (2 * np.pi) - np.pi # Angle wrapping

        # 2. Lyapunov Function Calculation
        # V = 0.5 * rho^2 + 0.5 * alpha^2
        # Ideally, this number should strictly decrease over time.
        lyapunov_energy = 0.5 * rho**2 + 0.5 * alpha**2

        # 3. Control Law (Standard Unicycle Model)
        # If we are very close to the target, stop to avoid singularity/jitter
        if rho < 0.05:
            v_cmd = 0.0
            omega_cmd = 0.0
        else:
            # Direction Logic: Allow Reversing
            # If the target is behind us (|alpha| > 90 deg), it's faster to reverse.
            if abs(alpha) > np.pi / 2:
                # Reverse mode: Invert alpha and beta logic
                alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
                beta = (beta + np.pi) % (2 * np.pi) - np.pi
                
                # Command negative velocity
                v_cmd = -self.k_rho * rho * np.cos(alpha)
                
                # Omega logic remains similar but acts on the new alpha
                omega_cmd = self.k_alpha * alpha + self.k_beta * beta
            else:
                # Forward mode
                v_cmd = self.k_rho * rho * np.cos(alpha)
                omega_cmd = self.k_alpha * alpha + self.k_beta * beta

        # 4. Saturation (Physical Limits)
        v_cmd = np.clip(v_cmd, -self.max_v, self.max_v)
        omega_cmd = np.clip(omega_cmd, -self.max_omega, self.max_omega)

        # 5. Ackermann Conversion (Omega -> Steering Angle)
        # Relationship: tan(delta) = (omega * L) / v
        # Singularity check: If v is near zero, steering angle is undefined (or 0)
        if abs(v_cmd) < 0.01:
            steering_angle = 0.0
        else:
            steering_angle = np.arctan((omega_cmd * wheelbase) / v_cmd)

        return v_cmd, steering_angle, lyapunov_energy
