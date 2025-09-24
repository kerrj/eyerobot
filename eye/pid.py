class PIDController:
    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        max_output=float("inf"),
        integral_bound=float("inf"),
        leakage_factor=0.9,
    ):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.max_output = max_output  # Maximum output limit
        self.integral_bound = integral_bound  # Absolute bound for integral term
        self.leakage_factor = leakage_factor  # Leakage factor for the integral term

        self.integral = 0.0
        self.previous_error = None
        self.output = 0.0
        self.last_time = None

    def update(self, target, signal, t):
        """
        Update the PID controller with the current target and signal values.

        Args:
            target (float): The desired target value.
            signal (float): The current signal value.

        Returns:
            float: The output of the PID controller.
        """
        error = target - signal

        # Proportional term
        P_term = self.Kp * error

        # Integral term with leakage and conditional integration
        if (
            abs(self.output) < self.max_output
        ):  # Conditional integration based on output saturation
            self.integral = (
                self.leakage_factor * self.integral + error * (t - self.last_time)
                if self.last_time is not None
                else 0.0
            )
            # Apply unified integral bounds
            self.integral = max(
                min(self.integral, self.integral_bound), -self.integral_bound
            )

        I_term = self.Ki * self.integral

        # Derivative term (finite differences)
        dedt = (
            (error - self.previous_error) / (t - self.last_time)
            if self.last_time is not None
            else 0.0
        )
        D_term = self.Kd * dedt
        self.previous_error = error
        self.last_time = t

        # PID output
        output = P_term + I_term + D_term

        # Clamp output to max_output
        self.output = max(min(output, self.max_output), -self.max_output)

        return self.output

    def reset_setpoint(self):
        self.integral = 0.0
        self.previous_error = None
        self.output = 0.0
        self.last_time = None
