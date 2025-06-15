class PIDController:
    def __init__(self, kp: float, ki: float, dt: float):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.dt = dt  # Time step
        self.integral = 0.0  # Integral term

    def update(self, error: float) -> float:
        """Update the PI controller with the given error and return the control output."""
        self.integral += error * self.dt  # Update integral term
        output = self.kp * error + self.ki * self.integral  # Calculate control output
        return output

    def reset(self):
        """Reset the integral term to zero."""
        self.integral = 0.0

    def set_gains(self, kp: float, ki: float):
        """Set the proportional and integral gains."""
        self.kp = kp
        self.ki = ki

    def get_gains(self) -> tuple:
        """Return the current proportional and integral gains."""
        return self.kp, self.ki

    def __repr__(self):
        return f"PIController(kp={self.kp}, ki={self.ki}, dt={self.dt})"

    def __str__(self):
        return f"PI Controller with Kp: {self.kp}, Ki: {self.ki}, dt: {self.dt}"

if __name__ == "__main__":
    # Example usage
    from device import Device  # Assuming you have a Device class defined in device.py
    from solar import solar  # Assuming you have a solar module with a Solar class
    print("Testing PIDController")
    controller = PIDController(kp=100, ki=0, dt=0)
    device = Device(
        base_load_energy_ma=20,
        full_load_energy_ma=500,
        battery_max_capacity_mah=6600,
        task_duration_s=1,
        init_battery_percentage=0.5,
        processing_rate_force_update_after_s=60*5,  # 5 minutes
    )
    solar = solar.Solar("../solcast2024.csv", scale_factor=0.18*0.28)
    time_s = 0
    battery_setpoint = 0.5  # Desired battery level (80%)
    while True:
        solar_production = solar.get_solar_a(time_s)
        device.set_pv_production_current_ma((solar.get_solar_a(time_s)/12)*1000)
        time_s += 60*5  # Increment time by 1 minute
        error = battery_setpoint - device.get_battery_percentage()
        control_output = -controller.update(error)
        control_output = max(0, min(control_output, 1))
        device.set_processing_rate(control_output)
        device.update(time_s)
        print(f"Time: {time_s}s, Battery Level: {device.get_battery_percentage()}, Control Output: {control_output}, solar production: {device.get_pv_production_normalized()}")
        if time_s >= 14 * 24 * 60 * 60 or device.get_battery_percentage() == 0:
            device.show_plot(show=True, save=False)
            break
