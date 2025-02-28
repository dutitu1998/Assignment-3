import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
g = 32.17  # Acceleration due to gravity in ft/s^2
L = 2      # Length of the pendulum in feet

# Convert the second-order ODE into a system of first-order ODEs
def pendulum_system(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Initial conditions
theta0 = np.pi / 6  # Initial angle in radians
omega0 = 0          # Initial angular velocity
y0 = [theta0, omega0]

# Time span and evaluation points
t_span = (0, 2)  # From t = 0 to t = 2 seconds
t_eval = np.arange(0, 2.1, 0.1)  # Time points at 0.1s increments

# Solve the system using solve_ivp
sol = solve_ivp(pendulum_system, t_span, y0, t_eval=t_eval)

# Extract the solution
theta = sol.y[0]  # Angle theta(t)
t = sol.t         # Time points

theta_degrees = np.degrees(theta)  # Convert radians to degrees

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t, theta_degrees, 'b-', label=r'$\theta(t)$ (degrees)')
plt.xlabel('Time (s)')
plt.ylabel(r'$\theta(t)$ (degrees)')
plt.title('Motion of a Swinging Pendulum')
plt.legend()
plt.grid()
plt.show()

# Print the values of theta at each time step in degrees
for time, angle in zip(t, theta_degrees):
    print(f"t = {time:.1f} s, θ = {angle:.4f} degrees")
