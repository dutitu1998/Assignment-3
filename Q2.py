import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lotka-Volterra system
def lotka_volterra(t, z):
    x, y = z
    dxdt = -0.1 * x + 0.02 * x * y
    dydt = 0.2 * y - 0.025 * x * y
    return [dxdt, dydt]

# Initial conditions
x0 = 6  # Initial predator population (in thousands)
y0 = 6  # Initial prey population (in thousands)
z0 = [x0, y0]

# Time span for the solution
t_span = (0, 50)  # Solve from t = 0 to t = 50
t_eval = np.linspace(0, 50, 1000)  # Points where the solution is evaluated

# Solve the system using solve_ivp
sol = solve_ivp(lotka_volterra, t_span, z0, t_eval=t_eval)

# Extract the solutions
x = sol.y[0]  # Predator population
y = sol.y[1]  # Prey population
t = sol.t     # Time points

# Plot the solutions
plt.figure(figsize=(10, 6))
plt.plot(t, x, 'r-', label='Predators (x(t))')
plt.plot(t, y, 'b-', label='Prey (y(t))')
plt.xlabel('Time (t)')
plt.ylabel('Population (in thousands)')
plt.title('Lotka-Volterra Predator-Prey Model')
plt.legend()
plt.grid()
plt.show()

# Find the time when x(t) = y(t)
# We look for the first time when the difference between x and y is zero
difference = np.abs(x - y)
idx = np.argmin(difference)  # Index of the minimum difference
t_equal = t[idx]  # Time when x(t) ≈ y(t)

print(f"The populations are first equal at t ≈ {t_equal:.2f}")