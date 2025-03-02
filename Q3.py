import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the competition model
def competition_model(t, z):
    x, y = z
    dxdt = x * (2 - 0.4 * x - 0.3 * y)
    dydt = y * (1 - 0.1 * y - 0.3 * x)
    return [dxdt, dydt]

# Initial conditions for each case
initial_conditions = [(1.5, 3.5),(1, 1),(2, 7),(4.5, 0.5)]

# Time span for the solution (in years)
t_span = (0, 50)  # Solve from t = 0 to t = 50 years
t_eval = np.linspace(0, 50, 1000)  # Points where the solution is evaluated

# Plot the solutions for each case
plt.figure(figsize=(14, 8))

for i, (x0, y0) in enumerate(initial_conditions):
    z0 = [x0, y0]
    
    # Solve the system using solve_ivp
    sol = solve_ivp(competition_model, t_span, z0, t_eval=t_eval)
    
    # Extract the solutions
    x = sol.y[0]  # Population x(t)
    y = sol.y[1]  # Population y(t)
    t = sol.t     # Time points
    
    # Plot the solutions
    plt.subplot(2, 2, i + 1)
    plt.plot(t, x, 'r-', label='x(t) (Population 1)')
    plt.plot(t, y, 'b-', label='y(t) (Population 2)')
    plt.xlabel('Time (years)')
    plt.ylabel('Population (in thousands)')
    plt.title(f'Case {chr(97 + i)}: x(0) = {x0}, y(0) = {y0}')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()