import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the system of ODEs
def system(t, y):
    x1, x2, x3, x4, x5, x6 = y  # y[0] = x1, y[1] = x2, etc.

    dx1_dt = x3
    dx3_dt = x5
    dx5_dt = -2 * x4**2 + x2
    dx2_dt = x4
    dx4_dt = x6
    dx6_dt = -x5**3 + x4 + x1 + np.sin(t)

    return [dx1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt, dx6_dt]

# Initial conditions
initial_conditions = [0, 0, 0, 0, 0, 0]  # Modify as needed

# Time range
t_span = (0, 10)  # Solve from t = 0 to t = 10
t_eval = np.linspace(0, 10, 100)  # 100 points for smooth plot

# Solve the system
solution = solve_ivp(system, t_span, initial_conditions, t_eval=t_eval)

# Extract results
t_values = solution.t
x1_values, x2_values = solution.y[0], solution.y[1]  # First and second equations

# Print the results
print(f"{'Time':<12} {'x1':<15} {'x2':<15}")
print("-" * 40)
for t, x1, x2 in zip(t_values, x1_values, x2_values):
    print(f"{t:<10.2f} {x1:<15.6f} {x2:<15.6f}")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(t_values, x1_values, label="x1(t)", linestyle='-', marker='o')
plt.plot(t_values, x2_values, label="x2(t)", linestyle='--', marker='s')
plt.xlabel("Time (t)")
plt.ylabel("Values of x1 and x2")
plt.title("Solution to the System of ODEs")
plt.legend(loc="best")
plt.grid()
plt.show()
