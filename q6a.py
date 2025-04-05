import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp

# Define the differential equation y'' = 100*y as a system of first-order ODEs
def shooting_system(x, y):
    y1, y2 = y
    return [y2, 100*y1]

# Solve using the Linear Shooting Method
def shooting_method(h, y0, y1, x_end):
    x = np.arange(0, x_end + h, h)
    
    # First IVP with v1(0) = 0, v1'(0) = 1
    sol1 = solve_ivp(shooting_system, [0, x_end], [0, 1], t_eval=x, method='RK45')
    
    # Second IVP with v2(0) = 1, v2'(0) = 0
    sol2 = solve_ivp(shooting_system, [0, x_end], [1, 0], t_eval=x, method='RK45')

    v1, v1p = sol1.y
    v2, v2p = sol2.y
    
    # Compute the correct initial slope
    s = (y1 - v2[-1]) / v1[-1]
    y_shoot = v2 + s * v1  # y = v2 + s*v1

    return x, y_shoot

# Solve using different step sizes for shooting
x1, y1 = shooting_method(0.1, 1, np.exp(-10), 1)
x2, y2 = shooting_method(0.05, 1, np.exp(-10), 1)

# Print values
print("\nShooting Method (h=0.1):")
for i in range(len(x1)):
    print(f"x = {x1[i]:.2f}, y(x) = {y1[i]:.6f}")

print("\nShooting Method (h=0.05):")
for i in range(len(x2)):
    print(f"x = {x2[i]:.2f}, y(x) = {y2[i]:.6f}")

# Solve using solve_bvp
def bvp_system(x, y):
    y1, y2 = y
    return np.vstack([y2, 100*y1])

def bc(Ya, Yb):
    return np.array([Ya[0] - 1, Yb[0] - np.exp(-10)])

x_mesh = np.linspace(0, 1, 50)
Y_guess = np.zeros((2, x_mesh.size))  # Initial guess

bvp_solution = solve_bvp(bvp_system, bc, x_mesh, Y_guess)
y_bvp = bvp_solution.sol(x_mesh)[0]

print("\nsolve_bvp Solution:")
for i in range(len(bvp_solution.x)):
    print(f"x = {bvp_solution.x[i]:.2f}, y(x) = {bvp_solution.y[0, i]:.6f}")

# Analytical solution
x_exact = np.linspace(0, 1, 100)
y_exact = np.exp(-10 * x_exact)

print("\nExact Solution:")
for i in range(len(x_exact)):
    print(f"x = {x_exact[i]:.2f}, y(x) = {y_exact[i]:.6f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_exact, y_exact, 'k-', label="Analytical Solution")
plt.plot(x1, y1, 'ro-', label="Shooting Method (h=0.1)", markersize=4)
plt.plot(x2, y2, 'bs-', label="Shooting Method (h=0.05)", markersize=4)
plt.plot(x_mesh, y_bvp, 'g*-', label="solve_bvp Solution", markersize=4)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Numerical and Analytical Solutions")
plt.legend()
plt.grid()
plt.show()