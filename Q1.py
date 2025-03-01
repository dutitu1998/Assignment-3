import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# Define the differential equations
def model1(t,y):
    return t * np.exp(3*t) - 2*y

def model2(t, y):
    return 1 + (t - y)**2

# Define the exact solutions
def exact_solution1(t):
    return (1/5)*t*np.exp(3*t) - (1/25)*np.exp(3*t) + (1/25)*np.exp(-2*t)

def exact_solution2(t):
    return t + 1/(1 - t)

# Initial conditions and time spans
t1 = np.linspace(0, 1, 100)
t2 = np.linspace(2, 3, 100)
y0_1 = 0
y0_2 = 1

# Solve using odeint
y_odeint1 = odeint(model1, y0_1, t1,tfirst=True)
y_odeint2 = odeint(model2, y0_2, t2,tfirst=True)

# Solve using solve_ivp
sol_ivp1 = solve_ivp(model1, [0, 1], [y0_1], t_eval=t1)
sol_ivp2 = solve_ivp(model2, [2, 3], [y0_2], t_eval=t2)

# Extract solutions from solve_ivp
y_ivp1 = sol_ivp1.y[0]
y_ivp2 = sol_ivp2.y[0]

# Calculate exact solutions
y_exact1 = exact_solution1(t1)
y_exact2 = exact_solution2(t2)

# Calculate errors
error_odeint1 = np.abs(y_odeint1.flatten() - y_exact1)
error_ivp1 = np.abs(y_ivp1 - y_exact1)

error_odeint2 = np.abs(y_odeint2.flatten() - y_exact2)
error_ivp2 = np.abs(y_ivp2 - y_exact2)

# Plot the solutions and errors
plt.figure(figsize=(12, 8))

# Plot for the first problem
plt.subplot(2, 2, 1)
plt.plot(t1, y_odeint1, 'b-', label='odeint')
plt.plot(t1, y_ivp1, 'r--', label='solve_ivp')
plt.plot(t1, y_exact1, 'g-', label='Exact')
plt.title('Solution for i)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t1, error_odeint1, 'b-', label='odeint error')
plt.plot(t1, error_ivp1, 'r--', label='solve_ivp error')
plt.title('Error for i)')
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()

# Plot for the second problem
plt.subplot(2, 2, 3)
plt.plot(t2, y_odeint2, 'b-', label='odeint')
plt.plot(t2, y_ivp2, 'r--', label='solve_ivp')
plt.plot(t2, y_exact2, 'g-', label='Exact')
plt.title('Solution for ii)')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t2, error_odeint2, 'b-', label='odeint error')
plt.plot(t2, error_ivp2, 'r--', label='solve_ivp error')
plt.title('Error for ii)')
plt.xlabel('t')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.show()
