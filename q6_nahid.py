import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp ,solve_bvp

def linear_shooting_method(f,a,b,alpha,beta,h,max_iter=10,tol=1e-06):
    n=int((b-a)/h)

    def ivp1(x,y):
        return [y[1],f(x,y[0],y[1])]
    
    def ivp2(x,y):
        return [y[1],f(x,y[0],y[1])]
    
    x=np.linspace(a,b,n+1)

    sol1=solve_ivp(ivp1,[a,b],[alpha,0],t_eval=x,method='RK45')
    y1=sol1.y[0]

    sol2=solve_ivp(ivp2,[a,b],[0,1],t_eval=x,method='RK45')
    y2=sol2.y[0]

    c=(beta-y1[-1])/y2[-1]
    y=y1+c*y2

    return x,y

def f(x,y,y_prime):
    return 100*y

a,b=0,1
alpha,beta=1,np.exp(-10)

def exact(x):
    return np.exp(-10*x)

h1=0.1
h2=0.05

x1,y1=linear_shooting_method(f,a,b,alpha,beta,h1)
x2,y2=linear_shooting_method(f,a,b,alpha,beta,h2)

y_exact1=exact(x1)
y_exact2=exact(x2)

y_error1=np.abs(y1-y_exact1)
y_error2=np.abs(y2-y_exact2)

print(f'results for h=0.1: \n')
for i in range(len(x1)):
    print(f'x={x1[i]:0.5f}, y={y1[i]:0.5f}, y_exact={y_exact1[i]:0.5f}, error={y_error1[i]:0.5f}')
print(f"Maximum error for h = 0.1: {np.max(y_error1):.10f}\n")

print(f'results for h=0.05: \n')
for i in range(len(x2)):
    print(f'x={x2[i]:0.5f}, y={y2[i]:0.5f}, y_exact={y_exact2[i]:0.5f}, error={y_error2[i]:0.5f}')
print(f"Maximum error for h = 0.05: {np.max(y_error1):.10f}\n")    

def bvp_func(x,y):
     return np.vstack((y[1],100*y[0]))

def bc(ya,yb):
    return np.array([ya[0]-1,yb[0]-np.exp(-10)])

x_bvp=np.linspace(0,1,20)
y_bvp=np.zeros((2,x_bvp.size))
y_bvp[0] = np.exp(-10 * x_bvp)  
y_bvp[1] = -10 * np.exp(-10 * x_bvp) 

bvp = solve_bvp(bvp_func, bc, x_bvp, y_bvp)

x_plot=np.linspace(0,1,100)
y_plot_bvp=bvp.sol(x_plot)[0]
y_plot_exact=exact(x_plot)

error_bvp = np.abs(y_plot_bvp - y_plot_exact)
print(f"\nMaximum error for solve_bvp: {np.max(error_bvp):.10f}")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(x_plot, y_plot_exact, 'k-', label='Exact: y = exp(-10x)')
plt.plot(x1, y1, 'ro--', label='Linear Shooting (h=0.1)')
plt.plot(x2, y2, 'gx--', label='Linear Shooting (h=0.05)')
plt.plot(x_plot, y_plot_bvp, 'b.-', label='solve_bvp')
plt.legend()
plt.title('Solutions to y" = 100y with y(0)=1, y(1)=exp(-10)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(x1, y_error1, 'ro-', label='Error (h=0.1)')
plt.semilogy(x2, y_error2, 'gx-', label='Error (h=0.05)')
plt.semilogy(x_plot, error_bvp, 'b.-', label='Error (solve_bvp)')
plt.legend()
plt.title('Error Comparison (log scale)')
plt.xlabel('x')
plt.ylabel('|Error|')
plt.grid(True)

plt.tight_layout()
plt.show()

