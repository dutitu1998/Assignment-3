import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp

def ode_1(y,t):
    return t*np.exp(3*t)-2*y

def ode_1_ivp(t,y):
    return t*np.exp(3*t)-2*y

def exact_1(t):
    return (1/5)*t*np.exp(3*t)-(1/25)*np.exp(3*t)+(1/25)*np.exp(-2*t)

def ode_2(y,t):
    return 1+(t-y)**2

def ode_2_ivp(t,y):
    return 1+(t-y)**2

def exact_2(t):
    return t+(1/(1-t))

t_span_1=np.linspace(0,1,101)
y0_1=0

y_odeint_1=odeint(ode_1,y0_1,t_span_1)

result_ivp_1=solve_ivp(ode_1_ivp,[0,1],[y0_1],t_eval=t_span_1,method='RK45')

y_ivp_1=result_ivp_1.y[0]

y_exact_1=exact_1(t_span_1)

error_odeint=np.abs(y_odeint_1.flatten()-y_exact_1)
error_ivp_1=np.abs(y_ivp_1-y_exact_1)

t_span_2=np.linspace(2,3,101)
y0_2=1

y_odeint_2=odeint(ode_2,y0_2,t_span_2)
result_ivp_2=solve_ivp(ode_2_ivp,[2,3],[y0_2],t_eval=t_span_2,method='RK45')
y_ivp_2=result_ivp_2.y[0]
y_exact_2=exact_2(t_span_2)

error_odeint_2=np.abs(y_odeint_2.flatten()-y_exact_2)
error_ivp_2=np.abs(y_ivp_2-y_exact_2)

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.plot(t_span_1,y_odeint_1,'r',label='odeint')
plt.plot(t_span_1,y_ivp_1,'c--',label='solve_ivp')
plt.plot(t_span_1,y_exact_1,'y',label='exact')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Problem 1')
plt.legend()

plt.subplot(2,2,2)
plt.semilogy(t_span_1,error_odeint,'k',label='odeint')
plt.semilogy(t_span_1,error_ivp_1,'c--',label='solve_ivp')
plt.xlabel('t')
plt.ylabel('error')
plt.title('error in problem 1')
plt.grid()
plt.legend()


plt.subplot(2,2,3)
plt.plot(t_span_2,y_odeint_2,'r',label='odeint')
plt.plot(t_span_2,y_ivp_2,'c--',label='solve_ivp')
plt.plot(t_span_2,y_exact_2,'y',label='exact')
plt.xlabel('t')
plt.ylabel('y')
plt.title('problem 2')
plt.grid()
plt.legend()

plt.subplot(2,2,4)
plt.semilogy(t_span_2,error_odeint_2,'k',label='odeint')
plt.semilogy(t_span_2,error_ivp_2,'c--',label='solve_ivp')
plt.xlabel('t')
plt.ylabel('error')
plt.title('error in problem 2')
plt.grid()
plt.legend()
plt.show()

