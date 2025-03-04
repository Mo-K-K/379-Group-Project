import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from scipy.integrate import quad

hbar = 1#1.05*10**(-34)
I=1 #ang mom

def psi(n, theta):
    '''initialise the wave function, this is just a test wave function not the actual one '''
    norm = 1/np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    return norm * np.exp(-theta**2 /2)*eval_hermite(n, theta) #approx wave function using hermite functions

def wigner_function(n, theta, p):
    '''function for the wigner function, includes the integrand setup'''
    def integrand(x):
        return np.exp(1j * p * x / hbar) * psi(n, theta - x / 2) * np.conj(psi(n, theta + x / 2)) #yeah yeah, function in function is shit, but it works, the integrand
    
    integral, _ = quad(lambda x: integrand(x).real, -10, 10)
    return (1 / (np.pi * hbar)) * integral #normalised integration being returned 

theta_vals = np.linspace(-3, 3, 50) #init the values
p_vals = np.linspace(-3, 3, 50)

wigner_vals = np.zeros((50, 50))
for i, theta in enumerate(theta_vals):
    for j, p in enumerate(p_vals):
        wigner_vals[i, j] = wigner_function(0, theta, p) #calcuate wigner for each theta and p value
        
#some cool plottng tings
        
plt.figure(figsize=(8, 6))
plt.contourf(theta_vals, p_vals, wigner_vals.T, levels=100, cmap='RdBu')
plt.colorbar(label='Wigner function')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title('Wigner')
plt.show()