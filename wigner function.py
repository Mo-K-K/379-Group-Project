def wigner_function(n, theta, p): #n is the energy level, theta is theta and p is momentum
    def integrand(x):
        return np.exp(1j * p * x / hbar) * psi(n, theta - x / 2) * np.conj(psi(n, theta + x / 2)) #i know functions in functions is a shit practice but it was an easy way forward
    
    integral, _ = quad(lambda x: integrand(x).real, -10, 10)
    return (1 / (np.pi * hbar)) * integral
