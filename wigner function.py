def wigner_function(n, theta, p):
    def integrand(x):
        return np.exp(1j * p * x / hbar) * psi(n, theta - x / 2) * np.conj(psi(n, theta + x / 2))
    
    integral, _ = quad(lambda x: integrand(x).real, -10, 10)
    return (1 / (np.pi * hbar)) * integral