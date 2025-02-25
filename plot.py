theta_vals = np.linspace(-3, 3, 50)
p_vals = np.linspace(-3, 3, 50)

wigner_vals = np.zeros((50, 50))
for i, theta in enumerate(theta_vals):
    for j, p in enumerate(p_vals):
        wigner_vals[i, j] = wigner_function(0, theta, p)

plt.figure(figsize=(8, 6))
plt.contourf(theta_vals, p_vals, wigner_vals.T, levels=100, cmap='RdBu')
plt.colorbar(label='Wigner function')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p$')
plt.title('Wigner')
plt.show()