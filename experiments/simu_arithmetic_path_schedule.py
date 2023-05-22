import numpy as np
from scipy.integrate import quad as integrator
import matplotlib.pyplot as plt




def _mse_ub_integrand(t, log10z, alpha):
    integrand = (10**log10z)**2 * alpha**2 * t**(2 * alpha - 2) / (t**alpha * ((10**log10z) - 1) + 1)**4
    return integrand


@np.vectorize
def mse_ub(log10z=0., alpha=1., verbose=False):
    result, abserr = integrator(
        _mse_ub_integrand,
        0., 1.,
        args=(log10z, alpha),
        # epsabs=tol
    )
    if verbose:
        print(f"Integration: score mean (nce) computed with error {abserr}")
    return result


# Figure 1
plt.close("all")
fig, ax = plt.subplots(figsize=(7.5, 3.5))
log10zs = np.linspace(-1, 1, 100)
mses = mse_ub(log10z=log10zs, alpha=1.)
log10mses = np.log10(mses)
ax.plot(log10zs, log10mses)
ax.set(xlabel="Z (log10)", ylabel="MSE (log10)")
plt.show()

# Figure 2
plt.close("all")
fig, ax = plt.subplots(figsize=(7.5, 3.5))
alphas = np.linspace(0.5, 10, 100)
mses = mse_ub(log10z=np.log10(2.), alpha=alphas)
log10mses = np.log10(mses)
ax.plot(alphas, log10mses)
ax.set(xlabel="alpha", ylabel="MSE (log10)")
plt.show()


# Figure 3
plt.close("all")
fig, ax = plt.subplots(figsize=(7.5, 3.5))
log10zs = np.linspace(-1, 1, 100)
alphas = np.linspace(0.5, 20, 100)
log10zs_grid, alphas_grid = np.meshgrid(log10zs, alphas)
mse_grid = mse_ub(log10z=log10zs_grid, alpha=alphas_grid)
log10mse_grid = np.log10(mse_grid)
img = ax.contourf(log10zs_grid, alphas_grid, log10mse_grid, cmap="magma")
# img = ax.imshow(log10mse_grid)
ax.set(xlabel="Z (log10)", ylabel="alpha")
cbar = fig.colorbar(img)
cbar.ax.set_ylabel('MSE (log10)', rotation=270, labelpad=20)
plt.show()
