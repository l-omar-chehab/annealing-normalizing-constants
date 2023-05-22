import numpy as np
from scipy.integrate import quad as integrator
from scipy.stats import norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def _fisher_integrand_old(x, t, mu, z1):
#     w_t = sigmoid(np.log(t * z1) - np.log(1 - t))
#     num = (norm.pdf(x, loc=mu, scale=1) - norm.pdf(x, loc=0, scale=1))**2
#     denom = w_t * norm.pdf(x, loc=mu, scale=1) + (1 - w_t) * norm.pdf(x, loc=0, scale=1)
#     return num / denom


def _fisher_integrand_log(x, t, mu, z1):
    w_t = sigmoid(np.log(t * z1) - np.log(1 - t))

    # first term
    a1 = [norm.logpdf(x, loc=mu, scale=1), norm.logpdf(x, loc=0, scale=1)]
    b1 = [1., -1.] if norm.logpdf(x, loc=mu, scale=1) > norm.logpdf(x, loc=0, scale=1) else [-1., 1.]
    first_term = 2 * logsumexp(a=a1, b=b1)

    # second term
    a2 = [norm.logpdf(x, loc=mu, scale=1), norm.logpdf(x, loc=0, scale=1)]
    b2 = [w_t, 1 - w_t]
    second_term = - logsumexp(a=a2, b=b2)

    return first_term + second_term


def _fisher_integrand(x, t, mu, z1):
    return np.exp(_fisher_integrand_log(x, t, mu, z1))


@np.vectorize
def fisher(t=0.5, mu=1., z1=1., verbose=False):
    result, abserr = integrator(
        _fisher_integrand,
        -np.inf, np.inf,
        args=(t, mu, z1),
        # epsabs=1e-6
    )
    if verbose:
        print(f"Integration: fisher computed with error {abserr}")
    return result


@np.vectorize
def fisher_integrated(mu=1., z1=1., verbose=False):
    result, abserr = integrator(
        fisher,
        0., 1.,
        args=(mu, z1),
        # epsabs=1e-6
    )
    if verbose:
        print(f"Integration: fisher computed with error {abserr}")
    return result


# Figure 1: fisher over the path (time) is high at the edges and almost zero elsewhere
plt.close("all")
fig, ax = plt.subplots(figsize=(7.5, 3.5))

ts = np.linspace(0., 1., 10)
mu = 5
z1 = 1.

fishers = fisher(t=ts, mu=mu, z1=z1)
ax.plot(ts, fishers)
ax.set(xlabel="t", ylabel="fisher")
fig.tight_layout()
plt.show()


# Figure 2: max fisher over the path evolves exponentially with mu, but that UB is too loose if we integrate on the path which is zero elsewhere
plt.close("all")
fig, ax = plt.subplots(figsize=(7.5, 3.5))

mus = np.linspace(0., 2., 20)

fishers_max = fisher(t=0., mu=mus, z1=z1)
ax.plot(mus, fishers_max)
ax.set(xlabel="mu", ylabel="fisher (max over t)")
plt.show()


# Figure 3: max fisher over the path vs. integrated fisher over the path, as a function of mu
plt.close("all")
fig, ax = plt.subplots(figsize=(7.5, 3.5))

mus = np.linspace(0., 2., 5)

fishers_max = fisher(t=0., mu=mus, z1=z1)
fishers_integrated = fisher_integrated(mu=mus, z1=z1)
ax.plot(mus, fishers_max, label="fisher (max over path)")
ax.plot(mus, fishers_integrated, label="fisher (integrated over path)")
ax.legend()
ax.set(xlabel="mu", ylabel="fisher")
plt.show()
