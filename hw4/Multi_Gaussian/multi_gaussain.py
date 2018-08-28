import numpy as np
import matplotlib.pyplot as plt

mu = [15, 5]
sigma = [[20, 0], [0, 10]]
samples = np.random.multivariate_normal(mu, sigma, size=100)
plt.scatter(samples[:, 0], samples[:, 1])
plt.show()

def calc_mu_sigma(mu_true, sigma_true):
    samples = np.random.multivariate_normal(mu_true, sigma_true, size=100)
    mu = np.mean(samples, axis=0)
    sigma = (samples - mu).T @ (samples - mu) / len(samples)
    return mu, sigma

sigma1 = [[20, 0], [0, 10]]
calc_mu_sigma(mu, sigma1)

sigma1 = [[20, 14], [14, 10]]
calc_mu_sigma(mu, sigma1)

sigma1 = [[20, -14], [-14, 10]]
calc_mu_sigma(mu, sigma1)
