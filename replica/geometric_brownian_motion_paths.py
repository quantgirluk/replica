from geometric_brownian_motion import GeometricBrownianMotion
from brownian_motion_paths import StochasticProcessPaths
from scipy.stats import lognorm
import numpy as np


class GBMPaths(StochasticProcessPaths):
    def __init__(self, N, times, drift=0.0, volatility=1.0, initial=1.0):
        self.N = N
        self.times = times
        self.drift = drift
        self.volatility = volatility
        self.initial = initial
        process = GeometricBrownianMotion(drift=drift, volatility=volatility)
        self.paths = [process.sample_at(times) for k in range(int(N))]
        self.name = "Geometric Brownian Motion"

    def _process_expectation(self):
        return self.initial*np.exp(self.drift * self.times)

    def process_expectation(self):
        expectations = self._process_expectation()
        return expectations

    def _process_variance(self):
        variances = (self.initial ** 2) * np.exp(2 * self.drift * self.times) * (np.exp(self.times * self.volatility ** 2) - 1)
        return variances

    def process_variance(self):
        variances = self._process_variance()
        return variances

    def get_marginal(self, t):

        mu_x = np.log(self.initial) + (self.drift - 0.5 * self.volatility ** 2) * t
        sigma_x = self.volatility * np.sqrt(t)
        marginal = lognorm(s=sigma_x, scale=np.exp(mu_x))

        return marginal

    def plot(self):
        self._plot_paths()
        return 1

    def draw(self):
        self._draw_paths()
        return 1