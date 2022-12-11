"""Brownian Motion"""
import numpy as np
from scipy.stats import norm

from replica.processes.base import SPExplicit
from replica.processes.analytical.gaussian import GaussianIncrements
from replica.utils.utils import check_positive_number, check_numeric, get_times


class BrownianMotion(SPExplicit):
    r"""Brownian motion

    .. image:: _static/brownian_motion_drawn.png


    A standard Brownian motion    :math:`\{W_t : t \geq 0\}` is defined by the following properties:

    1. Starts at zero, i.e. :math:`W(0) = 0`
    2. Independent increments
    3. :math:`W(t) - W(s)` follows a Gaussian distribution :math:`N(0, t-s)`
    4. Almost surely continuous

    A more general version of a Brownian motion, is the Drifted Brownian Motion which  is defined
    by the following SDE

    .. math::

        dX_t = \mu dt + \sigma dW_t \ \ \ \   t\in (0,T]


    with initial condition :math:`X_0 = x_0\in\mathbb{R}`, where

    - :math:`\mu` is the drift
    - :math:`\sigma>0` is the volatility
    - :math:`W_t` is a standard Brownian Motion.

    Clearly, the solution to this equation can be written as

    .. math::

        X_t = x_0 +  \mu t + \sigma W_t \ \ \ \ t \in [0,T]

    and each :math:`X_t \sim N(\mu t, \sigma^2 t)`.

    :param float drift: the parameter :math:`\mu` in the above SDE
    :param float scale: the parameter :math:`\sigma>0` in the above SDE
    :param float initial: the initial condition :math:`x_0` in the above SDE
    :param float T: the right hand endpoint of the time interval :math:`[0,T]`
        for the process
    :param numpy.random.Generator rng: a custom random number generator

    """

    def __init__(self, drift=0.0, scale=1.0, T=1.0, rng=None):
        super().__init__(T=T, rng=rng, initial=0.0)
        self.drift = drift
        self.scale = scale
        self.name = "Brownian Motion" if drift == 0.0 else "Brownian Motion with Drift"
        self.n = None
        self.times = None
        self.gaussian_increments = GaussianIncrements(T=self.T, rng=self.rng)

    @property
    def drift(self):
        return self._drift

    @drift.setter
    def drift(self, value):
        check_numeric(value, "Drift")
        self._drift = value

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        check_positive_number(value, "Scale")
        self._scale = value

    def _sample_brownian_motion(self, n):
        self.n = n
        self.times = get_times(self.T, self.n)
        bm = np.cumsum(self.scale * self.gaussian_increments.sample(n - 1))
        bm = np.insert(bm, 0, [0])
        if self.drift == 0:
            return bm
        else:
            return self.times * self.drift + bm

    def sample(self, n):
        """
        Generates a discrete time sample from a Brownian Motion instance.

        :param n: the number of steps
        :return: numpy array
        """
        return self._sample_brownian_motion(n)

    def _sample_brownian_motion_at(self, times):
        self.times = times
        bm = np.cumsum(self.scale * self.gaussian_increments.sample_at(times))

        if times[0] != 0:
            bm = np.insert(bm, 0, [0])

        if self.drift != 0:
            bm += [self.drift * t for t in times]

        return bm

    def sample_at(self, times):
        """
        Generates a sample from a Brownian motion at the specified times.

        :param times: the times which define the sample
        :return: numpy array
        """
        return self._sample_brownian_motion_at(times)

    def _process_expectation(self):
        return self.drift * self.times

    def process_expectation(self):
        expectations = self._process_expectation()
        return expectations

    def _process_variance(self):
        return (self.scale ** 2) * self.times

    def process_variance(self):
        variances = self._process_variance()
        return variances

    def _process_stds(self):
        return self.scale * np.sqrt(self.times)

    def process_stds(self):
        stds = self._process_stds()
        return stds

    def get_marginal(self, t):
        marginal = norm(loc=self.drift * t, scale=self.scale * np.sqrt(t))
        return marginal

    def draw(self, n, N, marginal=True, envelope=False, style='3sigma'):
        """
        Simulates and plots paths/trajectories from the instanced stochastic process.

        Produces different kind of visualisation illustrating the following elements:

        - times versus process values as lines
        - the expectation of the process across time
        - histogram showing the empirical marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - probability density function of the marginal distribution :math:`X_T` (optional when ``marginal = True``)
        - envelope of confidence intervals acroos time (optional when ``envelope = True``)

        :param n: number of steps in each path
        :param N: number of paths to simulate
        :param marginal: bool, default: True
        :param envelope: bool, default: False
        :param style: string, default: '3sigma'
        :return:
        """

        if style == '3sigma':
            return self._draw_3sigmastyle(n=n, N=N, marginal=marginal, envelope=envelope)
        elif style == 'qq':
            return self._draw_qqstyle(n, N, marginal=marginal, envelope=envelope)
        else:
            raise ValueError