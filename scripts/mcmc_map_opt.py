"""
MCMC benchmark
~~~~
Monte Carlo estimations wrt MAP estimates alone.
"""

import timeit
import numpy as np
import scipy
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

import theano.tensor as tt

# Controls the no. capsules x no. votes to be benchmarked.
NUM_PARAMS = 100000000

# a_k,n
mix_prob = np.random.normal(size=(1, NUM_PARAMS))

# N(x_m | a_k,n)
log_prob = np.random.normal(size=(1, NUM_PARAMS))


def compute_closed_argmax(mix_prob, log_prob):
    """Compute and return deterministic argmax."""

    mix_prob = mix_prob / np.sum(mix_prob)
    posterior = log_prob * mix_prob
    return np.argmax(posterior)


class LogLike(tt.Op):

    """
    Defines a "black-box" likelihood that is unable to leverage theano-native
    gradient computation.
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def likelihood(self, x):
        """Combining the capsule x vote matrix into a stacked vector."""

        if x >= NUM_PARAMS or x < 0:
            return 0.
        return mix_prob[0][x] * log_prob[0][x]

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs

        x = int(theta[0])

        logl = self.likelihood(x)

        outputs[0][0] = np.array(logl)


def mcmc_sample_w_plot(mix_prob, log_prob):
    """MCMC trace construction with pyplot visualization."""

    logl = LogLike()

    with pm.Model() as model:
        x = pm.Uniform('x', lower=0, upper=NUM_PARAMS)

        theta = tt.as_tensor_variable([x])

        pm.Potential('likelihood', logl(theta))
        trace = pm.sample(cores=1, chains=1, step=pm.Metropolis())

    pm.traceplot(trace)
    axes = plt.gcf().get_axes()
    ax = axes[0]  # select the first subplot
    plt.sca(ax)  # set current subplot
    plt.plot()
    print(az.summary(trace, round_to=2))

    return pm.find_MAP(model=model)


def mcmc_sample(mix_prob, log_prob):
    """MCMC trace construction alone."""

    logl = LogLike()

    with pm.Model() as model:
        x = pm.Uniform('x', lower=0, upper=NUM_PARAMS)

        theta = tt.as_tensor_variable([x])

        pm.Potential('likelihood', logl(theta))
        trace = pm.sample(cores=1, chains=1, step=pm.Metropolis())

    return trace


closed_benchmark = timeit.timeit(lambda: compute_closed_argmax(
    mix_prob, log_prob), number=10)/10

print("Welcome to Monte Carlo")
print("\n")
mcmc_benchmark = timeit.timeit(lambda: mcmc_sample(
    mix_prob, log_prob), number=10)/10

print(
    f'Closed evaluation with {NUM_PARAMS} parameters averaged over 10 runs - {closed_benchmark} seconds.')
print(
    f'Sampled evaluation with {NUM_PARAMS} parameters averaged over 10 runs - {mcmc_benchmark} seconds')

plt.show()

##
# Attempt to construct well-behaved synthetic distribution.
# Use negative binomial ks,ns evaluated ~N(k, 1)
##

# a_k,n

mix_prob = np.random.negative_binomial(3, 0.5, size=(1, NUM_PARAMS))
gaussian_mixtures = scipy.stats.norm(
    loc=mix_prob, scale=np.ones((1, NUM_PARAMS)))
log_prob = gaussian_mixtures.pdf(np.random.rand(1, NUM_PARAMS))
