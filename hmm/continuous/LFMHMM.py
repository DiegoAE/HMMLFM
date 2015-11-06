__author__ = 'diego'

from hmm._BaseHMM import _BaseHMM
from hmm.lfm2kernel import SecondOrderLFMKernel
from scipy import stats
import numpy as np


class LFMHMM(_BaseHMM):

    def __init__(self, n, A, pi, number_outputs, start_t, end_t,
                 locations_per_segment, damper, spring, lengthscales,
                 precision=np.double, verbose=False):
        assert n > 0
        assert locations_per_segment > 0
        assert A.shape == (n, n)
        assert (pi.shape == (n, 1)) or (pi.shape == (n, ))
        assert number_outputs > 0
        assert len(damper) == len(spring) == len(lengthscales) == n
        assert all([len(x) == number_outputs for x in damper])
        assert all([len(x) == number_outputs for x in spring])
        _BaseHMM.__init__(self, n, None, precision, verbose)
        self.n = n  # number of hidden states
        self.A = A  # transition matrix
        self.pi = pi  # initial state PMF
        self.number_outputs = number_outputs
        self.start_t = start_t
        self.end_t = end_t
        self.sample_locations = np.linspace(start_t, end_t,
                                            locations_per_segment)
        self.locations_per_segment = locations_per_segment
        self.spring_cons = spring
        self.damper_cons = damper
        self.lengthscales = lengthscales
        self.memo_covs = {}

    def generate_observations(self, segments):
        output = np.zeros((segments, self.locations_per_segment),
                          dtype=self.precision)
        initial_state = np.nonzero(np.random.multinomial(1, self.pi))[0][0]
        hidden_states = [initial_state]
        for x in xrange(1, segments):
            hidden_states.append(np.nonzero(np.random.multinomial(
                1, self.A[hidden_states[-1]]))[0][0])
        for i in xrange(len(hidden_states)):
            state = hidden_states[i]
            cov = self.get_cov_function(state)
            realization = np.random.multivariate_normal(
                mean=np.zeros(cov.shape[0]), cov=cov)
            output[i, :] = realization
        print "Hidden States", hidden_states
        return output


    def get_cov_function(self, hidden_state, cache=True):
        if cache and (hidden_state in self.memo_covs):
            return self.memo_covs[hidden_state]
        B = np.asarray(self.spring_cons[hidden_state])
        C = np.asarray(self.damper_cons[hidden_state])
        l = self.lengthscales[hidden_state]
        # ts = self.sample_locations[segment_idx * lps: (segment_idx + 1) * lps]
        ts = self.sample_locations  # Each LFM is sampled over the same t
        assert len(ts) == self.locations_per_segment
        cov = SecondOrderLFMKernel.K(B, C, l, ts.reshape((-1, 1)))
        self.memo_covs[hidden_state] = cov
        return cov

    def _mapB(self,observations):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        '''

        numbers_of_segments = len(observations)

        self.B_map = np.zeros((self.n, numbers_of_segments),
                              dtype=self.precision)

        # strange behavior found between numpy and stats. See below.

        for j in xrange(self.n):
            for t in xrange(numbers_of_segments):
                cov = self.get_cov_function(j)
                self.B_map[j][t] = stats.multivariate_normal.pdf(
                    observations[t], np.zeros(cov.shape[0]), cov,
                    True)  # Allowing singularity in cov. This is weird

        print self.B_map







