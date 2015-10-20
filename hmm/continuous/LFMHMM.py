__author__ = 'diego'

from hmm._BaseHMM import _BaseHMM
from hmm.lfm2kernel import SecondOrderLFMKernel
import numpy as np


class LFMHMM(_BaseHMM):

    def __init__(self, n, A, pi, number_outputs, start_t, end_t, n_time_steps,
                 segments, damper, spring, lengthscales,
                 precision=np.double, verbose=False):
        assert (n_time_steps % segments) == 0
        assert n > 0
        assert segments > 0
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
        self.n_time_steps = n_time_steps
        self.segments = segments
        self.sample_locations = np.linspace(start_t, end_t, n_time_steps)
        self.locations_per_segment = n_time_steps / segments
        self.spring_cons = spring
        self.damper_cons = damper
        self.lengthscales = lengthscales

    def generate_observations(self):
        output = np.zeros(0, dtype=self.precision)
        initial_state = np.nonzero(np.random.multinomial(1, self.pi))[0][0]
        hidden_states = [initial_state]
        for x in xrange(1, self.segments):
            hidden_states.append(np.nonzero(np.random.multinomial(
                1, self.A[hidden_states[-1]]))[0][0])
        for i in xrange(len(hidden_states)):
            state = hidden_states[i]
            B = np.asarray(self.spring_cons[state])
            C = np.asarray(self.damper_cons[state])
            l = self.lengthscales[state]
            print state, B, C, l
            # TODO: compute the GP output with B, C, l and sample_locations
        return hidden_states










