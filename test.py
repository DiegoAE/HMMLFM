__author__ = 'diego'

from hmm.continuous.LFMHMM import LFMHMM
import numpy as np


pi = np.array([0.3, 0.3, 0.4])
print "initial state distribution", pi
A = np.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
print "hidden state transition matrix\n", A

number_lfm = 3
outputs = 1
start_t = 0.0
end_t = 10.0
n_time_steps = 100
segments = 10
# list of lists in case of multiple outputs
damper_constants = np.asarray([[1.], [3.], [5.]])
spring_constants = np.asarray([[3.], [1.], [5.]])
# implicitly assuming there is only one latent force governing the system.
lengthscales = np.asarray([100., 100., 10.])

lfm_hmm = LFMHMM(
    number_lfm,
    A,
    pi,
    outputs,
    start_t,
    end_t,
    n_time_steps,
    segments,
    damper_constants,
    spring_constants,
    lengthscales,
)


hs = lfm_hmm.generate_observations()

print hs