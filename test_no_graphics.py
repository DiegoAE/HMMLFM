__author__ = 'diego'

from hmm.continuous.LFMHMM import LFMHMM
import numpy as np

#np.random.seed(200)

pi = np.array([0.3, 0.3, 0.4])
print "initial state distribution", pi
A = np.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
print "hidden state transition matrix\n", A

number_lfm = 3
outputs = 1
start_t = 0.1  # finding: it's problematic to choose 0 as starting point.
end_t = 5.1  # finding: it's problematic to choose long times.
# since the cov's tend to be the same.
locations_per_segment = 20
# list of lists in case of multiple outputs
damper_constants = np.asarray([[1.], [3.], [6.]])
spring_constants = np.asarray([[3.], [1.], [5.]])
# implicitly assuming there is only one latent force governing the system.
# lengthscales = np.asarray([8., 10., 12.])
lengthscales = np.asarray([[10.], [10.], [10.]])
# it seems to be quite problematic when you choose big lenghtscales
noise_var = 0.0005  # Viterbi starts failing when this noise is set.

lfm_hmm = LFMHMM(
    number_lfm,
    A,
    pi,
    outputs,
    start_t,
    end_t,
    locations_per_segment,
    damper_constants,
    spring_constants,
    lengthscales,
    noise_var,
    verbose=True,
)


obs = []
n_training_sequences = 30
hidden_states = np.zeros(n_training_sequences, dtype=object)
for i in xrange(n_training_sequences):
    segments = np.random.randint(1, 100)
    print "The %d-th sequence has length %d" % (i, segments)
    output, hidden = lfm_hmm.generate_observations(segments)
    obs.append(output)
    hidden_states[i] = hidden


lfm_hmm.set_observations(obs)
lfm_hmm.reset(emissions_reset=True)  # Reset to A and pi

print lfm_hmm.pi
print lfm_hmm.A
print lfm_hmm.LFMparams

print "start training"

lfm_hmm.train()

print "after training"

print lfm_hmm.pi
print lfm_hmm.A
print lfm_hmm.LFMparams