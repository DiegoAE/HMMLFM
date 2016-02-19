from hmm.continuous.LFMHMM import LFMHMM
from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

seed = np.random.random_integers(10000)
np.random.seed(seed)
print "USED SEED", seed

pi = np.array([0.3, 0.3, 0.4])
print "initial state distribution", pi
A = np.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
print "hidden state transition matrix\n", A

number_lfm = 3
outputs = 3
start_t = 0.1  # finding: it's problematic to choose 0 as starting point.
end_t = 5.1  # finding: it's problematic to choose long times.
# since the cov's tend to be the same.
locations_per_segment = 20
# list of lists in case of multiple outputs
damper_constants = np.asarray([[1., 3., 7.5], [3., 1, 0.5], [6., 5., 4.]])
spring_constants = np.asarray([[3., 1, 2.5], [1., 3, 9.0], [5., 6., 4.5]])
# implicitly assuming there is only one latent force governing the system.
lengthscales = np.asarray([[10.], [10.], [10.]])
# it seems to be quite problematic when you choose big lenghtscales
noise_var = np.array([0.0005, 0.0005, 0.0005])  # Viterbi starts failing when this noise is set.

lfm_hmm = LFMHMMcontinuousMO(outputs, number_lfm, locations_per_segment, start_t, end_t,
                 verbose=True)
lfm_hmm.set_params(A, pi, damper_constants, spring_constants, lengthscales,
                   noise_var)

# Plotting
# Keep in mind that this doesn't generate continuous observations.
segments = 10
obs_1, _ = lfm_hmm.generate_observations(segments)
last_value = 0
for i in xrange(segments):
    plt.axvline(x=last_value, color='red', linestyle='--')
    sl = lfm_hmm.sample_locations
    current_obs = obs_1[i]
    current_outputs = np.zeros((locations_per_segment, outputs))
    # separating the outputs accordingly.
    for j in xrange(outputs):
        current_outputs[:, j] = current_obs[j::outputs]
    plt.plot(last_value + sl - sl[0], current_outputs)
    last_value += end_t - start_t
plt.show()