__author__ = 'diego'

from hmm.continuous.LFMHMM import LFMHMM
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

np.random.seed(200)

pi = np.array([0.3, 0.3, 0.4])
print "initial state distribution", pi
A = np.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
print "hidden state transition matrix\n", A

number_lfm = 3
outputs = 1
start_t = 0.1  # finding: it's problematic to choose 0 as starting point.
end_t = 5.1 # finding: it's problematic to choose long times.
            # since the cov's tend to be the same.
locations_per_segment = 100
# list of lists in case of multiple outputs
damper_constants = np.asarray([[1.], [3.], [6.]])
spring_constants = np.asarray([[3.], [1.], [5.]])
# implicitly assuming there is only one latent force governing the system.
# lengthscales = np.asarray([8., 10., 12.])
lengthscales = np.asarray([10., 10., 10.])
# it seems to be quite problematic when you choose big lenghtscales
noise_var = 0.0 # TODO: The Viterbi algorithm is failing when this noise is set.

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

# segments = 10
#
# obs = lfm_hmm.generate_observations(segments)
#
# plt.plot(np.linspace(start_t, end_t , segments), obs.flatten())
# for s in xrange(segments):
#     plt.axvline(x=lfm_hmm.sample_locations[lfm_hmm.locations_per_segment * s],
#                 color='red', linestyle='--')
# plt.show()

def aux_get_end(start, end, locations_per_segment, segments):
    assert locations_per_segment > 1
    ret = start + (end - start) * segments
    ret += ((end - start)/(locations_per_segment - 1)) * (segments - 1)
    return ret

# segments = 10
#
# obs_1, _ = lfm_hmm.generate_observations(segments)
# computed_end = aux_get_end(start_t, end_t, locations_per_segment, segments)
#
# sample_locations = np.linspace(start_t, computed_end,
#                                locations_per_segment * segments)
#
# plt.plot(sample_locations, obs_1.flatten())
# for s in xrange(segments):
#     plt.axvline(x=sample_locations[lfm_hmm.locations_per_segment * s],
#                 color='red', linestyle='--')
# plt.show()

obs = []
n_training_sequences = 5
hidden_states = np.zeros(n_training_sequences, dtype=object)
for i in xrange(n_training_sequences):
    segments = np.random.randint(1, 100)
    print "The %d-th sequence has length %d" % (i, segments)
    output, hidden = lfm_hmm.generate_observations(segments)
    obs.append(output)
    hidden_states[i] = hidden

# plotting covariance matrix

# plt.figure(1)
# plt.subplot(131)
# plt.imshow(lfm_hmm.get_cov_function(0))
# plt.subplot(132)
# plt.imshow(lfm_hmm.get_cov_function(1))
# plt.subplot(133)
# plt.imshow(lfm_hmm.get_cov_function(2))
# plt.show()

####

lfm_hmm.set_observations(obs)
lfm_hmm.reset()  # Reset to A and pi

print lfm_hmm.pi
print lfm_hmm.A

# print "====  B_maps[0] ======="
# print lfm_hmm.B_maps[0]
# print "======================="

print "start training"

lfm_hmm.train()

print "after training"
print lfm_hmm.pi
print lfm_hmm.A

recovered_paths = lfm_hmm._viterbi()

# Testng GP-LFM fitting from the Viterbi estimation

one_observation = obs[0][0]
one_hidden_state = recovered_paths[0][0]
# print one_observation, one_hidden_state

t_test = np.linspace(start_t, end_t, 600)

mean_pred, cov_pred = lfm_hmm.predict(t_test, one_hidden_state, one_observation)

diag_cov = np.diag(cov_pred)

plt.scatter(lfm_hmm.sample_locations, one_observation)
plt.plot(t_test, mean_pred)
# plt.plot(t_test, mean_pred.flatten() - 2 * np.sqrt(diag_cov), 'k--')
# plt.plot(t_test, mean_pred.flatten() + 2 * np.sqrt(diag_cov), 'k--')
plt.show()


# Recomendaciones de mauricio para los problemas numericos
# 1. Mirar el caso en en que el kernel no funciona bien (overdamped, underdamped, critically damped).
#   There is not system critically dammped.
# 2. Asegurarse que el lengthscale sea mas grande que las distancias entre samples.
#   Asegurado
# 3. Sumar un jitter para la matriz de covarianza.
# 4. Mirar el codigo de Matlab por que no funciona?
# 5. Graficar la matriz de covarianza obtenida para visualizar que puede estar fallando. Done. Good hint!






