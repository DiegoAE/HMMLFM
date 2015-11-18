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
start_t = 10.0  # finding: it's problematic to choose 0 as starting point.
end_t = 110.
locations_per_segment = 300
# list of lists in case of multiple outputs
damper_constants = np.asarray([[1.], [3.], [6.]])
spring_constants = np.asarray([[3.], [1.], [5.]])
# implicitly assuming there is only one latent force governing the system.
lengthscales = np.asarray([8., 10., 12.])
# it seems to be quite problematic when you choose big lenghtscales

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

segments = 10

obs_1 = lfm_hmm.generate_observations(segments)

computed_end = aux_get_end(start_t, end_t, locations_per_segment, segments)

sample_locations = np.linspace(start_t, computed_end,
                               locations_per_segment * segments)

# plt.plot(sample_locations, obs.flatten())
# for s in xrange(segments):
#     plt.axvline(x=sample_locations[lfm_hmm.locations_per_segment * s],
#                 color='red', linestyle='--')
# plt.show()

segments = 20

obs_2 = lfm_hmm.generate_observations(segments)

computed_end = aux_get_end(start_t, end_t, locations_per_segment, segments)

sample_locations = np.linspace(start_t, computed_end,
                               locations_per_segment * segments)

# plt.plot(sample_locations, obs.flatten())
# for s in xrange(segments):
#     plt.axvline(x=sample_locations[lfm_hmm.locations_per_segment * s],
#                 color='red', linestyle='--')
# plt.show()

obs = []
n_training_sequences = 20
for i in xrange(n_training_sequences):
    segments = np.random.randint(1, 100)
    print "The %d-th sequence has length %d" % (i, segments)
    obs.append(lfm_hmm.generate_observations(segments))

lfm_hmm.set_observations(obs)
lfm_hmm.train()

# lfm_hmm.train(obs, 100)
print lfm_hmm.pi
print lfm_hmm.A

# TODO: Change the initial values of pi and A.

# Recomendaciones de mauricio para los problemas numericos
# 1. Mirar el caso en en que el kernel no funciona bien (overdamped, underdamped, critically damped).
#   There is not system critically dammped.
# 2. Asegurarse que el lengthscale sea mas grande que las distancias entre samples.
#   Asegurado
# 3. Sumar un jitter para la matriz de covarianza.
# 4. Mirar el codigo de Matlab por que no funciona?
# 5. Graficar la matriz de covarianza obtenida para visualizar que puede estar fallando




