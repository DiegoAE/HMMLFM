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
end_t = 1010.
n_time_steps = 3000
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
    verbose=True,
)


obs = lfm_hmm.generate_observations()

lfm_hmm._mapB(obs)


# plt.plot(lfm_hmm.sample_locations, obs.flatten())
# for s in xrange(segments):
#     plt.axvline(x=lfm_hmm.sample_locations[lfm_hmm.locations_per_segment * s],
#                 color='red', linestyle='--')
# plt.show()

# lfm_hmm.train(obs, 5)
# print lfm_hmm.pi
# print lfm_hmm.A

# Recomendaciones de mauricio para los problemas numericos
# 1. Mirar el caso en en que el kernel no funciona bien (overdamped, underdamped, critically damped).
# 2. Asegurarse que el lengthscale sea más grande que las distancias entre samples.
# 3. Sumar un jitter para la matriz de covarianza.
# 4. Mirar el código de Matlab por que no funciona?
# 5. Graficar la matriz de covarianza obtenida para visualizar que puede estar fallando




