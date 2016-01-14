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
end_t = 5.1  # finding: it's problematic to choose long times.
# since the cov's tend to be the same.
locations_per_segment = 20
# list of lists in case of multiple outputs
damper_constants = np.asarray([[1.], [3.], [6.]])
spring_constants = np.asarray([[3.], [1.], [5.]])
# implicitly assuming there is only one latent force governing the system.
# lengthscales = np.asarray([8., 10., 12.])
lengthscales = np.asarray([10., 10., 10.])
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

# Plotting

# segments = 10
# obs_1, _ = lfm_hmm.generate_observations(segments)
# last_value = 0
# for i in xrange(segments):
#     plt.axvline(x=last_value, color='red', linestyle='--')
#     sl = lfm_hmm.sample_locations
#     plt.plot(last_value + sl - sl[0], obs_1[i])
#     print last_value
#     print last_value + sl - sl[0]
#     last_value += end_t - start_t
# plt.show()


obs = []
n_training_sequences = 10
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

lfm_hmm.set_observations(obs)
lfm_hmm.reset()  # Reset to A and pi

print lfm_hmm.pi
print lfm_hmm.A

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


print recovered_paths - hidden_states

plt.scatter(lfm_hmm.sample_locations, one_observation)
plt.plot(t_test, mean_pred)
plt.plot(t_test, mean_pred.flatten() - 2 * np.sqrt(diag_cov), 'k--')
plt.plot(t_test, mean_pred.flatten() + 2 * np.sqrt(diag_cov), 'k--')
plt.show()


# Recomendaciones de mauricio para los problemas numericos
# 1. Mirar el caso en en que el kernel no funciona bien (overdamped, underdamped, critically damped).
#   There is not system critically dammped.
# 2. Asegurarse que el lengthscale sea mas grande que las distancias entre samples.
#   Asegurado
# 3. Sumar un jitter para la matriz de covarianza.
# 4. Mirar el codigo de Matlab por que no funciona?
# 5. Graficar la matriz de covarianza obtenida para visualizar que puede estar fallando. Done. Good hint!


# First: experiment Validation of the viterbi algorithm

# now the lfm_hmm has the estimated pi and A
# it's necessary to create a new instance of LFMHMM

lfm_hmm_reference = LFMHMM(
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
n_training_sequences = 20
hidden_states_reference = np.zeros(n_training_sequences, dtype=object)
for i in xrange(n_training_sequences):
    segments = np.random.randint(1, 100)
    print "The %d-th sequence has length %d" % (i, segments)
    output, hidden = lfm_hmm_reference.generate_observations(segments)
    obs.append(output)
    hidden_states_reference[i] = hidden


recovered_paths = lfm_hmm._viterbi(obs)

diff = recovered_paths - hidden_states_reference

mismatchs = 0
total_segments = 0
for i in xrange(len(diff)):
    mismatchs += np.count_nonzero(diff[i])
    total_segments += len(diff[i])
print "the viterbi algorithm failed in %d from %d" % (mismatchs, total_segments)


# Second experiment: Regression
regression_observation = [obs[0]]
hidden_states_ground_truth = np.array(hidden_states_reference[0])
lfm_hmm.set_observations(regression_observation)
regression_hidden_states = lfm_hmm._viterbi()[0]

print "The number of wrong predicted motor primitives is %d" % \
      np.count_nonzero(regression_hidden_states - hidden_states_ground_truth)

considered_segments = min(len(regression_hidden_states), 5)  # a few segments
number_testing_points = 100

last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
means = np.zeros((considered_segments, number_testing_points))
for i in xrange(considered_segments):
    c_hidden_state = regression_hidden_states[i]
    c_obv = regression_observation[0][i]
    t_test = np.linspace(start_t, end_t, number_testing_points)  # predicting more time steps
    mean_pred, cov_pred = lfm_hmm.predict(t_test, c_hidden_state, c_obv)
    means[i, :] = mean_pred.flatten()
    sl = lfm_hmm.sample_locations
    plt.scatter(last_value + sl - sl[0], c_obv, facecolors='none',
                label=[None, 'observations'][i == 0])
    plt.plot(last_value + t_test - t_test[0], mean_pred, color = 'green',
             label=[None, 'predicted mean'][i == 0])
    diag_cov = np.diag(cov_pred)
    plt.plot(last_value + t_test - t_test[0], mean_pred.flatten() - 2 * np.sqrt(diag_cov), 'k--')
    plt.plot(last_value + t_test - t_test[0], mean_pred.flatten() + 2 * np.sqrt(diag_cov), 'k--')
    last_value = last_value + end_t - start_t
    plt.axvline(x=last_value, color='red', linestyle='--')


plt.title("Fitting of the model given an observation sequence.")
plt.legend(loc='upper left')
plt.show()

#print regression_hidden_states

# Third experiment

number_testing_points = 191

lfm_validation = LFMHMM(
    number_lfm,
    A,
    pi,
    outputs,
    start_t,
    end_t,
    number_testing_points,
    damper_constants,
    spring_constants,
    lengthscales,
    noise_var,
    verbose=True,
)

n_segments = 5
full_observation, reference_states = lfm_validation.generate_observations(n_segments)
sampled_observation = np.zeros((n_segments, 20))
for i in xrange(n_segments):
    for j in xrange(20):
        sampled_observation[i][j] = full_observation[i][j * 10]

obtained_states = lfm_hmm._viterbi([sampled_observation])


rmse = 0
for i in xrange(n_segments):
    c_hidden_state = obtained_states[0][i]
    t_test = np.linspace(start_t, end_t, number_testing_points)  # predicting more time steps
    mean_pred, cov_pred = lfm_hmm.predict(t_test, c_hidden_state,
                                          sampled_observation[i])
    rmse += np.power(mean_pred.flatten() - full_observation[i].flatten(), 2).sum()

print "The RMSE error in regression is %f" % np.sqrt(rmse)

# the same motor primitive were recovered.
















