__author__ = 'diego'

from cycler import cycler
from hmm.continuous.ICMHMMcontinuousMO import ICMHMMcontinuousMO
from matplotlib import pyplot as plt
import numpy as np

seed = np.random.random_integers(10000)
seed = 5879
np.random.seed(seed)
print "USED SEED", seed

pi = np.array([0.3, 0.3, 0.4])
print "initial state distribution", pi
A = np.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
print "hidden state transition matrix\n", A

number_hidden_states = 3
outputs = 2
start_t = 0.1
end_t = 5.1
locations_per_segment = 20
rbf_variances = np.ones(number_hidden_states) * 2
rbf_lengthscales = np.ones(number_hidden_states)
B_Ws = np.array([[0., 0.], [1., 1.], [1., -1.]])
kappas = 0.5 * np.ones((number_hidden_states, outputs))
noise_var = np.array([0.0005, 0.0005])

icm_hmm = ICMHMMcontinuousMO(outputs, number_hidden_states, locations_per_segment, start_t,
                           end_t, verbose=True)
icm_hmm.set_params(A, pi, rbf_variances, rbf_lengthscales, B_Ws, kappas,
                   noise_var)

# Testing packing and unpacking
print "Testing packing and unpacking: ",
test_unit_1 = icm_hmm.unpack_params(icm_hmm.pack_params(icm_hmm.ICMparams))
test_unit_2 = icm_hmm.ICMparams
for k in test_unit_2.keys():
    assert np.allclose(test_unit_2[k], test_unit_1[k])
print "Accepted!"

for i in xrange(number_hidden_states):
    print icm_hmm.icms[i].icm_kernel

# Plotting

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', ['red', 'green']))

segments = 10
obs_1, hidden_states_1 = icm_hmm.generate_observations(segments)
last_value = 0
for i in xrange(segments):
    plt.axvline(x=last_value, color='red', linestyle='--')
    sl = icm_hmm.sample_locations
    current_obs = obs_1[i]
    current_outputs = np.zeros((locations_per_segment, outputs))
    # separating the outputs accordingly.
    for j in xrange(outputs):
        current_outputs[:, j] = current_obs[j::outputs]
    plt.plot(last_value + sl - sl[0], current_outputs)
    last_value += end_t - start_t
plt.show()

obs = []
n_training_sequences = 10
hidden_states = np.zeros(n_training_sequences, dtype=object)
for i in xrange(n_training_sequences):
    segments = np.random.randint(1, 100)
    print "The %d-th sequence has length %d" % (i, segments)
    output, hidden = icm_hmm.generate_observations(segments)
    obs.append(output)
    hidden_states[i] = hidden

icm_hmm.set_observations(obs)
icm_hmm.reset()

print icm_hmm.pi
print icm_hmm.A
print icm_hmm.ICMparams

print "start training"

train_flag = False
file_name = "First-MO-ICM-continuous"
if train_flag:
    icm_hmm.train()
    icm_hmm.save_params("/home/diego/tmp/Parameters/ICM", file_name)
else:
    icm_hmm.read_params("/home/diego/tmp/Parameters/ICM", file_name)

print icm_hmm.pi
print icm_hmm.A
print icm_hmm.ICMparams

recovered_paths = icm_hmm._viterbi()
print recovered_paths

obs_1 = obs[1]
hidden_states_1 = recovered_paths[1]

considered_segments = 5
number_testing_points = 100
# prediction
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
for i in xrange(considered_segments):
    c_hidden_state = hidden_states_1[i]
    c_obv = obs_1[i]
    # predicting more time steps
    t_test = np.linspace(start_t, end_t, number_testing_points)
    mean_pred, cov_pred = icm_hmm.predict(t_test, c_hidden_state, c_obv)
    mean_pred = mean_pred.flatten()
    cov_pred = np.diag(cov_pred)

    current_outputs = np.zeros((number_testing_points, outputs))
    current_covariances = np.zeros((number_testing_points, outputs))
    # separating the outputs accordingly.
    for j in xrange(outputs):
        current_outputs[:, j] = mean_pred[j::outputs]
        current_covariances[:, j] = cov_pred[j::outputs]

    sl = icm_hmm.sample_locations
    for j in xrange(outputs):
        plt.scatter(last_value + sl - sl[0], c_obv[j::outputs],
                    facecolors='none', label=[None, 'observations'][i == 0])

    plt.plot(last_value + t_test - t_test[0], current_outputs, color='green',
             label=[None, 'predicted mean'][i == 0])
    diag_cov = np.diag(cov_pred)
    plt.plot(last_value + t_test - t_test[0],
             current_outputs - 2 * np.sqrt(current_covariances), 'k--')
    plt.plot(last_value + t_test - t_test[0],
             current_outputs + 2 * np.sqrt(current_covariances), 'k--')
    last_value = last_value + end_t - start_t
    plt.axvline(x=last_value, color='red', linestyle='--')
plt.show()

print "USED SEED", seed
