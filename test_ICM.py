__author__ = 'diego'

from cycler import cycler
from hmm.continuous.ICMHMMcontinuousMO import ICMHMMcontinuousMO
from matplotlib import pyplot as plt
import numpy as np

seed = np.random.random_integers(10000)
seed = 5879
np.random.seed(seed)
print("USED SEED", seed)

pi = np.array([0.3, 0.3, 0.4])
print("initial state distribution", pi)
A = np.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
print("hidden state transition matrix\n", A)

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
print("Testing packing and unpacking: ", end=' ')
test_unit_1 = icm_hmm.unpack_params(icm_hmm.pack_params(icm_hmm.ICMparams))
test_unit_2 = icm_hmm.ICMparams
for k in list(test_unit_2.keys()):
    assert np.allclose(test_unit_2[k], test_unit_1[k])
print("Accepted!")

for i in range(number_hidden_states):
    print(icm_hmm.icms[i].icm_kernel)

# Plotting

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', ['red', 'green']))

segments = 10
obs_1, hidden_states_1 = icm_hmm.generate_observations(segments)
last_value = 0
for i in range(segments):
    plt.axvline(x=last_value, color='red', linestyle='--')
    sl = icm_hmm.sample_locations
    current_obs = obs_1[i]
    current_outputs = np.zeros((locations_per_segment, outputs))
    # separating the outputs accordingly.
    for j in range(outputs):
        current_outputs[:, j] = current_obs[j::outputs]
    plt.plot(last_value + sl - sl[0], current_outputs)
    last_value += end_t - start_t
plt.show()

obs = []
n_training_sequences = 10
hidden_states = np.zeros(n_training_sequences, dtype=object)
for i in range(n_training_sequences):
    segments = np.random.randint(1, 100)
    print("The %d-th sequence has length %d" % (i, segments))
    output, hidden = icm_hmm.generate_observations(segments)
    obs.append(output)
    hidden_states[i] = hidden

icm_hmm.set_observations(obs)
icm_hmm.reset()

print(icm_hmm.pi)
print(icm_hmm.A)
print(icm_hmm.ICMparams)

print("start training")

train_flag = False
file_name = "First-MO-ICM-continuous"
if train_flag:
    icm_hmm.train()
    icm_hmm.save_params("/home/diego/tmp/Parameters/ICM", file_name)
else:
    icm_hmm.read_params("/home/diego/tmp/Parameters/ICM", file_name)

dummy_model = ICMHMMcontinuousMO(outputs, number_hidden_states, locations_per_segment, start_t,
                           end_t, verbose=True)
dummy_model.set_params(A, pi, rbf_variances, rbf_lengthscales, B_Ws, kappas,
                   noise_var)

# plotting covariances

def transform_covariance(cov):
    ret = cov.copy()
    rows, cols = cov.shape
    lps = locations_per_segment
    for r in range(rows):
        for o in range(outputs):
            ret[r][lps * o:lps * (o + 1)] = cov[r][o::outputs]
    nret = ret.copy()
    for o in range(outputs):
        nret[lps * o:lps * (o + 1)] = ret[o::outputs]
    return nret

plt.figure()
for i in range(icm_hmm.n):
    if (i == 0):
        plt.xlabel("hola")
    plt.subplot(2, 3, i + 1)
    plt.imshow(transform_covariance(dummy_model.get_cov_function(i, False)))
    if i == 1:
        plt.title('(a)')
    plt.axis('off')
    plt.subplot(2, 3, i + 4)
    plt.imshow(transform_covariance(icm_hmm.get_cov_function(i, False)))
    if i == 1:
        plt.title('(b)')
    plt.axis('off')
plt.show()

print(icm_hmm.pi)
print(icm_hmm.A)
print(icm_hmm.ICMparams)

recovered_paths = icm_hmm._viterbi()
print(recovered_paths)

obs_1 = obs[1]
hidden_states_1 = recovered_paths[1]

considered_segments = 5
number_testing_points = 100
# prediction
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
for i in range(considered_segments):
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
    for j in range(outputs):
        current_outputs[:, j] = mean_pred[j::outputs]
        current_covariances[:, j] = cov_pred[j::outputs]

    sl = icm_hmm.sample_locations
    for j in range(outputs):
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

print("USED SEED", seed)

viterbi_training = recovered_paths
# This is only useful for synthetic experiments.

def f(a):
    if a == 0:
        return 2
    if a == 1:
        return 0
    return 1

TMP = hidden_states
print("Training Vit")
malos = 0
totales = 0
for i in range(len(TMP)):
    # print map(f, TMP[i])
    prueba = np.array(list(map(f, TMP[i])))
    diff = prueba - viterbi_training[i]
    malos += np.count_nonzero(diff)
    totales += np.size(diff)
    print(diff)

print(malos, totales)

colors_cycle = ['red', 'green', 'blue', 'purple']
labels = ["output 1", "output 2"]
considered_idx = 0
regression_hidden_states = viterbi_training[considered_idx]
# print regression_hidden_states
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
considered_segments = 10
# print considered_segments
for i in range(considered_segments):
    model = icm_hmm
    c_hidden_state = regression_hidden_states[i]
    plt.text(1 + i * 20 - i, 8., r'$z_{%d}=%d$' % (i, c_hidden_state),
             fontsize=23)
    c_obv = obs[considered_idx][i]
    # predicting more time steps
    t_test = np.linspace(start_t, end_t, number_testing_points)
    mean_pred, cov_pred = model.predict(t_test, c_hidden_state, c_obv)
    mean_pred = mean_pred.flatten()
    cov_pred = np.diag(cov_pred)

    current_outputs = np.zeros((number_testing_points, outputs))
    current_covariances = np.zeros((number_testing_points, outputs))
    # separating the outputs accordingly.
    for j in range(outputs):
        current_outputs[:, j] = mean_pred[j::outputs]
        current_covariances[:, j] = cov_pred[j::outputs]

    # NOTE: there is an important distinction between sample locations
    # for evaluation and for plotting because different spaces are being used.
    # Particularly, in the plotting space each sample is a unit away from each
    # other. On the other hand, evaluation locations depend on start_t and end_t

    obs_plotting_locations = last_value + np.linspace(
            0, model.locations_per_segment - 1, model.locations_per_segment)
    for j in range(outputs):
        plt.scatter(obs_plotting_locations, c_obv[j::outputs],
                    color=colors_cycle[j],
                    label=[None, 'output %d' % (j + 1)][i == 0])
    test_plotting_locations = last_value + np.linspace(
            0, model.locations_per_segment - 1, number_testing_points)
    for j in range(outputs):
        plt.plot(test_plotting_locations, current_outputs[:, j],
                 color=colors_cycle[j],
        )
        lower_trajectory = current_outputs[:, j] -\
                           2 * np.sqrt(current_covariances[:, j])
        upper_trajectory = current_outputs[:, j] +\
                           2 * np.sqrt(current_covariances[:, j])
        plt.fill_between(test_plotting_locations, lower_trajectory,
                         upper_trajectory, alpha=0.4, facecolor=colors_cycle[j])
    last_value = last_value + model.locations_per_segment - 1
    plt.axvline(x=last_value, color='red', linestyle='--')
plt.legend(loc='lower left')
plt.show()
