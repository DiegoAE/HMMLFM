from cycler import cycler
from hmm.continuous.LFMHMM import LFMHMM
from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
from hmm.continuous.ICMHMMcontinuousMO import ICMHMMcontinuousMO
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

seed = np.random.random_integers(100000)
seed = 10000
np.random.seed(seed)
print("USED SEED", seed)

pi = np.array([0.3, 0.3, 0.4])
print("initial state distribution", pi)
A = np.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
print("hidden state transition matrix\n", A)

number_lfm = 3
outputs = 3
start_t = 0.1  # finding: it's problematic to choose 0 as starting point.
end_t = 5.1  # finding: it's problematic to choose long times.
# since the cov's tend to be the same.
locations_per_segment = 20
# list of lists in case of multiple outputs
damper_constants = np.asarray([[1., 3., 7.5], [3., 1, 0.5], [6., 5., 4.]])
spring_constants = np.asarray([[3., 1, 2.5], [1., 3, 9.0], [5., 6., 4.5]])
# damper_constants = np.random.rand(number_lfm, outputs) * 10.0
# spring_constants = np.random.rand(number_lfm, outputs) * 10.0
# implicitly assuming there is only one latent force governing the system.
lengthscales = np.asarray([[10.], [1.], [2.5]])
# it seems to be quite problematic when you choose big lenghtscales
noise_var = np.array([0.005, 0.005, 0.005])

lfm_hmm = LFMHMMcontinuousMO(outputs, number_lfm, locations_per_segment,
                             start_t, end_t, verbose=True)
lfm_hmm.set_params(A, pi, damper_constants, spring_constants, lengthscales,
                   noise_var)

# Plotting
fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', ['red', 'green', 'blue']))

segments = 10
obs_1, hidden_states_obs1 = lfm_hmm.generate_observations(segments)
last_value = 0
for i in range(segments):
    plt.axvline(x=last_value, color='red', linestyle='--')
    sl = lfm_hmm.sample_locations
    current_obs = obs_1[i]
    current_outputs = np.zeros((locations_per_segment, outputs))
    # separating the outputs accordingly.
    for j in range(outputs):
        current_outputs[:, j] = current_obs[j::outputs]
    plt.plot(last_value + sl - sl[0], current_outputs)
    last_value += end_t - start_t
plt.show()


icm_hmm = ICMHMMcontinuousMO(outputs, number_lfm,
                             locations_per_segment, start_t, end_t, verbose=True)

icm_hmm.set_observations([obs_1])

print(icm_hmm.pi)
print(icm_hmm.A)
print(icm_hmm.ICMparams)

print("start training")

train_flag = False
file_name = "MO-ICM-continuous-with-LFM-obs"
if train_flag:
    icm_hmm.train()
    icm_hmm.save_params("/home/diego/tmp/Parameters/ICM", file_name)
else:
    icm_hmm.read_params("/home/diego/tmp/Parameters/ICM", file_name)

print(icm_hmm.pi)
print(icm_hmm.A)
print(icm_hmm.ICMparams)

recovered_paths = icm_hmm._viterbi()
print(recovered_paths)


hidden_states_1 = recovered_paths[0]

considered_segments = segments
number_testing_points = 100
# prediction
last_value = 0
colors_cycle = ['red', 'green', 'blue']
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
                    color=colors_cycle[j])

    for j in range(outputs):
        plt.plot(last_value + t_test - t_test[0], current_outputs[:, j],
                 color=colors_cycle[j], label=[None, 'predicted mean'][i == 0])
    plt.plot(last_value + t_test - t_test[0],
             current_outputs - 2 * np.sqrt(current_covariances), 'k--')
    plt.plot(last_value + t_test - t_test[0],
             current_outputs + 2 * np.sqrt(current_covariances), 'k--')
    last_value = last_value + end_t - start_t
    plt.axvline(x=last_value, color='red', linestyle='--')
plt.show()

print("USED SEED", seed)



