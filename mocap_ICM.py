import GPy
from hmm.continuous.LFMHMMcontinuous import LFMHMMcontinuous
from hmm.continuous.ICMHMMcontinuousMO import ICMHMMcontinuousMO
from matplotlib import pyplot as plt
import numpy as np

seed = np.random.random_integers(10000)
np.random.seed(seed)

print("Using GPy version: ", GPy.__version__)

data = GPy.util.datasets.cmu_mocap('05', ['20'], sample_every=1)
print(data['info'])
Y = data['Y']
nsamples, nfeatures = Y.shape
print("Y's shape ", Y.shape)


channel_ids = [9, 16]  # ltibia and rtibia

for idx in channel_ids:
    plt.plot(np.arange(nsamples), Y[:, idx])
plt.show()

number_hidden_states = 3
outputs = len(channel_ids)
start_t = 0.1
end_t = 5.1
locations_per_segment = 20
n_latent_forces = 1  # TODO: currently not passing this argument to the model.

lfm_hmm = ICMHMMcontinuousMO(outputs, number_hidden_states, locations_per_segment, start_t,
                           end_t, verbose=True)

number_training_sequences = 1
obs = []
for s in range(number_training_sequences):
    number_segments = 55  # fixed for now.
    c_obs = np.zeros((number_segments, locations_per_segment * outputs))
    for output_id in range(outputs):
        signal = Y[:, channel_ids[output_id]]
        idx = 0
        for i in range(number_segments):
            c_obs[i, output_id::outputs] = signal[idx:idx + locations_per_segment]
            idx = idx + locations_per_segment - 1
    obs.append(c_obs)

lfm_hmm.set_observations(obs)

print("before training")
print(lfm_hmm.pi)
print(lfm_hmm.A)
print(lfm_hmm.ICMparams)

train_flag = False
file = "second-ICM-MOCAP-MO"
if train_flag:
    lfm_hmm.train()
    lfm_hmm.save_params("/home/diego/tmp/Parameters/MOCAP", file)
else:
    lfm_hmm.read_params("/home/diego/tmp/Parameters/MOCAP", file)

print("after training")
print(lfm_hmm.pi)
print(lfm_hmm.A)
print(lfm_hmm.ICMparams)


obs_1 = obs[0]
hidden_states_1 = lfm_hmm._viterbi()[0]

considered_segments = len(obs_1)
number_testing_points = 100
# prediction
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
for i in range(considered_segments):
    c_hidden_state = hidden_states_1[i]
    c_obv = obs_1[i]
    # predicting more time steps
    t_test = np.linspace(start_t, end_t, number_testing_points)
    mean_pred, cov_pred = lfm_hmm.predict(t_test, c_hidden_state, c_obv)
    mean_pred = mean_pred.flatten()
    cov_pred = np.diag(cov_pred)

    current_outputs = np.zeros((number_testing_points, outputs))
    current_covariances = np.zeros((number_testing_points, outputs))
    # separating the outputs accordingly.
    for j in range(outputs):
        current_outputs[:, j] = mean_pred[j::outputs]
        current_covariances[:, j] = cov_pred[j::outputs]

    sl = lfm_hmm.sample_locations
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

print("Inferred hidden states ", hidden_states_1)
print("USED SEED", seed)