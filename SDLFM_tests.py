from hmm.continuous.LFMHMMcontinuous import LFMHMMcontinuous
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio

seed = np.random.random_integers(10000)
# seed = 4748
np.random.seed(seed)
print("USED SEED", seed)

### LFM HMM
number_lfm = 3
outputs = 1
start_t = 0.1
end_t = 10.1
locations_per_segment = 201
n_latent_forces = 1  # TODO: currently not passing this argument to the model.

lfm_hmm = LFMHMMcontinuous(outputs, number_lfm, locations_per_segment, start_t,
                           end_t, verbose=True)

mat_file = sio.loadmat('Samples.mat')

x = mat_file['XTest'][0][-1]  # all sample locations
n_samples = np.size(x)
picked_sample = 0
f = mat_file['yTest'][0][-1][0][picked_sample]
n_outputs = f.shape[1]
Y = np.zeros((n_samples, n_outputs))
for i in range(n_outputs):
    Y[:, i] = f[0][i].flatten()


# testing_idx = 62
# print "X ", x[0][testing_idx]
# print "Y ", obs[testing_idx, 0], obs[testing_idx, 1]
#
# testing_idx = 63
# print "X ", x[0][testing_idx]
# print "Y ", obs[testing_idx, 0], obs[testing_idx, 1]

plt.plot(x.flatten(), Y)
for i in range(1, 6):
    plt.axvline(x=10 * i, color='red', linestyle='--')
plt.show()


# Setting observations in the model.
channel_id = 0
number_training_sequences = 1
obs = []
for s in range(number_training_sequences):
    number_segments = 6  # fixed for now.
    c_obs = np.zeros((number_segments, locations_per_segment))
    signal = Y[:, channel_id]
    idx = 0
    for i in range(number_segments):
        c_obs[i, :] = signal[idx:idx + locations_per_segment]
        idx = idx + locations_per_segment - 1
    obs.append(c_obs)
lfm_hmm.set_observations(obs)


print("before training")
print(lfm_hmm.pi)
print(lfm_hmm.A)
print(lfm_hmm.LFMparams)

train_flag = False
if train_flag:
    lfm_hmm.train()
    lfm_hmm.save_params("/home/diego/tmp/Parameters", "pruebaSDLFM_1")
else:
    lfm_hmm.read_params("/home/diego/tmp/Parameters", "pruebaSDLFM_1")

print("after training")
print(lfm_hmm.pi)
print(lfm_hmm.A)
print(lfm_hmm.LFMparams)

# Second experiment: Regression
number_testing_points = 100
regression_hidden_states = lfm_hmm._viterbi()[0]
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
considered_segments = 6
for i in range(considered_segments):
    c_hidden_state = regression_hidden_states[i]
    c_obv = obs[0][i]
    # predicting more time steps
    t_test = np.linspace(start_t, end_t, number_testing_points)
    mean_pred, cov_pred = lfm_hmm.predict(t_test, c_hidden_state, c_obv)
    sl = lfm_hmm.sample_locations
    plt.scatter(last_value + sl - sl[0], c_obv, facecolors='none',
                label=[None, 'observations'][i == 0])
    plt.plot(last_value + t_test - t_test[0], mean_pred, color='green',
             label=[None, 'predicted mean'][i == 0])
    diag_cov = np.diag(cov_pred)
    plt.plot(last_value + t_test - t_test[0], mean_pred.flatten() - 2 * np.sqrt(diag_cov), 'k--')
    plt.plot(last_value + t_test - t_test[0], mean_pred.flatten() + 2 * np.sqrt(diag_cov), 'k--')
    last_value = last_value + end_t - start_t
    plt.axvline(x=last_value, color='red', linestyle='--')


print("Inferred hidden states ", regression_hidden_states)

plt.title("Fitting of the model given an observation sequence.")
plt.legend(loc='upper left')
plt.show()


# Plotting the priors
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
for i in range(considered_segments):
    c_hidden_state = regression_hidden_states[i]
    # predicting more time steps
    t_test = np.linspace(start_t, end_t, locations_per_segment)
    mean_prior = np.zeros(len(t_test))
    cov_prior = lfm_hmm.lfms[c_hidden_state].Kyy()
    plt.plot(last_value + t_test - t_test[0], mean_prior, color='green')
    diag_cov = np.diag(cov_prior)
    plt.plot(last_value + t_test - t_test[0], mean_prior.flatten() - 2 * np.sqrt(diag_cov), 'k--')
    plt.plot(last_value + t_test - t_test[0], mean_prior.flatten() + 2 * np.sqrt(diag_cov), 'k--')
    last_value = last_value + end_t - start_t
    plt.axvline(x=last_value, color='red', linestyle='--')

plt.title("Plotting priors.")
plt.show()









