import GPy
from hmm.continuous.LFMHMMcontinuous import LFMHMMcontinuous
from matplotlib import pyplot as plt
import numpy as np

seed = np.random.random_integers(10000)
seed = 4748
np.random.seed(seed)

print("Using GPy version: ", GPy.__version__)

data = GPy.util.datasets.cmu_mocap('43', ['01'], sample_every=1)
print(data['info'])
Y = data['Y'][70:, :]
nsamples, nfeatures = Y.shape
print("Y's shape ", Y.shape)


channel_id = 9

plt.plot(np.arange(nsamples), Y[:, channel_id])
plt.show()

### LFM HMM
number_lfm = 7
outputs = 1
start_t = 0.1
end_t = 5.1
locations_per_segment = 20
n_latent_forces = 1  # TODO: currently not passing this argument to the model.

lfm_hmm = LFMHMMcontinuous(outputs, number_lfm, locations_per_segment, start_t,
                           end_t, verbose=True)

number_training_sequences = 1
obs = []
for s in range(number_training_sequences):
    number_segments = 18  # fixed for now.
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
    lfm_hmm.save_params("/home/diego/tmp/Parameters/MOCAP", "pruebaMOCAP")
else:
    lfm_hmm.read_params("/home/diego/tmp/Parameters/MOCAP", "pruebaMOCAP")

print("after training")
print(lfm_hmm.pi)
print(lfm_hmm.A)
print(lfm_hmm.LFMparams)


# Second experiment: Regression
number_testing_points = 100
regression_hidden_states = lfm_hmm._viterbi()[0]
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
considered_segments = 18  # fixed for now.
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

print("USED SEED", seed)