import GPy
from hmm.continuous.LFMHMMcontinuous import LFMHMMcontinuous
from matplotlib import pyplot as plt
import numpy as np

seed = np.random.random_integers(10000)
seed = 4748
np.random.seed(seed)
print "USED SEED", seed

print "Using GPy version: ", GPy.__version__

data = GPy.util.datasets.cmu_mocap('5', ['13'], sample_every=1)
print data['info']
Y = data['Y']
nsamples, nfeatures = Y.shape
print "Y's shape ", Y.shape


channel_id = 20

plt.plot(np.arange(nsamples), Y[:, channel_id])
plt.show()

# print Y[0, :]


# Visualisation not working for now.
# Y[:, 0:3] = 0.   # Make figure walk in place
# visualize = GPy.util.visualize.skeleton_show(Y[0, :], data['skel'])
# GPy.util.visualize.data_play(Y, visualize)


### LFM HMM
number_lfm = 3
outputs = 1
start_t = 0.1
end_t = 5.1
locations_per_segment = 20
n_latent_forces = 1  # currently not passing this argument to the model.

# TODO: Refactor the __init__ method of the model in such a way that you create
# the model without explicit parameters and they get filled with random values.
# Dummy initial values for model parameters.
pi = np.zeros(number_lfm)
A = np.zeros((number_lfm, number_lfm))
damper_constants = np.ones((number_lfm, outputs))
spring_constants = np.ones((number_lfm, outputs))
lengthscales = np.ones((number_lfm, n_latent_forces))
noise_var = 0.0005

lfm_hmm = LFMHMMcontinuous(
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
lfm_hmm.reset()

number_training_sequences = 1
obs = []
for s in xrange(number_training_sequences):
    number_segments = 30  # fixed for now.
    c_obs = np.zeros((number_segments, locations_per_segment))
    signal = Y[:, channel_id]
    idx = 0
    for i in xrange(number_segments):
        c_obs[i, :] = signal[idx:idx + locations_per_segment]
        idx = idx + locations_per_segment - 1
    obs.append(c_obs)

lfm_hmm.set_observations(obs)

print "before training"
print lfm_hmm.pi
print lfm_hmm.A
print lfm_hmm.LFMparams

train_flag = False
if train_flag:
    lfm_hmm.train()
    lfm_hmm.save_params("/home/diego/tmp/Parameters", "pruebaMOCAP")
else:
    lfm_hmm.read_params("/home/diego/tmp/Parameters", "pruebaMOCAP")

print "after training"
print lfm_hmm.pi
print lfm_hmm.A
print lfm_hmm.LFMparams


# Second experiment: Regression
number_testing_points = 100
regression_hidden_states = lfm_hmm._viterbi()[0]
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
considered_segments = 30
for i in xrange(considered_segments):
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


print "Inferred hidden states ", regression_hidden_states

plt.title("Fitting of the model given an observation sequence.")
plt.legend(loc='upper left')
plt.show()



