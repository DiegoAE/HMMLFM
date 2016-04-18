from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
from matplotlib import pyplot as plt
import numpy as np


seed = np.random.random_integers(100000)
seed = 47859
np.random.seed(seed)
print "USED SEED", seed


input_file = file('mocap_walking_subject_07.npz', 'rb')
data = np.load(input_file)
training_observations = data['training']
testing_observations = data['test']
outputs = data['outputs']
locations_per_segment = data['lps']

number_lfm = 3
start_t = 0.1
end_t = 5.1
model = LFMHMMcontinuousMO(outputs, number_lfm, locations_per_segment,
                             start_t, end_t, verbose=True)

model.set_observations(training_observations)


print model.pi
print model.A
print model.LFMparams

print "start training"

train_flag = False
if train_flag:
    model.train()
    model.save_params("/home/diego/tmp/Parameters/WALKING", "MOToy")
else:
    model.read_params("/home/diego/tmp/Parameters/WALKING", "MOToy")


print "after training"
print model.pi
print model.A
print model.LFMparams

print "USED SEED", seed

viterbi_training =  model._viterbi()
print viterbi_training

# Testing data

# print model._viterbi(testing_observations)

# Looking at the resulting fit


number_testing_points = 100

colors_cycle = ['red', 'green', 'blue', 'purple']
regression_hidden_states = viterbi_training[0]
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
considered_segments = training_observations[0].shape[0]
print considered_segments
for i in xrange(considered_segments):
    c_hidden_state = regression_hidden_states[i]
    c_obv = training_observations[0][i]
    # predicting more time steps
    t_test = np.linspace(start_t, end_t, number_testing_points)
    mean_pred, cov_pred = model.predict(t_test, c_hidden_state, c_obv)
    mean_pred = mean_pred.flatten()
    cov_pred = np.diag(cov_pred)

    current_outputs = np.zeros((number_testing_points, outputs))
    current_covariances = np.zeros((number_testing_points, outputs))
    # separating the outputs accordingly.
    for j in xrange(outputs):
        current_outputs[:, j] = mean_pred[j::outputs]
        current_covariances[:, j] = cov_pred[j::outputs]

    # NOTE: there is an important distinction between sample locations
    # for evaluation and for plotting because different spaces are being used.
    # Particularly, in the plotting space each sample is a unit away from each
    # other. On the other hand, evaluation locations depend on start_t and end_t

    obs_plotting_locations = last_value + np.linspace(
            0, model.locations_per_segment - 1, model.locations_per_segment)
    for j in xrange(outputs):
        plt.scatter(obs_plotting_locations, c_obv[j::outputs],
                    color=colors_cycle[j])
    test_plotting_locations = last_value + np.linspace(
            0, model.locations_per_segment - 1, number_testing_points)
    for j in xrange(outputs):
        plt.plot(test_plotting_locations, current_outputs[:, j],
                 color=colors_cycle[j], label=[None, 'predicted mean'][i == 0])
    plt.plot(test_plotting_locations,
             current_outputs - 2 * np.sqrt(current_covariances), 'k--')
    plt.plot(test_plotting_locations,
             current_outputs + 2 * np.sqrt(current_covariances), 'k--')
    last_value = last_value + model.locations_per_segment - 1
    plt.axvline(x=last_value, color='red', linestyle='--')
plt.show()



