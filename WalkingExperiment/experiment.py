from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
from hmm.continuous.ICMHMMcontinuousMO import ICMHMMcontinuousMO
from hmm.continuous.LFMHMMTyingMO import LFMHMMTyingMO
from matplotlib import pyplot as plt
import numpy as np


seed = np.random.random_integers(100000)
seed = 79861  # LFM
# seed = 8629  # ICM
np.random.seed(seed)
print "USED SEED", seed

def pick_outputs(input, noutputs, required_outputs):
    output = input.copy()
    for i in xrange(len(output)):
        output_idx = []
        total_cols = output[i].shape[1]
        for idx in required_outputs:
            output_idx.extend(range(idx, total_cols, noutputs))
        output_idx.sort()
        output[i] = output[i][:, output_idx]
    return output

input_file = file('mocap_walking_subject_07.npz', 'rb')
# input_file = file('toy_lfm.npz', 'rb')
data = np.load(input_file)
outputs = data['outputs'].item()
training_observations = data['training']
testing_observations = data['test']
locations_per_segment = data['lps']
# Picking outputs
# training_observations = pick_outputs(training_observations, outputs, [0])
# testing_observations = pick_outputs(testing_observations, outputs, [0])
# outputs = 1
#

number_lfm = 3
number_latent_forces = 3
start_t = 0.1
end_t = 5.1
model = LFMHMMcontinuousMO(outputs, number_lfm, locations_per_segment,
                             start_t, end_t, number_latent_forces, verbose=True)

colors_cycle = ['red', 'green', 'blue', 'purple']
joints_name = ['left tibia', 'right tibia', 'left radius', 'right radius']
# Plotting the inputs
# considered_idx = 0
# last_value = 0
# plt.axvline(x=last_value, color='red', linestyle='--')
# for i in xrange(len(training_observations[considered_idx])):
#     c_obv = training_observations[considered_idx][i]
#     obs_plotting_locations = last_value + np.linspace(
#             0, model.locations_per_segment - 1, model.locations_per_segment)
#     for j in xrange(outputs):
#         plt.scatter(obs_plotting_locations, c_obv[j::outputs],
#                     color=colors_cycle[j])
#     last_value = last_value + model.locations_per_segment - 1
#     plt.axvline(x=last_value, color='red', linestyle='--')
# plt.show()
# end plotting

model.set_observations(training_observations)
# model.read_params("/home/diego/tmp/Parameters/WALKING", "MANUAL_INIT")

print model.pi
print model.A
print model.LFMparams

print "start training"

train_flag = False
file_name = "MO_MOCAP_3_forces"
if train_flag:
    model.train()
    model.save_params("/home/diego/tmp/Parameters/WALKING", file_name)
else:
    model.read_params("/home/diego/tmp/Parameters/WALKING", file_name)


print "after training"
print model.pi
print model.A
print model.LFMparams

print "USED SEED", seed

viterbi_training =  model._viterbi()
print "Viterbi"
print viterbi_training

# Testing data

viterbi_testing = model._viterbi(testing_observations)
print "Viterbi for testing"
print viterbi_testing

# Looking at the resulting fit

number_testing_points = 100

considered_idx = 0
regression_hidden_states = viterbi_testing[considered_idx]
# print regression_hidden_states
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
considered_segments = testing_observations[considered_idx].shape[0]
# print considered_segments
for i in xrange(considered_segments):
    c_hidden_state = regression_hidden_states[i]
    plt.text(1 + i * 20 - i, .75, r'$z_{%d}=%d$' % (i, c_hidden_state),
             fontsize=23)
    c_obv = testing_observations[considered_idx][i]
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
                    color=colors_cycle[j], label=[None, joints_name[j]][i == 0])
    test_plotting_locations = last_value + np.linspace(
            0, model.locations_per_segment - 1, number_testing_points)
    for j in xrange(outputs):
        plt.plot(test_plotting_locations, current_outputs[:, j],
                 color=colors_cycle[j])
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

