from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
from hmm.continuous.ICMHMMcontinuousMO import ICMHMMcontinuousMO
from hmm.continuous.LFMHMMTyingMO import LFMHMMTyingMO
from matplotlib import pyplot as plt
import copy
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

viterbi_testing = model._viterbi(testing_observations)
print "Viterbi for testing"
print viterbi_testing

#

def explicit_predict_wrapper(input_model, t, ind, y, hidden_state, ts, inds,
                             offset):
    """
    Assuming that y contain all but the one output.
    This was originally in LFMHMMcontinuousMO but I think it makes dirty the
    API since it's a very specific use case.
    """
    obs = copy.deepcopy(y)
    current_shift = obs[:outputs - 1].copy()
    for j in xrange(outputs - 1):
        obs[j::outputs - 1] -= current_shift[j]
    mean_pred, cov_pred = input_model.explicit_predict(
            t, ind, obs, hidden_state, ts, inds)
    mean_pred += offset  # prediction over a single output
    return mean_pred, cov_pred

# Looking at the resulting fit

number_testing_points = 100

considered_idx = 0
regression_hidden_states = viterbi_testing[considered_idx]
# print regression_hidden_states
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
considered_segments = testing_observations[considered_idx].shape[0]
# print considered_segments
mean_missing_output = 0.0
chosen_output = 3
plt.figure(1)
for i in xrange(considered_segments):
    c_hidden_state = regression_hidden_states[i]
    # plt.text(1 + i * 20 - i, .75, r'$z_{%d}=%d$' % (i, c_hidden_state),
    #          fontsize=23)
    c_obv = testing_observations[considered_idx][i]
    # predicting more time steps
    sample_test_locations = np.linspace(start_t, end_t, number_testing_points)

    t, ind = model.lfms[c_hidden_state].get_stacked_time_and_indexes(
            model.sample_locations)
    ts, inds = sample_test_locations.reshape((-1, 1)),\
               np.ones((sample_test_locations.size, 1),
                       dtype='int') * chosen_output

    # removing the last output's sample locations and observations
    t = np.delete(t, np.s_[chosen_output::outputs], axis=0)
    ind = np.delete(ind, np.s_[chosen_output::outputs], axis=0)
    cut_obs = np.delete(c_obv, np.s_[chosen_output::outputs])

    # mean_pred, cov_pred = model.predict(t_test, c_hidden_state, c_obv)
    mean_pred, cov_pred = explicit_predict_wrapper(
            model, t, ind, cut_obs, c_hidden_state, ts, inds,
            mean_missing_output)

    mean_pred = mean_pred.flatten()
    cov_pred = np.diag(cov_pred)
    mean_missing_output = mean_pred[-1]

    # NOTE: there is an important distinction between sample locations
    # for evaluation and for plotting because different spaces are being used.
    # Particularly, in the plotting space each sample is a unit away from each
    # other. On the other hand, evaluation locations depend on start_t and end_t

    obs_plotting_locations = last_value + np.linspace(
            0, locations_per_segment - 1, locations_per_segment)

    # for j in xrange(outputs):
    #     plt.scatter(obs_plotting_locations, c_obv[j::outputs],
    #                 color=colors_cycle[j], label=[None, joints_name[j]][i == 0])
    plt.subplot(2, 1, 1)
    plt.xlim((0, 250))
    plt.scatter(obs_plotting_locations, c_obv[chosen_output::outputs],
                color=colors_cycle[chosen_output],
                label=[None, joints_name[chosen_output]][i == 0])
    test_plotting_locations = last_value + np.linspace(
            0, locations_per_segment - 1, number_testing_points)
    last_value = last_value + model.locations_per_segment - 1
    plt.axvline(x=last_value, color='red', linestyle='--')
    plt.legend(loc='lower left')
    plt.subplot(2, 1, 2)
    plt.xlim((0, 250))
    plt.plot(test_plotting_locations, mean_pred,
             color=colors_cycle[chosen_output],
             label=[None, 'predictive mean'][i == 0])
    lower_trajectory = mean_pred - 2 * np.sqrt(cov_pred)
    upper_trajectory = mean_pred + 2 * np.sqrt(cov_pred)
    plt.fill_between(test_plotting_locations, lower_trajectory,
                     upper_trajectory, alpha=0.4,
                     facecolor=colors_cycle[chosen_output])
    plt.axvline(x=last_value, color='red', linestyle='--')
    plt.legend(loc='lower left')
plt.savefig("missing_output_%d.png" % chosen_output, bbox_inches='tight')
plt.savefig("missing_output_%d.eps" % chosen_output, dpi=1000,
            bbox_inches='tight')
plt.show()
