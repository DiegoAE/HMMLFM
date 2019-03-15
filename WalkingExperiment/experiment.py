from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
from hmm.continuous.ICMHMMcontinuousMO import ICMHMMcontinuousMO
from hmm.continuous.LFMHMMTyingMO import LFMHMMTyingMO
from matplotlib import pyplot as plt
import numpy as np
import os

seed = np.random.random_integers(100000)
seed = 79861
np.random.seed(seed)
print("USED SEED", seed)

def pick_outputs(input, noutputs, required_outputs):
    output = input.copy()
    for i in range(len(output)):
        output_idx = []
        total_cols = output[i].shape[1]
        for idx in required_outputs:
            output_idx.extend(list(range(idx, total_cols, noutputs)))
        output_idx.sort()
        output[i] = output[i][:, output_idx]
    return output

file_path = os.path.join(os.path.dirname(__file__),
                         'mocap_walking_subject_07.npz')
data = np.load(file_path, mmap_mode='rb', encoding='bytes')
outputs = data['outputs'].item()
training_observations = data['training']
testing_observations = data['test']
locations_per_segment = data['lps']

number_lfm = 3
number_latent_forces = 3
start_t = 0.1
end_t = 5.1
model = LFMHMMcontinuousMO(outputs, number_lfm, locations_per_segment,
                             start_t, end_t, number_latent_forces, verbose=True)

colors_cycle = ['red', 'green', 'blue', 'purple']
joints_name = ['left tibia', 'right tibia', 'left radius', 'right radius']

model.set_observations(training_observations)

train_flag = False

PRETRAINED_MODElS_DIRECTORY = os.path.join(os.path.dirname(__file__),
                                           '../PretrainedModels')
file_name = "MO_MOCAP_3_forces"
if train_flag:
    print("Start training")
    model.train()
    model.save_params(PRETRAINED_MODElS_DIRECTORY + "/WALKING", file_name)
else:
    print("Loading a pretrained model.")
    model.read_params(PRETRAINED_MODElS_DIRECTORY + "/WALKING", file_name)

print("USED SEED", seed)

viterbi_training =  model._viterbi()
print("Viterbi")
print(viterbi_training)

# Testing data
# With a number of key points equal to 10 we have the whole set of observations.
# By reducing it we shorten the observations just leaving a prefix and a suffix
# of the same length.
number_key_points = 10
test_sample_locations = np.hstack((model.sample_locations[:number_key_points],
                                   model.sample_locations[-number_key_points:]))

new_testing_observations = np.zeros(testing_observations.size, dtype='object')

for observation_idx in range(testing_observations.size):
    current_ob = testing_observations[observation_idx]
    new_testing_observations[observation_idx] = np.zeros(
            (current_ob.shape[0], test_sample_locations.size * outputs))
    for segment_idx in range(current_ob.shape[0]):
        segment = current_ob[segment_idx]
        new_testing_observations[observation_idx][segment_idx] = np.hstack(
                (segment[:number_key_points * outputs],
                 segment[-number_key_points * outputs:])
        )

model.set_sample_locations(test_sample_locations)
viterbi_testing = model._viterbi(new_testing_observations)
print("Viterbi for testing")
print(viterbi_testing)

# Looking at the resulting fit

number_testing_points = 100

considered_idx = 0
regression_hidden_states = viterbi_testing[considered_idx]
last_value = 0
plt.axvline(x=last_value, color='red', linestyle='--')
considered_segments = testing_observations[considered_idx].shape[0]
# print considered_segments
for i in range(considered_segments):
    c_hidden_state = regression_hidden_states[i]
    plt.text(1 + i * 20 - i, .75, r'$z_{%d}=%d$' % (i, c_hidden_state),
             fontsize=23)
    c_obv = new_testing_observations[considered_idx][i]
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
            0, locations_per_segment - 1, locations_per_segment)
    obs_plotting_locations = np.hstack(
            (obs_plotting_locations[:number_key_points],
             obs_plotting_locations[-number_key_points:])
    )
    for j in range(outputs):
        plt.scatter(obs_plotting_locations, c_obv[j::outputs],
                    color=colors_cycle[j], label=[None, joints_name[j]][i == 0])
    test_plotting_locations = last_value + np.linspace(
            0, locations_per_segment - 1, number_testing_points)
    for j in range(outputs):
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
for i in range(model.n):
    plt.subplot(1, model.n, i + 1)
    cov_m = model.get_cov_function(i, False)
    plt.imshow(transform_covariance(cov_m))
    plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
