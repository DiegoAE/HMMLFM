import GPy
from matplotlib import pyplot as plt
import numpy as np

motions_for_training = ['04', '05', '06']
offsets = [60, 35, 20]
limits = [450, 450, 450]

data = []
for motion in motions_for_training:
    data.append(GPy.util.datasets.cmu_mocap(
            subject='41', train_motions=[motion]))

# FINDING: The joints ordering in matrix Y is given by data['skel']. Take into
# account that some joints span more than a degree of freedom.

# The following mapping is based on the amc file names. However, when I
# visualize the movement some of them seem to be swapped, this is, left is right
# and the other way around.

lfemur_x_id = 6
lfemur_y_id = 7
lfemur_z_id = 8
ltibia_id = 9
rtibia_id = 16
rradius_id = 55
lradius_id = 43

joints_to_analyse = [ltibia_id, rtibia_id, lradius_id, rradius_id]

animation_flag = False
if animation_flag:
    idx = 2
    Y = data[idx]['Y'][offsets[idx]:limits[idx]]
    Y[:, 0:3] = 0.   # Make figure walk in place
    visualize = GPy.plotting.matplot_dep.visualize.skeleton_show(
            Y[0, :], data[0]['skel'])
    GPy.plotting.matplot_dep.visualize.data_play(Y, visualize, 60)
    plt.close()

locations_per_segment = 20

# Scaling the original signal to be between -1 and 1.

max_value = 0
for i in range(len(motions_for_training)):
    max_value = max(max_value, np.abs(data[i]['Y']).max())

for i in range(len(motions_for_training)):
    data[i]['Y'] *= (1.0/max_value)

# end scaling

plt.figure(1)
for i in range(len(motions_for_training)):
    plt.subplot(len(motions_for_training), 1, i + 1)
    Y = data[i]['Y'][offsets[i]:limits[i], joints_to_analyse]
    nsamples, _ = Y.shape
    plt.xlim(0, 450)
    plt.plot(np.arange(nsamples), Y)
    switching_line = 0
    while(switching_line < 450):
        plt.axvline(x=switching_line, color='red', linestyle='--')
        # the -1 may be surprising but is necessary to compensate that we are
        # using a switching point twice.

        switching_line += locations_per_segment - 1
plt.show()

# Saving data

training_observations = 2
training = np.zeros(training_observations, dtype='object')
test_observations = len(motions_for_training) - training_observations
test = np.zeros(test_observations, dtype='object')

for s in range(len(motions_for_training)):
    Y = data[s]['Y'][offsets[s]:limits[s], joints_to_analyse]
    nsamples, noutputs = Y.shape
    number_segments = (nsamples - 1) / (locations_per_segment - 1)
    c_obs = np.zeros((number_segments, locations_per_segment * noutputs))
    for output_id in range(noutputs):
        signal = Y[:, output_id]
        idx = 0
        for i in range(number_segments):
            c_obs[i, output_id::noutputs] = \
                signal[idx:idx + locations_per_segment]
            idx = idx + locations_per_segment - 1
    if s < training_observations:
        training[s] = c_obs
    else:
        test[s - training_observations] = c_obs

saving = False
if saving:
    output_file = open('mocap_navigate_subject_41_normalized.npz', 'w')
    np.savez(output_file, training=training, test=test, outputs=noutputs,
             lps=locations_per_segment)
    output_file.close()
