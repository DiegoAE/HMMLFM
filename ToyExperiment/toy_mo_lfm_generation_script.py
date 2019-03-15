from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
from matplotlib import pyplot as plt
import numpy as np

locations_per_segment = 20

seed = 75599
np.random.seed(seed)
print(("USED SEED", seed))

pi = np.array([0.1, 0.3, 0.6])
print(("initial state distribution", pi))
A = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.3, 0.2, 0.5]])
print(("hidden state transition matrix\n", A))

number_lfm = 3
outputs = 4
start_t = 0.1
end_t = 5.1
locations_per_segment = 20

damper_constants = np.asarray(
        [[1., 3., 7.5, 10.],
         [3., 10., 0.5, 0.1],
         [6., 5., 4., 9.]]
)

spring_constants = np.asarray(
        [[3., 1, 2.5, 10.],
         [1., 3.5, 9.0, 5.0],
         [5., 8., 4.5, 1.]]
)


# implicitly assuming there is only one latent force governing the system.
lengthscales = np.asarray([[10.], [2.], [5.]])

noise_var = np.array([0.0005, 0.005, 0.0025, 0.0008])

lfm_hmm = LFMHMMcontinuousMO(outputs, number_lfm, locations_per_segment,
                             start_t, end_t, verbose=True)
lfm_hmm.set_params(A, pi, damper_constants, spring_constants, lengthscales,
                   noise_var)

# Generating observations

number_realizations = 10
training_data = np.zeros(number_realizations, dtype='object')
testing_data = np.zeros(number_realizations, dtype='object')
training_viterbi = np.zeros(number_realizations, dtype='object')
testing_viterbi = np.zeros(number_realizations, dtype='object')

for f in range(number_realizations):
    # fixed length
    training_data[f], viterbi_tr = lfm_hmm.generate_observations(20)
    testing_data[f], viterbi_te = lfm_hmm.generate_observations(20)
    training_viterbi[f] = np.array(viterbi_tr, dtype='int')
    testing_viterbi[f] = np.array(viterbi_te, dtype='int')

# Scaling the original signal to be between -1 and 1.

max_value = 0
for i in range(number_realizations):
    max_value = max(max_value, np.abs(training_data[i]).max())
    max_value = max(max_value, np.abs(testing_data[i]).max())

for i in range(number_realizations):
    training_data[i] *= (1.0/max_value)
    testing_data[i] *= (1.0/max_value)

# end scaling

saving = False
if saving:
    output_file = open('toy_lfm.npz', 'w')
    np.savez(output_file, training=training_data, test=testing_data,
             training_viterbi=training_viterbi, testing_viterbi=testing_viterbi,
             outputs=outputs, lps=locations_per_segment)
    output_file.close()
