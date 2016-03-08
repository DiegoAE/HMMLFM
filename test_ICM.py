__author__ = 'diego'

from hmm.continuous.ICMHMM import ICMHMM
from matplotlib import pyplot as plt
import numpy as np

seed = np.random.random_integers(10000)
np.random.seed(seed)
print "USED SEED", seed

pi = np.array([0.3, 0.3, 0.4])
print "initial state distribution", pi
A = np.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
print "hidden state transition matrix\n", A

number_hidden_states = 3
outputs = 2
start_t = 0.1
end_t = 5.1
locations_per_segment = 20
rbf_variances = np.ones(number_hidden_states) * 2
rbf_lengthscales = np.ones(number_hidden_states)
B_Ws = np.ones((number_hidden_states, outputs))
kappas = 0.5 * np.ones((number_hidden_states, outputs))
noise_var = 0.0005 * np.ones(outputs)

icm_hmm = ICMHMM(outputs, number_hidden_states, locations_per_segment, start_t,
                           end_t, verbose=True)
icm_hmm.set_params(A, pi, rbf_variances, rbf_lengthscales, B_Ws, kappas,
                   noise_var)

# Testing packing and unpacking
print "Testing packing and unpacking: ",
test_unit_1 = icm_hmm.unpack_params(icm_hmm.pack_params(icm_hmm.ICMparams))
test_unit_2 = icm_hmm.ICMparams
for k in test_unit_2.keys():
    assert np.allclose(test_unit_2[k], test_unit_1[k])
print "Accepted!"

for i in xrange(number_hidden_states):
    print icm_hmm.icms[i].icm_kernel.B.kappa