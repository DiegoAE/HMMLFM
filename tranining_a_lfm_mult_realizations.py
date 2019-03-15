import numpy as np
from hmm.kernels.lfm2 import lfm2
from scipy import stats
from scipy import optimize

class MultipleRealizationsLFM():
    def __init__(self, Q, noutputs, lfms):
        #Initialize parameters
        self.D = noutputs
        self.Q = Q
        self.nvar = 3*self.D+self.Q*(1+self.D)
        #TODO: the following can be done in update parameters part
        self.B = np.random.rand(self.D)*2.
        self.C = np.random.rand(self.D)*2.
        self.l = np.random.rand(self.Q)*2.
        self.S = np.random.randn(self.D,self.Q)
        self.sn = np.ones(self.D)*1e2
        self.params = np.concatenate((self.B, self.C, np.hstack(self.S), self.l,
                                      self.sn), axis=0)
        self._set_params_to_lfms()
        self.lfms = lfms

    def set_params(self, params):
        # params=params.flatten()
        assert np.size(params) == self.nvar
        self.params = params
        for q in range(self.Q):
            self.l[q] = params[(2+self.Q)*self.D+q]
        for d in range(self.D):
            self.B[d] = params[d]  # Spring coefficients
            self.C[d] = params[d+self.D]  # Damper coefficients
            self.sn[d] = params[(2+self.Q)*self.D+self.Q+d]
            for q in range(self.Q):
                self.S[d][q] = params[2*self.D+q+d*self.Q]

    def _set_params_to_lfms(self, x):
        for lfm in self.lfms:
            lfm.set_params(x)

    def wrapped_objective(self, x):
        self._set_params_to_lfms(x)
        sum = 0
        for lfm in self.lfms:
            sum += lfm.LLeval()
        return -sum

    def Train(self):
        results = optimize.minimize(self.wrapped_objective, self.params)
        print(results.fun)
        self.set_params(results.x)


seed = np.random.random_integers(100000)
seed = 79861  # LFM
np.random.seed(seed)
print("USED SEED", seed)

input_file = file('WalkingExperiment/mocap_walking_subject_07.npz', 'rb')
# input_file = file('toy_lfm.npz', 'rb')
data = np.load(input_file)
outputs = data['outputs'].item()
training_observations = data['training']
testing_observations = data['test']
locations_per_segment = data['lps']

number_latent_f = 3
start_t = 0.1
end_t = 5.1

# Translating everything from segments to continuous things:
lfms = []
for realization in training_observations:
    nsegments = realization.shape[0]
    lfm = lfm2(number_latent_f, outputs)
    t = np.zeros((0,1), dtype='float64')
    last_value = 0
    for s in range(nsegments):
        tmp = np.linspace(start_t, end_t, locations_per_segment).reshape((-1,1))
        tmp += last_value
        t = np.vstack((t, tmp))
        last_value += end_t - start_t
    flattened_realization = realization.reshape((-1,1))
    lfm.set_inputs_with_same_ind(t)
    lfm.set_outputs(flattened_realization)
    lfms.append(lfm)
    print(flattened_realization.shape)




