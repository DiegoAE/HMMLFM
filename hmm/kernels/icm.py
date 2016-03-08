import GPy
import numpy as np

class icm():
    def __init__(self, noutputs, sample_locations):
        self.noutputs = noutputs
        # Assuming the same sample locations for all outputs.
        self.sample_locations = sample_locations
        self.X = None
        self.set_inputs_with_same_ind(sample_locations)
        # TODO: not sure if I need to initiliaze params here.
        # I don't think I should store the params.
        self.reset()

    def reset(self):
        # Assuming the default kernel as RBF.
        self.kernel = GPy.kern.RBF(1)  # one input dimension
        self.icm_kernel = GPy.util.multioutput.ICM(
                1, self.noutputs, self.kernel)
        # Note: just the parameter B.W gets randomly initialized.

    def set_inputs(self, X):
        self.X = X

    def set_inputs_with_same_ind(self, t):
        X = self.get_stacked_time_and_indexes(t, self.noutputs)
        self.set_inputs(X)

    def get_stacked_time_and_indexes(self, t, noutputs):
        """
        This function assumes the same input locations for ALL the outputs. So
        it returns the sample locations vector (t) stacked noutputs times and
        the corresponding array of indexes for each output.
        """
        stacked_time = np.repeat(t, noutputs).reshape((-1, 1))
        idx = np.tile(np.arange(noutputs, dtype='int'),
                      np.size(t)).reshape((-1, 1))
        return np.hstack((stacked_time, idx))

    def set_params(self, params):
        total_params = 2 + 3 * self.noutputs
        assert np.size(params) == total_params
        rbf_variance = params[0]
        rbf_lengthscale = params[1]
        W = params[2:2 + self.noutputs]
        kappa = params[2 + self.noutputs: 2 + 2*self.noutputs]
        # TODO: see if it is possible to add a noise kernel. That way is not
        # necessary to store the noise variance neither.
        self.noise_var = params[-self.noutputs:]
        self.icm_kernel.rbf.variance = rbf_variance
        self.icm_kernel.rbf.lengthscale = rbf_lengthscale
        self.icm_kernel.B.W[:, 0] = W
        self.icm_kernel.B.kappa = kappa

