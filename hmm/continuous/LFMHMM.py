__author__ = 'diego'

from hmm._BaseHMM import _BaseHMM
from hmm.kernels.lfm2 import lfm2
from scipy import optimize
from scipy import stats
import multiprocessing as mp
import numpy as np


class LFMHMMError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def _parallel_hyperparameter_objective(tup):
    '''
    Auxiliar function to make it easier to evaluate the objective function for
    emission parameters optimization in a parallel way.
    '''
    idx, d = tup
    gamma = d['gammas'][idx]
    mvgs = d['mvgs']
    observation = d['obs'][idx]
    n = len(mvgs)
    n_observations = len(gamma)
    weighted_sum = 0.0
    for i in xrange(n):
        mvg = mvgs[i]
        for t in xrange(n_observations):
            weighted_sum += gamma[t][i] * mvg.logpdf(observation[t])
    return weighted_sum

class LFMHMM(_BaseHMM):

    def __init__(self, number_outputs, n, locations_per_segment, start_t, end_t,
                 number_latent_forces=1, precision=np.double, verbose=False):
        assert n > 0
        assert locations_per_segment > 0
        assert number_outputs > 0
        assert type(number_outputs) is type(1)
        self.n = n  # number of hidden states
        self.number_outputs = number_outputs
        self.start_t = start_t
        self.end_t = end_t
        self.sample_locations = np.linspace(start_t, end_t,
                                            locations_per_segment)
        self.locations_per_segment = locations_per_segment
        # Pool of workers to perform parallel computations when need it.
        self.pool = mp.Pool()
        # covariance memoization
        self.memo_covs = {}
        self.number_latent_f = number_latent_forces
        # initially not LFM params.
        self.LFMparams = {}
        # Latent Force Model objects
        self.lfms = np.zeros(n, dtype='object')
        for i in xrange(n):
            self.lfms[i] = lfm2(self.number_latent_f, number_outputs)
            self.lfms[i].set_inputs_with_same_ind(self.sample_locations)
        _BaseHMM.__init__(self, n, None, precision, verbose)
        self.reset()

    def pack_params(self, params_dict):
        # Note: reestimate parameters has to work with self.LFMparams
        # and in the optimization process you should work with the flattened
        # array.
        spring = params_dict['spring']
        damper = params_dict['damper']
        sensi = params_dict['sensi']
        lengthscales = params_dict['lengthscales']
        noise_var = params_dict['noise_var']
        packed_params = []
        for i in xrange(self.n):
            p = np.concatenate((spring[i], damper[i], np.hstack(sensi[i]),
                                lengthscales[i]), axis=0)
            packed_params.append(p)
        packed_params = np.concatenate(packed_params)
        packed_params = np.concatenate((packed_params, noise_var))
        return packed_params

    def unpack_params(self, params_array):
        ret_dict = {}
        spring = np.zeros((self.n, self.number_outputs))
        damper = np.zeros((self.n, self.number_outputs))
        sensi = np.zeros((self.n, self.number_outputs, self.number_latent_f))
        lengthscales = np.zeros((self.n, self.number_latent_f))
        noise_var = np.zeros(self.number_outputs)
        idx = 0
        for i in xrange(self.n):
            for j in xrange(self.number_outputs):
                spring[i][j] = params_array[idx]
                idx += 1
            for j in xrange(self.number_outputs):
                damper[i][j] = params_array[idx]
                idx += 1
            for j in xrange(self.number_outputs):
                for k in xrange(self.number_latent_f):
                    sensi[i][j][k] = params_array[idx]
                    idx += 1
            for j in xrange(self.number_latent_f):
                lengthscales[i][j] = params_array[idx]
                idx += 1
        for j in xrange(self.number_outputs):
            noise_var[j] = params_array[idx]
            idx += 1
        assert idx == np.size(params_array)
        ret_dict['spring'] = spring
        ret_dict['damper'] = damper
        ret_dict['sensi'] = sensi
        ret_dict['noise_var'] = noise_var
        ret_dict['lengthscales'] = lengthscales
        return ret_dict


    def reset(self,init_type='uniform', emissions_reset=True):
        """If required, initialize the model parameters according the selected
        policy."""
        if init_type == 'uniform':
            pi = np.ones(self.n, dtype=self.precision)*(1.0/self.n)
            A = np.ones((self.n,self.n), dtype=self.precision)*(1.0/self.n)
            new_params = {'A': A, 'pi': pi}
            if emissions_reset:
                LFMparams = {}
                LFMparams['spring'] = np.random.rand(
                        self.n, self.number_outputs) * 2.
                LFMparams['damper'] = np.random.rand(
                        self.n, self.number_outputs) * 2.
                LFMparams['lengthscales'] = np.random.rand(
                        self.n, self.number_latent_f) * 2.
                LFMparams['sensi'] = np.ones(
                        (self.n, self.number_outputs, self.number_latent_f))
                # LFMparams['sensi'] = np.random.randn(
                #         self.n, self.number_outputs, self.number_latent_f)
                # Assuming different noises for each output.
                LFMparams['noise_var'] = np.ones(self.number_outputs) * 100.
                new_params['LFMparams'] = LFMparams
            else:
                new_params['LFMparams'] = self.LFMparams
            self._updatemodel(new_params)
            if self.observations is not None:
                self._mapB()
        else:
            raise LFMHMMError("reset init_type not supported.")

    def set_params(self, A, pi, damper, spring, lengthscales, noise_var):
        n = self.n
        assert A.shape == (n, n)
        assert (pi.shape == (n, 1)) or (pi.shape == (n, ))
        assert len(damper) == len(spring) == n
        assert lengthscales.shape == (self.n, self.number_latent_f)
        assert all([len(x) == self.number_outputs for x in damper])
        assert all([len(x) == self.number_outputs for x in spring])
        assert noise_var.shape == (self.number_outputs,)
        # TODO: Assumption of sensitivities being equal to one. Make a parameter
        sensi = np.ones((n, self.number_outputs, self.number_latent_f))
        # sensi = np.random.randn(n, self.number_outputs, self.number_latent_f)
        pdict = {}
        pdict['spring'] = spring
        pdict['damper'] = damper
        pdict['sensi'] = sensi
        pdict['noise_var'] = noise_var
        pdict['lengthscales'] = lengthscales
        self.LFMparams = pdict
        # Setting the transition matrix, the initial state PMF and the emission
        # params.
        params_to_set = {'A': A, 'pi': pi, 'LFMparams': self.LFMparams}
        self._updatemodel(params_to_set)
        if self.observations is not None:
            self._mapB()

    def generate_observations(self, segments, hidden_s=None):
        output = np.zeros((segments,
                           self.locations_per_segment * self.number_outputs),
                           dtype=self.precision)
        initial_state = np.nonzero(np.random.multinomial(1, self.pi))[0][0]
        hidden_states = [initial_state]
        for x in xrange(1, segments):
            hidden_states.append(np.nonzero(np.random.multinomial(
                    1, self.A[hidden_states[-1]]))[0][0])
        if hidden_s:
            hidden_states = hidden_s
        for i in xrange(len(hidden_states)):
            state = hidden_states[i]
            cov = self.get_cov_function(state)
            realization = np.random.multivariate_normal(
                    mean=np.zeros(cov.shape[0]), cov=cov)
            output[i, :] = realization
        print "Hidden States", hidden_states
        return output, hidden_states

    def get_cov_function(self, hidden_state, cache=True):
        if cache and (hidden_state in self.memo_covs):
            return self.memo_covs[hidden_state]
        assert hidden_state < self.n
        cov = self.lfms[hidden_state].Kyy()
        self.memo_covs[hidden_state] = cov
        return cov

    def get_cov_function_explicit(self, hidden_state, t, tp):
        # Notice that the noise variance doesn't appear here.
        # The noise variance only affects Ktt.
        assert hidden_state < self.n
        nt, ind = self.lfms[hidden_state].get_stacked_time_and_indexes(t)
        ntp, indp = self.lfms[hidden_state].get_stacked_time_and_indexes(tp)
        cov = self.lfms[hidden_state].Kff(nt, ind, ntp, indp)
        return cov

    def predict(self, t_step, hidden_state, obs):
        # TODO: In lfm2.py there is a predict function that you can try to plug in
        if self.verbose and \
                (np.any(t_step < self.start_t) or np.any(t_step > self.end_t)):
            print "WARNING:prediction step.Time step out of the sampling region"
        if hidden_state < 0 or hidden_state >= self.n:
            raise LFMHMMError("ERROR: Invalid lfm state -> %d." % hidden_state)
        obs = obs.reshape((-1, 1))
        Ktt = self.get_cov_function(hidden_state)
        ktstar = self.get_cov_function_explicit(
                hidden_state, self.sample_locations, np.asarray(t_step))
        Kstarstar = self.get_cov_function_explicit(
                hidden_state, np.asarray(t_step),  np.asarray(t_step))
        mean_pred = np.dot(ktstar.T, np.linalg.solve(Ktt, obs))
        cov_pred = Kstarstar - np.dot(ktstar.T, np.linalg.solve(Ktt, ktstar))
        return mean_pred, cov_pred

    def _reestimateLFMparams(self, gammas):
        new_LFMparams = self.optimize_hyperparams(gammas)
        print "CURRENT VALUE OF EMISSION PARAMS: "
        print self.unpack_params(new_LFMparams)
        return self.unpack_params(new_LFMparams)

    def objective_for_hyperparameters(self, gammas, parallelized=True):
        if parallelized:
            return self.parallel_hyperparameters(gammas)
        # non-parallel code
        weighted_sum = 0.0
        n_sequences = len(gammas)
        for i in xrange(self.n):
            # print "HIDDEN STATE:", i
            current_lfm = self.lfms[i]
            cov = self.get_cov_function(i, False)
            mvg = stats.multivariate_normal(np.zeros(cov.shape[0]), cov, True)
            for s in xrange(n_sequences):
                gamma = gammas[s]
                n_observations = len(gamma)
                for t in xrange(n_observations):
                    # current_lfm.set_outputs(self.observations[s][t])
                    # weighted_sum += gamma[t][i] * current_lfm.LLeval()
                    # other way
                    weighted_sum += gamma[t][i] * mvg.logpdf(self.observations[s][t])
        return weighted_sum

    def parallel_hyperparameters(self, gammas):
        n_sequences = len(gammas)
        mvgs = []
        for i in xrange(self.n):
            cov = self.get_cov_function(i, False)
            mvg = stats.multivariate_normal(np.zeros(cov.shape[0]), cov, True)
            mvgs.append(mvg)
        d = {
            'mvgs': mvgs,
            'obs': self.observations,
            'gammas': gammas,
        }
        l = zip(range(n_sequences), [d] * n_sequences)
        ret = self.pool.map(_parallel_hyperparameter_objective, l)
        return np.sum(ret)


    def _wrapped_objective(self, params, gammas):
        # print "parameters :", self.unpack_params(params)
        self._update_emission_params(params)
        return -self.objective_for_hyperparameters(gammas)

    def _reestimate(self, stats):
        new_model = _BaseHMM._reestimate(self, stats)
        new_model['LFMparams'] = self._reestimateLFMparams(stats['gammas'])
        return new_model

    def optimize_hyperparams(self, gammas):
        # initilization with the current LFMparams.
        packed = self.pack_params(self.LFMparams)
        result = optimize.minimize(self._wrapped_objective, packed, gammas,
                                   bounds=self._get_bounds())
        print "==============="
        print result.message
        print "iterations:", result.nit
        print "==============="
        return result.x

    def _update_emission_params(self, lfms_params):
        # Notice that this function works with the packed params.
        # Be careful because this function doesn't update self.LFMparams
        # So it is expected to update it  after/before using this.
        per_lfm = 2*self.number_outputs + \
                  self.number_latent_f * (1 + self.number_outputs)
        noise_params = lfms_params[per_lfm * self.n:]
        # noise_params = np.ones(self.number_outputs) * 0.005
        assert np.size(noise_params) == self.number_outputs
        # updating each of the lfm's (i.e. hidden states) with the new params.
        for i in xrange(self.n):
            no_noise_params = lfms_params[i * per_lfm: (i + 1) * per_lfm]
            self.lfms[i].set_params(
                    np.concatenate((no_noise_params, noise_params)), False)

    def _get_bounds(self):
        tam_sensi = self.number_outputs * self.number_latent_f
        upper = 10000.0
        lower = 0.0005
        bounds = []
        for i in xrange(self.n):
            f = [(lower, upper)] * (2 * self.number_outputs)  # spring & damper.
            s = [(None, None)] * tam_sensi  # sensitivities bound.
            t = [(lower, None)] * self.number_latent_f  # length-scales bound.
            bounds.extend(f)
            bounds.extend(s)
            bounds.extend(t)
        for i in xrange(self.number_outputs):
            bounds.append((lower, None))  # noise variance bounds.
        total_length = self.n * (2*self.number_outputs + self.number_latent_f *
                                 (1 + self.number_outputs))+self.number_outputs
        assert len(bounds) == total_length
        return bounds

    def _updatemodel(self, new_model):
        '''
        This function updates the values of model attributes. Namely
        self.LFMparams, self.A, self.pi and self.lfms.
        Note that this doesn't update the probabilities B_maps
        '''
        self.LFMparams = new_model['LFMparams']
        packed_params = self.pack_params(self.LFMparams)
        self._update_emission_params(packed_params)
        _BaseHMM._updatemodel(self, new_model)
        # erasing the covariances cache since the parameters were updated.
        self.memo_covs = {}

    def _mapB(self):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        '''
        # Erasing the covariance cache here. Should I do this here?
        self.memo_covs = {}
        if self.observations is None:
            raise LFMHMMError("The training sequences haven't been set.")

        numbers_of_sequences = len(self.observations)

        self.B_maps = np.zeros((numbers_of_sequences, self.n), dtype=object)

        for j in xrange(numbers_of_sequences):
            for i in xrange(self.n):
                self.B_maps[j][i] = np.zeros(len(self.observations[j]),
                                             dtype=self.precision)

        # strange behavior found between numpy and stats. See below.

        for j in xrange(numbers_of_sequences):
            number_of_segments = len(self.observations[j])
            for i in xrange(self.n):
                cov = self.get_cov_function(i)
                for t in xrange(number_of_segments):
                    self.B_maps[j][i][t] = stats.multivariate_normal.pdf(
                            self.observations[j][t], np.zeros(cov.shape[0]),
                            cov, True)  # Allowing singularity in cov sometimes.
                    #  This is weird.
        # print self.B_maps

    def save_params(self, dir, name):
        f = file('%s/%s.param' % (dir, name), 'w')
        f.write(repr(self.LFMparams) + "\n")
        f.write("#\n")
        f.write(repr(self.A) + "\n")
        f.write("#\n")
        f.write(repr(self.pi) + "\n")
        f.close()

    def read_params(self, dir, name):
        f = file('%s/%s.param' % (dir, name), 'r')
        LFMparams_string = ""
        A_string = ""
        pi_string = ""
        read_line = f.readline()
        while not read_line.startswith("#"):
            LFMparams_string += read_line
            read_line = f.readline()
        read_line = f.readline()
        while not read_line.startswith("#"):
            A_string += read_line
            read_line = f.readline()
        read_line = f.readline()
        while read_line:
            pi_string += read_line
            read_line = f.readline()
        f.close()
        from numpy import array, nan  # required for eval to work.
        model_to_set = {
            'LFMparams': eval(LFMparams_string),
            'A': eval(A_string),
            'pi': eval(pi_string),
        }
        assert model_to_set['A'].shape == (self.n, self.n)
        assert np.size(model_to_set['pi']) == self.n
        assert model_to_set['LFMparams']['spring'].shape == \
               model_to_set['LFMparams']['damper'].shape == \
               (self.n, self.number_outputs)
        assert model_to_set['LFMparams']['lengthscales'].shape == \
               (self.n, self.number_latent_f)
        assert model_to_set['LFMparams']['noise_var'].shape == \
               (self.number_outputs,)
        self._updatemodel(model_to_set)
        if self.observations is not None:
            self._mapB()
