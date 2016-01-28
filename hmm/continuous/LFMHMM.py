__author__ = 'diego'

from hmm._BaseHMM import _BaseHMM
from hmm.lfm2kernel.lfm2 import lfm2
from hmm.lfm2kernel import SecondOrderLFMKernel
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

    def __init__(self, n, A, pi, number_outputs, start_t, end_t,
                 locations_per_segment, damper, spring, lengthscales,
                 noise_var, precision=np.double, verbose=False):
        assert n > 0
        assert locations_per_segment > 0
        assert A.shape == (n, n)
        assert (pi.shape == (n, 1)) or (pi.shape == (n, ))
        assert number_outputs > 0
        assert len(damper) == len(spring) == len(lengthscales) == n
        assert all([len(x) == number_outputs for x in damper])
        assert all([len(x) == number_outputs for x in spring])
        _BaseHMM.__init__(self, n, None, precision, verbose)
        self.n = n  # number of hidden states
        self.number_outputs = number_outputs
        self.start_t = start_t
        self.end_t = end_t
        self.sample_locations = np.linspace(start_t, end_t,
                                            locations_per_segment)
        self.locations_per_segment = locations_per_segment
        # TODO: make the number of latent forces a parameter.
        # and the same thing with the sensitivities. For now all them are 1.
        self.number_latent_f = 1
        # implicit assumption of sensitivities being equal to one.
        self.sensi = np.ones((n, number_outputs, self.number_latent_f))
        # ======
        # Once the the get_cov_* and predict functions are refactored I can
        # get rid of this.
        self.spring_cons = spring
        self.damper_cons = damper
        self.lengthscales = lengthscales
        self.noise_var = noise_var
        # ======
        # Pool of workers to perform parallel computations when need it.
        self.pool = mp.Pool()
        #
        pdict = {}
        pdict['spring'] = spring
        pdict['damper'] = damper
        pdict['sensi'] = self.sensi
        pdict['noise_var'] = noise_var
        pdict['lengthscales'] = lengthscales
        self.LFMparams = pdict
        # covariance memoization
        self.memo_covs = {}
        self.lfms = np.zeros(n, dtype='object')
        # ======
        # TODO: I think this part of the code should be in lfm2.py
        idx = np.zeros(shape=(0, 1), dtype=np.int8)
        time_length = len(self.sample_locations)
        stacked_time = np.zeros(shape=(0, 1))
        for d in xrange(number_outputs):
            idx = np.vstack((idx, d * np.ones((time_length, 1), dtype=np.int8)))
            stacked_time = np.vstack((stacked_time,
                                      self.sample_locations.reshape(-1,1)))
        # ======
        for i in xrange(n):
            self.lfms[i] = lfm2(1, number_outputs)
            self.lfms[i].set_inputs(stacked_time, idx)
        # Setting the transition matrix, the initial stater PMF and the emission
        # params.
        params_to_set = {'A': A, 'pi': pi, 'LFMparams': self.LFMparams}
        self._updatemodel(params_to_set)

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
            p = np.concatenate((np.log(spring[i]), np.log(damper[i]),
                                np.hstack(sensi[i]),
                                np.log(lengthscales[i])), axis=0)
            packed_params.append(p)
        packed_params = np.concatenate(packed_params)
        packed_params = np.concatenate((packed_params,
                                        np.log(np.array([noise_var])) ))
        return packed_params

    def unpack_params(self, params_array):
        ret_dict = {}
        spring = np.zeros((self.n, self.number_outputs))
        damper = np.zeros((self.n, self.number_outputs))
        sensi = np.zeros((self.n, self.number_outputs, self.number_latent_f))
        lengthscales = np.zeros((self.n, self.number_latent_f))
        noise_var = 0
        idx = 0
        for i in xrange(self.n):
            for j in xrange(self.number_outputs):
                spring[i][j] = np.exp(params_array[idx])
                idx += 1
            for j in xrange(self.number_outputs):
                damper[i][j] = np.exp(params_array[idx])
                idx += 1
            for j in xrange(self.number_outputs):
                for k in xrange(self.number_latent_f):
                    sensi[i][j][k] = params_array[idx]
                    idx += 1
            for j in xrange(self.number_latent_f):
                lengthscales[i][j] = np.exp(params_array[idx])
                idx += 1
        noise_var = np.exp(params_array[idx])
        idx += 1
        assert idx == np.size(params_array)
        ret_dict['spring'] = spring
        ret_dict['damper'] = damper
        ret_dict['sensi'] = sensi
        ret_dict['noise_var'] = noise_var
        ret_dict['lengthscales'] = lengthscales
        return ret_dict


    def reset(self,init_type='uniform', emissions_reset=True):
        '''
        If required, initalize the model parameters according the selected policy
        '''
        if init_type == 'uniform':
            pi = np.ones(self.n, dtype=self.precision)*(1.0/self.n)
            A = np.ones((self.n,self.n), dtype=self.precision)*(1.0/self.n)
            new_params = {'A': A, 'pi': pi, 'LFMparams': self.LFMparams}
            if emissions_reset:
                LFMparams = {}
                LFMparams['spring'] = np.random.rand(
                        self.n, self.number_outputs) * 2.
                LFMparams['damper'] = np.random.rand(
                        self.n, self.number_outputs) * 2.
                LFMparams['lengthscales'] = np.random.rand(
                        self.n, self.number_outputs) * 2.
                LFMparams['sensi'] = np.random.randn(
                        self.n, self.number_outputs, self.number_latent_f)
                # Assuming the same noise for all the outputs and LFM's.
                LFMparams['noise_var'] = 1e2
                new_params['LFMparams'] = LFMparams
            self._updatemodel(new_params)
            self._mapB()
        else:
            raise LFMHMMError("reset init_type not supported.")

    def generate_observations(self, segments, hidden_s=None):
        output = np.zeros((segments, self.locations_per_segment),
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

    def generate_continuous_observations(self, segments):
        # Notice the difference in the number of locations.
        output = np.zeros((segments, self.locations_per_segment - 1),
                          dtype=self.precision)
        initial_state = np.nonzero(np.random.multinomial(1, self.pi))[0][0]
        hidden_states = [initial_state]
        for x in xrange(1, segments):
            hidden_states.append(np.nonzero(np.random.multinomial(
                1, self.A[hidden_states[-1]]))[0][0])
        last_observation_value = 1.0
        for i in xrange(len(hidden_states)):
            state = hidden_states[i]
            cov = self.get_cov_function(state)
            # Conditioning the first value to be equal to the last observation
            # value.
            A = cov[0, 0].item()
            C = cov[0, 1:].reshape((1, -1))
            B = cov[1:, 1:]
            mean_cond = C * (last_observation_value/A)
            cov_cond = B - (1./A) * np.dot(C.T, C)
            realization = np.random.multivariate_normal(
                mean=mean_cond.flatten(), cov=cov_cond)
            output[i, :] = realization
            last_observation_value = realization[-1]
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
        # TODO: This function doesn't take into account the existence of
        # lfm2.py. So it can be refactored.
        # TODO: SUPERIMPORTANT using the attributes self.spring, self.damper,
        # etc. is a bug since they are not updated. Fix throughout.
        B = np.asarray(self.spring_cons[hidden_state])
        C = np.asarray(self.damper_cons[hidden_state])
        l = self.lengthscales[hidden_state][0]
        # Notice that the noise variance doesn't appear here.
        # The noise variance only affects Ktt.
        cov = SecondOrderLFMKernel.K_pred(B, C, l, t.reshape((-1, 1)),
                                          tp.reshape((-1, 1)))
        return cov

    def predict(self, t_step, hidden_state, obs):
        # TODO: there is a bad smell here. obs vs set observations.
        # This function doesn't take into account the existence of lfm2.py. So
        # it can be refactored.
        if self.verbose and \
                (np.any(t_step < self.start_t) or np.any(t_step > self.end_t)):
            print "WARNING:prediction step.Time step out of the sampling region"
        if hidden_state < 0 or hidden_state >= self.n:
            raise LFMHMMError("ERROR: Invalid hidden state.")
        obs = obs.reshape((-1, 1))
        # TODO: figure out if it is OK to use caching here. I guess it is.
        Ktt = self.get_cov_function(hidden_state)
        ktstar = self.get_cov_function_explicit(
            hidden_state, self.sample_locations, np.asarray(t_step))
        Kstarstar = self.get_cov_function_explicit(
            hidden_state, np.asarray(t_step),  np.asarray(t_step))
        mean_pred = np.dot(ktstar.T, np.linalg.solve(Ktt, obs))
        cov_pred = Kstarstar - np.dot(ktstar.T, np.linalg.solve(Ktt, ktstar))
        return mean_pred, cov_pred

    # def set_observations(self, observations):
    #     # TODO: not sure if I have to set outputs here for lfms.
    #     _BaseHMM.set_observations(self, observations)

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
        # updating each of the lfm's (i.e. hidden states) with the new params.
        for i in xrange(self.n):
            no_noise_params = lfms_params[i * per_lfm: (i + 1) * per_lfm]
            noise_param = lfms_params[-1:]
            self.lfms[i].set_params(
                    np.concatenate((no_noise_params, noise_param)), False)

    def _get_bounds(self):
        tam_sensi = self.number_outputs * self.number_latent_f
        upper = 10.0
        bounds = []
        for i in xrange(self.n):
            f = [(None, upper)] * (2 * self.number_outputs)  # spring & damper.
            s = [(None, None)] * tam_sensi  # sensitivities bound.
            t = [(None, upper)] * self.number_latent_f  # length-scales bound.
            bounds.extend(f)
            bounds.extend(s)
            bounds.extend(t)
        bounds.append((None, upper))  # noise variance bound.
        total_length = self.n * (2*self.number_outputs +
                                 self.number_latent_f *
                                 (1 + self.number_outputs)) + 1
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

    def _mapB(self):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        '''
        # Erasing the covariance cache
        self.memo_covs = {}
        if not self.observations:
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
                        cov, False)  # Allowing singularity in cov sometimes.
                                     # This is weird.

