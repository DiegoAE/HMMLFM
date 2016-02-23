import copy
from hmm.continuous.LFMHMM import LFMHMM
import numpy as np

class LFMHMMcontinuousMO(LFMHMM):
    def __init__(self, *args, **kwargs):
        self.segment_shifts = None
        super(LFMHMMcontinuousMO, self).__init__(*args, **kwargs)

    def generate_observations(self, segments):
        """ This method returns a continuous realization of the model."""
        output = np.zeros((segments,
                           self.locations_per_segment * self.number_outputs),
                          dtype=self.precision)
        initial_state = np.nonzero(np.random.multinomial(1, self.pi))[0][0]
        hidden_states = [initial_state]
        for x in xrange(1, segments):
            hidden_states.append(np.nonzero(np.random.multinomial(
                1, self.A[hidden_states[-1]]))[0][0])
        last_observation_value = np.zeros(self.number_outputs)
        for i in xrange(len(hidden_states)):
            state = hidden_states[i]
            cov = self.get_cov_function(state)
            # Conditioning the first value to be equal to the last observation
            # value in that output.
            A = cov[:self.number_outputs, :self.number_outputs]
            C = cov[:self.number_outputs, self.number_outputs:]
            B = cov[self.number_outputs:, self.number_outputs:]
            # The last observed values are 0.
            mean_cond = np.zeros(output.shape[1] - self.number_outputs)
            cov_cond = B - np.dot(C.T, np.linalg.solve(A, C))
            realization = np.random.multivariate_normal(
                mean=mean_cond.flatten(), cov=cov_cond)
            output[i, :self.number_outputs] = 0
            output[i, self.number_outputs:] = realization
            for j in xrange(self.number_outputs):
                output[i, j::self.number_outputs] += last_observation_value[j]
            last_observation_value = output[i, -self.number_outputs:]
        print "Hidden States", hidden_states
        return output, hidden_states

    def set_observations(self, observations):
        # The input obs. must not to be changed.
        obs = copy.deepcopy(observations)
        # is it necessary to store the original observations?
        number_of_sequences = len(obs)
        self.segment_shifts = np.zeros(number_of_sequences, dtype='object')
        for s in xrange(number_of_sequences):
            length_ob = len(obs[s])
            self.segment_shifts[s] = np.zeros((length_ob, self.number_outputs),
                                              dtype=self.precision)
            for i in xrange(length_ob):
                self.segment_shifts[s][i] = obs[s][i][:self.number_outputs]
                for j in xrange(self.number_outputs):
                    obs[s][i][j::self.number_outputs] -=\
                        self.segment_shifts[s][i][j]
        super(LFMHMMcontinuousMO, self).set_observations(obs)

    def predict(self, t_step, hidden_state, observations):
        obs = copy.deepcopy(observations)
        current_shift = obs[:self.number_outputs].copy()
        for j in xrange(self.number_outputs):
            obs[j::self.number_outputs] -= current_shift[j]
        mean_pred, cov_pred = super(LFMHMMcontinuousMO, self).predict(
                t_step, hidden_state, obs)
        for j in xrange(self.number_outputs):
            mean_pred[j::self.number_outputs] += current_shift[j]
        return mean_pred, cov_pred