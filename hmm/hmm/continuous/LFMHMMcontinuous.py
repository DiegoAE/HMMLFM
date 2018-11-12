import copy
from hmm.continuous.LFMHMM import LFMHMM
import numpy as np

class LFMHMMcontinuous(LFMHMM):
    def __init__(self, *args, **kwargs):
        super(LFMHMMcontinuous, self).__init__(*args, **kwargs)

    def generate_observations(self, segments):
        """ This method returns a continuous realization of the model."""
        output = np.zeros((segments, self.locations_per_segment),
                          dtype=self.precision)
        initial_state = np.nonzero(np.random.multinomial(1, self.pi))[0][0]
        hidden_states = [initial_state]
        for x in xrange(1, segments):
            hidden_states.append(np.nonzero(np.random.multinomial(
                1, self.A[hidden_states[-1]]))[0][0])
        last_observation_value = 0.0
        for i in xrange(len(hidden_states)):
            state = hidden_states[i]
            cov = self.get_cov_function(state)
            # Conditioning the first value to be equal to the last observation
            # value.
            A = cov[0, 0].item()
            C = cov[0, 1:].reshape((1, -1))
            B = cov[1:, 1:]
            # mean_cond = C * (last_observation_value/A)
            mean_cond = np.zeros(C.size)  # The last observed value is 0.
            cov_cond = B - (1./A) * np.dot(C.T, C)
            realization = np.random.multivariate_normal(
                mean=mean_cond.flatten(), cov=cov_cond)
            output[i, 0] = 0
            output[i, 1:] = realization
            output[i] += last_observation_value
            last_observation_value = output[i][-1]
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
            self.segment_shifts[s] = np.zeros(length_ob, dtype=self.precision)
            for i in xrange(length_ob):
                self.segment_shifts[s][i] = obs[s][i][0]
                obs[s][i] -= self.segment_shifts[s][i]
        super(LFMHMMcontinuous, self).set_observations(obs)

    def predict(self, t_step, hidden_state, obs):
        current_shift = obs[0]
        mean_pred, cov_pred = super(LFMHMMcontinuous, self).predict(
                t_step, hidden_state, obs - current_shift)
        return mean_pred + current_shift, cov_pred



