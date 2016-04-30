from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
import numpy as np


class LFMHMMTyingMO(LFMHMMcontinuousMO):

    def __init__(self, *args, **kwargs):
        self.tying_flag = True
        super(LFMHMMTyingMO, self).__init__(*args, **kwargs)

    def set_params(self, A, pi, damper, spring, lengthscales, noise_var):
        super(LFMHMMTyingMO, self).set_params(A, pi, damper, spring,
                                              lengthscales, noise_var)
        if self.tying_flag:
            assert (damper == damper[0]).all(), "The dampers must be equal."
        if self.tying_flag:
            assert (spring == spring[0]).all(), "The springs must be equal."

    def _update_emission_params(self, input_params):
        # it is important to not modifying input_params in place.
        lfms_params = input_params.copy()
        per_lfm = 2*self.number_outputs + \
                  self.number_latent_f * (1 + self.number_outputs)
        noise_params = lfms_params[per_lfm * self.n:]
        assert np.size(noise_params) == self.number_outputs
        reference = lfms_params[:per_lfm]  # first state emission params
        for i in xrange(self.n):
            no_noise_params = lfms_params[i * per_lfm: (i + 1) * per_lfm]
            # spring & damper constants equal across hidden states.
            no_noise_params[:2*self.number_outputs] = \
                reference[:2*self.number_outputs]
        super(LFMHMMTyingMO, self)._update_emission_params(lfms_params)

    def _reestimateLFMparams(self, gammas):
        LFMparams = super(LFMHMMTyingMO, self)._reestimateLFMparams(gammas)
        # ensuring consistency in the parameters
        if self.tying_flag:
            actual_spring_row = LFMparams['spring'][0]
            LFMparams['spring'] = np.tile(actual_spring_row, (self.n, 1))
            actual_damper_row = LFMparams['damper'][0]
            LFMparams['damper'] = np.tile(actual_damper_row, (self.n, 1))
        return LFMparams