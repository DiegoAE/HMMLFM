'''
Created on Oct 31, 2012

@author: GuyZ

This code is based on:
 - QSTK's HMM implementation - http://wiki.quantsoftware.org/
 - A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition, LR RABINER 1989 
'''

import numpy

class _BaseHMM(object):
    '''
    Implements the basis for all deriving classes, but should not be used directly.
    '''
    
    def __init__(self,n,m,precision=numpy.double,verbose=False):
        self.n = n
        self.m = m
        self.observations = None
        
        self.precision = precision
        self.verbose = verbose
        self._eta = self._eta1
        
    def _eta1(self,t,T):
        '''
        Governs how each sample in the time series should be weighed.
        This is the default case where each sample has the same weigh, 
        i.e: this is a 'normal' HMM.
        '''
        return 1.
      
    def forwardbackward(self, observations=None):
        '''
        Forward-Backward procedure is used to efficiently calculate the probability of the observation, given the model - P(O|model)
        alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the observation up to time t, given the model.
        
        The returned value is the log of the probability, i.e: the log likehood model, give the observation - logL(model|O).
        
        In the discrete case, the value returned should be negative, since we are taking the log of actual (discrete)
        probabilities. In the continuous case, we are using PDFs which aren't normalized into actual probabilities,
        so the value could be positive.
        '''

        if observations:
            self.set_observations(observations)

        # TODO: this call can be avoided because of performance.
        # The log-likelihood can be computed in _calcstats.

        number_sequences = self.B_maps.shape[0]
        log_likelihood = 0.0
        for i in xrange(number_sequences):
            alpha, scaling_c = self._calcalpha(self.B_maps[i])
            log_likelihood += numpy.log(scaling_c).sum()
        return log_likelihood

    def _calcalpha(self, B_map):
        '''
        Calculates 'alpha' the forward variable.

        '''
        assert len(B_map) > 0
        number_observations = len(B_map[0])
        scaled_alpha = numpy.zeros((number_observations, self.n),
                                   dtype=self.precision)

        scaling_c = numpy.zeros(number_observations)

        for x in xrange(self.n):
            scaled_alpha[0][x] = self.pi[x]*B_map[x][0]

        scaling_c[0] = scaled_alpha[0][:].sum()
        scaled_alpha[0][:] *= (1.0/scaling_c[0])
        # induction
        for t in xrange(1, number_observations):
            for j in xrange(self.n):
                for i in xrange(self.n):
                    scaled_alpha[t][j] += scaled_alpha[t-1][i]*self.A[i][j]
                scaled_alpha[t][j] *= B_map[j][t]
            scaling_c[t] = scaled_alpha[t][:].sum()
            scaled_alpha[t][:] *= (1.0/scaling_c[t])
        return scaled_alpha, scaling_c

    def _calcbeta(self, B_map, scaling_c):

        assert len(B_map) > 0
        number_observations = len(B_map[0])
        scaled_beta = numpy.zeros((number_observations, self.n),
                                  dtype=self.precision)

        # init stage
        for s in xrange(self.n):
            scaled_beta[number_observations-1][s] = 1.
        
        # induction
        for t in xrange(number_observations-2, -1, -1):
            for i in xrange(self.n):
                for j in xrange(self.n):
                    scaled_beta[t][i] += \
                        self.A[i][j]*B_map[j][t+1]*scaled_beta[t+1][j]
            scaled_beta[t][:] *= (1.0/scaling_c[t+1])
        return scaled_beta

    def _viterbi(self, observations=None):
        '''
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.

        delta[t][i] = max(P[q1..qt=i,O1...Ot|model]) - the path ending in Si and until time t,
        that generates the highest probability.

        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1),
        i.e: the previous state.

        Notice that actually the Viterbi algorithm is implemented taking the
        logarithm of the probabilities because of numerical stability issues.
        '''
        # similar to the forward-backward algorithm.
        # For now this method RESETS the input and B_map if input is provided.

        if observations:
            self.set_observations(observations)

        output = numpy.zeros(self.B_maps.shape[0], dtype=object)
        for s in xrange(self.B_maps.shape[0]):
            assert len(self.B_maps[s]) > 0
            n_observations = len(self.B_maps[s][0])
            delta = numpy.zeros((n_observations, self.n), dtype=self.precision)
            psi = numpy.zeros((n_observations, self.n), dtype=self.precision)

            # init
            for x in xrange(self.n):
                delta[0][x] = numpy.log(self.pi[x]) + numpy.log(
                    self.B_maps[s][x][0])
                psi[0][x] = 0

            # induction
            for t in xrange(1, n_observations):
                for j in xrange(self.n):
                    for i in xrange(self.n):
                        if delta[t][j] < delta[t-1][i]+numpy.log(self.A[i][j]):
                            delta[t][j] = delta[t-1][i]+numpy.log(self.A[i][j])
                            psi[t][j] = i
                    delta[t][j] += numpy.log(self.B_maps[s][j][t])

            # termination: find the maximum probability for
            # the entire sequence (=highest prob path)
            p_max = -numpy.inf  # max value in time T (max)
            # the states are discrete.
            path = numpy.zeros(n_observations, dtype=numpy.int)
            for i in xrange(self.n):
                if p_max < delta[n_observations-1][i]:
                    p_max = delta[n_observations-1][i]
                    path[n_observations-1] = i
            assert p_max > -numpy.inf, "There is not a sequence of hidden" \
                                        "states with (numerically) nonzero" \
                                        "probability."
            # path backtracing
            for i in xrange(1, n_observations):
                path[n_observations-i-1] = \
                    psi[n_observations-i][path[n_observations-i]]
            output[s] = path
        return output

    def _calcxi(self, B_map, alpha, beta, scaling_c):
        '''
        Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.
        
        The xi variable is a numpy array indexed by time, state, and state (TxNxN).
        xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        time 't+1' given the entire observation sequence.
        '''
        # There is a bug in the original implementation below
        assert len(B_map) > 0
        number_observations = len(B_map[0])
        xi = numpy.zeros((number_observations - 1, self.n, self.n),
                         dtype=self.precision)
        
        for t in xrange(number_observations - 1):
            denom = scaling_c[t+1]
            for i in xrange(self.n):
                for j in xrange(self.n):
                    numer = 1.0
                    numer *= alpha[t][i]
                    numer *= self.A[i][j]
                    numer *= B_map[j][t+1]
                    numer *= beta[t+1][j]
                    xi[t][i][j] = numer/denom
                    
        return xi

    def _calcgamma(self, seq_len, scaled_alpha, scaled_beta):
        '''
        Calculates 'gamma' from alpha and beta.
        
        Gamma is a (TxN) numpy array, where gamma[t][i] = the probability of being
        in state 'i' at time 't' given the full observation sequence.
        '''        
        gamma = numpy.zeros((seq_len,self.n),dtype=self.precision)

        # There is an important bug here in the original implementation. Since
        # gamma is compute from xi and the last row of xi is full of zeros the
        # resulting row of gamma for the last position is zero as well.

        # TODO: Something better (i.e OOP) can be done with alpha, beta and
        # observations. Currently there are some functions which need other
        # functions to be called beforehand and this dependency isn't explicit
        # right now.
        # TODO: Setter method for observations.


        for t in xrange(seq_len):
            for i in xrange(self.n):
                gamma[t][i] = scaled_alpha[t][i] * scaled_beta[t][i]
        
        return gamma
    
    def train(self, observations=None, iterations=100,
              epsilon=0.0001, thres=-0.001):
        '''
        Updates the HMMs parameters given a new set of observed sequences.
        
        observations can either be a single (1D) array of observed symbols, or when using
        a continuous HMM, a 2D array (matrix), where each row denotes a multivariate
        time sample (multiple features).
        
        Training is repeated 'iterations' times, or until log likelihood of the model
        increases by less than 'epsilon'.
        
        'thres' denotes the algorithms sensitivity to the log likelihood decreasing
        from one iteration to the other.
        '''
        if observations:
            self.set_observations(observations)
        
        for i in xrange(iterations):
            prob_old, prob_new = self.trainiter()

            if (self.verbose):      
                print "iter: ", i, ", L(model|O) =", prob_old, ", L(model_new|O) =", prob_new, ", converging =", ( prob_new-prob_old > thres )
                
            if ( abs(prob_new-prob_old) < epsilon ):
                # converged
                break
                
    def _updatemodel(self,new_model):
        '''
        Replaces the current model parameters with the new ones.
        '''
        self.pi = new_model['pi']
        self.A = new_model['A']
                
    def trainiter(self):
        '''
        A single iteration of an EM algorithm, which given the current HMM,
        computes new model parameters and internally replaces the old model
        with the new one.
        
        Returns the log likelihood of the old model (before the update),
        and the one for the new model.

        This method assumes that the observations were already set through
        _mapB.
        '''        
        # call the EM algorithm
        new_model = self._baumwelch()
        
        # calculate the log likelihood of the previous model
        prob_old = self.forwardbackward()
        
        # update the model with the new estimation
        self._updatemodel(new_model)

        # Since the emission parameters were reestimated the emission
        # probabilities need to be computed again. This can be avoided
        # sometimes for efficiency (e.g the emission parameters are fixed).
        self._mapB()
        
        # calculate the log likelihood of the new model.
        prob_new = self.forwardbackward()
        
        return prob_old, prob_new
    
    def _reestimateA(self, xis, gammas):
        '''
        Reestimation of the transition matrix (part of the 'M' step of Baum-Welch).
        Computes A_new = expected_transitions(i->j)/expected_transitions(i)
        
        Returns A_new, the modified transition matrix. 
        '''
        assert len(xis) == len(gammas)
        n_sequences = len(xis)
        A_new = numpy.zeros((self.n, self.n), dtype=self.precision)
        for i in xrange(self.n):
            for j in xrange(self.n):
                numer = 0.0
                denom = 0.0
                for s in xrange(n_sequences):
                    xi = xis[s]
                    gamma = gammas[s]
                    n_observations = len(gamma)
                    for t in xrange(n_observations - 1):
                        numer += (self._eta(t, n_observations - 1)*xi[t][i][j])
                        denom += (self._eta(t, n_observations - 1)*gamma[t][i])
                A_new[i][j] = numer/denom
        return A_new

    def _reestimatePi(self, gammas):
        '''
        Estimation of the initial state density from multiple training sequences
        using the gamma values for each sequence.
        '''
        n_sequences = len(gammas)
        pi = numpy.zeros(self.n)
        for i in xrange(n_sequences):
            pi += gammas[i][0]
        pi /= n_sequences
        return pi

    def _calcstats(self):
        '''
        Calculates required statistics of the current model, as part
        of the Baum-Welch 'E' step.
        
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.
        
        Returns 'stat's, a dictionary containing required statistics.

        This method assumes that the observations were already set through
        _mapB.
        '''
        stats = {}

        number_sequences = self.B_maps.shape[0]
        alphas = numpy.zeros(number_sequences, dtype=object)
        betas = numpy.zeros(number_sequences, dtype=object)
        xis = numpy.zeros(number_sequences, dtype=object)
        gammas = numpy.zeros(number_sequences, dtype=object)
        # log_likelihood = 0.0
        for i in xrange(number_sequences):
            assert len(self.B_maps[i]) > 0
            n_observations = len(self.B_maps[i][0])
            alpha, scaling_c = self._calcalpha(self.B_maps[i])
            beta = self._calcbeta(self.B_maps[i], scaling_c)
            xi = self._calcxi(self.B_maps[i], alpha, beta, scaling_c)
            gamma = self._calcgamma(n_observations, alpha, beta)
            alphas[i] = alpha
            betas[i] = beta
            xis[i] = xi
            gammas[i] = gamma
            # log_likelihood += numpy.log(scaling_c).sum()

        stats['alphas'] = alphas
        stats['betas'] = betas
        stats['xis'] = xis
        stats['gammas'] = gammas
        # stats['ll'] = log_likelihood
        
        return stats
    
    def _reestimate(self, stats):
        '''
        Performs the 'M' step of the Baum-Welch algorithm.

        Deriving classes should override (extend) this method to include
        any additional computations their model requires.

        Returns 'new_model', a dictionary containing the new maximized
        model's parameters.

        This method assumes that the observations were already set
        through _mapB.
        '''
        new_model = {
            'pi': self._reestimatePi(stats['gammas']),
            'A': self._reestimateA(stats['xis'], stats['gammas']),
            # Reestimate the observation model in the child classes.
        }
        return new_model
    
    def _baumwelch(self):
        '''
        An EM(expectation-modification) algorithm devised by Baum-Welch. Finds a local maximum
        that outputs the model that produces the highest probability, given a set of observations.
        
        Returns the new maximized model parameters.

        This method assumes that the observations were already set
        through _mapB.
        '''        
        # E step - calculate statistics
        stats = self._calcstats()
        
        # M step
        return self._reestimate(stats)

    def set_observations(self, observations):
        if type(observations) is not list:
            raise TypeError("Expecting a list of observations as input.")
        self.observations = observations
        self._mapB()

    def _mapB(self):
        '''
        Deriving classes should implement this method, so that it maps the observations'
        mass/density Bj(Ot) to Bj(t).
        
        This method has no explicit return value, but it expects that 'self.B_map' is internally computed
        as mentioned above. 'self.B_map' is an (TxN) numpy array.
        
        The purpose of this method is to create a common parameter that will conform both to the discrete
        case where PMFs are used, and the continuous case where PDFs are used.
        
        For the continuous case, since PDFs of vectors could be computationally 
        expensive (Matrix multiplications), this method also serves as a caching mechanism to significantly
        increase performance.
        '''
        raise NotImplementedError("a mapping function for B(observable probabilities) must be implemented")







