'''
Created on Nov 12, 2012

@author: GuyZ
'''

from hmm._BaseHMM import _BaseHMM
import numpy

class DiscreteHMM(_BaseHMM):
    '''
    A Discrete HMM - The most basic implementation of a Hidden Markov Model,
    where each hidden state uses a discrete probability distribution for
    the physical observations.
    
    Model attributes:
    - n            number of hidden states
    - m            number of observable symbols
    - A            hidden states transition probability matrix ([NxN] numpy array)
    - B            PMFs denoting each state's distribution ([NxM] numpy array)
    - pi           initial state's PMF ([N] numpy array).
    
    Additional attributes:
    - precision    a numpy element size denoting the precision
    - verbose      a flag for printing progress information, mainly when learning
    '''

    def __init__(self,n,m,A=None,B=None,pi=None,init_type='uniform',precision=numpy.double,verbose=False):
        '''
        Construct a new Discrete HMM.
        In order to initialize the model with custom parameters,
        pass values for (A,B,pi), and set the init_type to 'user'.
        
        Normal initialization uses a uniform distribution for all probablities,
        and is not recommended.
        '''
        _BaseHMM.__init__(self,n,m,precision,verbose) #@UndefinedVariable
        
        self.A = A
        self.pi = pi
        self.B = B
        
        self.reset(init_type=init_type)

    def reset(self,init_type='uniform'):
        '''
        If required, initalize the model parameters according the selected policy
        '''
        if init_type == 'uniform':
            self.pi = numpy.ones( (self.n), dtype=self.precision) *(1.0/self.n)
            self.A = numpy.ones( (self.n,self.n), dtype=self.precision)*(1.0/self.n)
            # self.B = numpy.ones( (self.n,self.m), dtype=self.precision)*(1.0/self.m)
            # TODO: allow the emission estimation, i.e. reestimateB
    
    def _mapB(self):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        '''

        if not self.observations:
            raise ValueError("The training sequences haven't been set.")

        numbers_of_sequences = len(self.observations)

        self.B_maps = numpy.zeros((numbers_of_sequences, self.n), dtype=object)

        for j in xrange(numbers_of_sequences):
            for i in xrange(self.n):
                self.B_maps[j][i] = numpy.zeros(len(self.observations[j]),
                                                dtype=self.precision)

        for j in xrange(numbers_of_sequences):
            B_map = self.B_maps[j]
            for i in xrange(self.n):
                sequence_len = len(B_map[i])
                for t in xrange(sequence_len):
                    B_map[i][t] = self.B[i][self.observations[j][t]]

    def generate_observations(self, segments):
        output = numpy.zeros(segments, dtype=numpy.int)
        initial_state = numpy.nonzero(numpy.random.multinomial(1, self.pi))[0][0]
        hidden_states = [initial_state]
        for x in xrange(1, segments):
            hidden_states.append(numpy.nonzero(numpy.random.multinomial(
                1, self.A[hidden_states[-1]]))[0][0])
        for i in xrange(len(hidden_states)):
            state = hidden_states[i]
            v = numpy.random.multinomial(1, self.B[state])
            output[i] = numpy.nonzero(v)[0][0]
        print "Hidden States", hidden_states
        return output



if __name__ == '__main__':
    numpy.random.seed(123456)
    pi = numpy.array([0.1, 0.3, 0.6])
    print "initial state distribution", pi
    A = numpy.array([[0.1, 0.5, 0.4], [0.6, 0.1, 0.3], [0.4, 0.5, 0.1]])
    print "hidden state transition matrix\n", A
    # B = numpy.array([[0.25, 0.12, 0.13,  0.5], [0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]])
    B = numpy.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    # for x in xrange(len(B)):
    #     print B[x].sum()
    print "observations matrix\n", B
    dhmm = DiscreteHMM(3, 3, A, B, pi, init_type=None, verbose=True)
    # generate obs
    n_sequences = 100
    obs = []
    for i in xrange(n_sequences):
        length = numpy.random.randint(1, 100)
        obs.append(dhmm.generate_observations(length))
    dhmm.reset()
    # train
    dhmm.set_observations(obs)
    dhmm.train()

    print dhmm.A
    print dhmm.pi
    print dhmm.B