from ndlutil import jitChol
import numpy as np
from scipy import optimize, integrate
from scipy.special import wofz

class lfm2():
    def __init__(self, Q, noutputs, params=None):
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
        self.params = np.concatenate((np.log(self.B), np.log(self.C),
                                      np.hstack(self.S), np.log(self.l),
                                      np.log(self.sn)), axis=0)
        if params is not None:
            self.params = params
        self.l2pi = np.log(2.*np.pi)
        # Forcing to not update without observations.
        self.set_params(self.params, False)  # forcing to not update.
        self.updated = False  # flag to keep track if the model is 'updated'.

    def set_inputs(self, t, ind):
        self.t = t
        self.ind = ind
        self.updated = False

    def set_outputs(self, y):
        self.y = y.reshape((-1, 1))
        self.updated = False
        self._update() # at this point we are ready to go since it's assumed
        # that the corresponding inputs and outputs are set.

    def set_params(self, params, update=True):
        #params=params.flatten()
        assert np.size(params) == self.nvar
        self.params = params
        for q in range(self.Q):
            self.l[q] = np.exp(params[(2+self.Q)*self.D+q])
        for d in range(self.D):
            self.B[d] = np.exp(params[d]) #Spring coefficients
            self.C[d] = np.exp(params[d+self.D]) #Damper coefficients
            self.sn[d] = np.exp(params[(2+self.Q)*self.D+self.Q+d])
            for q in range(self.Q):
                self.S[d][q] = params[2*self.D+q+d*self.Q]
        self.updated = False
        if update:
            self._update()

    def _update(self):
        self.updated = True
        self.K = self.Kyy()
        self._L, jitter = jitChol(self.K)
        invU = np.linalg.solve(self._L,np.eye(self._L.shape[0]))
        Ki = np.dot(invU, invU.T)
        self.alpha = np.dot(Ki, self.y)

    def LLeval(self):
        assert self.updated
        self.LL = -.5*(np.dot(self.y.T, self.alpha)[0][0]+self.y.size*self.l2pi + 2.*np.log(np.diag(self._L)).sum())
        return self.LL

    def Kyy(self):
        K = self.Kff(self.t, self.ind)
        n = range(K.shape[0])
        K[n,n] += self.sn[self.ind].reshape((self.ind.size,)) #Adding noise
        return K

    def Kff(self,t,index,tp=None,indexp=None):
        other = t
        if tp is not None:
            other = tp
        K = np.zeros((t.size,other.size))
        for q in range(self.Q):
            K += (self.S[index,q]*self.S[index,q].T)*self.Kff_q(self.l[q],t,index,tp,indexp)
        return K


    def Kff_q(self,lq,t,index,tp=None,indexp=None):
        if tp is None:
            tp = t.copy()
            indexp = index.copy()

        def hnew(gam1, gam2, gam3, t, tp, wofznu, nu, nu2):
            c1_1 = 1./(gam2 + gam1)
            c1_2 = 1./(gam3 + gam1)

            tp_lq = tp/lq
            t_lq = t/lq
            dif_t_lq = t_lq - tp_lq
            #Exponentials
            #egam1tp = np.exp(-gam1*tp)
            gam1tp = gam1*tp
            egam2t = np.exp(-gam2*t)
            egam3t = np.exp(-gam3*t)
            #squared exponentials
            #edif_t_lq = np.exp(-dif_t_lq*dif_t_lq)
            dif_t_lq2 = dif_t_lq*dif_t_lq
            #et_lq = np.exp(-t_lq*t_lq)
            t_lq2 = t_lq*t_lq
            #etp_lq = np.exp(-tp_lq*tp_lq)
            tp_lq2 = tp_lq*tp_lq
            ec = egam2t*c1_1 - egam3t*c1_2

            #Terms of h
            A = dif_t_lq + nu
            temp = np.zeros(A.shape, dtype = complex)
            boolT = A.real >=0.
            if np.any(boolT):
                wofzA = wofz(1j*A[boolT])
                temp[boolT] = np.exp(np.log(wofzA) - dif_t_lq2[boolT])
            boolT = np.logical_not(boolT)
            if np.any(boolT):
                dif_t = t[boolT] - tp[boolT]
                wofzA = wofz(-1j*A[boolT])
                temp[boolT] = 2.*np.exp(nu2[boolT] + gam1[boolT]*dif_t) - np.exp(np.log(wofzA) - dif_t_lq2[boolT])

            b = t_lq + nu
            wofzb = wofz(1j*b)
            e = nu - tp_lq
            temp2 = np.zeros(e.shape, dtype = complex)
            boolT = e.real >= 0.
            if np.any(boolT):
                wofze = wofz(1j*e[boolT])
                temp2[boolT] = np.exp(np.log(wofze) - tp_lq2[boolT])
            boolT = np.logical_not(boolT)
            if np.any(boolT):
                wofze = wofz(-1j*e[boolT])
                temp2[boolT] = 2.*np.exp(nu2[boolT] - gam1tp[boolT]) - np.exp(np.log(wofze) - tp_lq2[boolT])
            return (c1_1 - c1_2)*(temp \
                - np.exp(np.log(wofzb) - t_lq2 - gam1tp))\
                - ec*( temp2 \
                - np.exp(np.log(wofznu) - gam1tp))

        indexp = indexp.reshape((1,indexp.size))
        alpha = self.C/2.
        w = np.sqrt(4.*self.B - self.C*self.C + 0j)/2.
        wbool = self.C*self.C>4.*self.B
        wbool = np.logical_or(wbool[:,None],wbool)

        ind2t, ind2tp = np.where(wbool[index,indexp])
        ind3t, ind3tp = np.where(np.logical_not(wbool[index,indexp])) #TODO: from the original index can be done

        gam = alpha + 1j*w
        gamc = alpha - 1j*w
        W = w*w.reshape((w.size,1))
        K0 = lq*np.sqrt(np.pi)/(8.*W[index, indexp])
        nu = lq*gam/2.
        nu2 = nu*nu
        wofznu = wofz(1j*nu)

        kff = np.zeros((t.size, tp.size), dtype = complex)

        t = t.reshape(t.size,)
        tp = tp.reshape(tp.size,)
        index = index.reshape(index.size,)
        indexp = indexp.reshape(indexp.size,)
        indbf, indbc = np.where(np.ones(kff.shape, dtype=bool))
        index2 = index[indbf]
        index2p = indexp[indbc]
        #Common computation for both cases
        kff[indbf,indbc] = hnew(gam[index2p], gamc[index2], gam[index2], t[indbf], tp[indbc], wofznu[index2p], nu[index2p], nu2[index2p]) \
        + hnew(gam[index2], gamc[index2p], gam[index2p], tp[indbc], t[indbf], wofznu[index2], nu[index2], nu2[index2])

        #Now we calculate when w_d or w_d' are not real
        if np.any(wbool):
            #Precomputations
            nuc = lq*gamc/2.
            nuc2 = nuc**2
            wofznuc = wofz(1j*nuc)
            #A new way to work with indexes
            ind = index[ind2t]
            indp = indexp[ind2tp]
            t1 = t[ind2t]
            t2 = tp[ind2tp]
            kff[ind2t, ind2tp] += hnew(gamc[indp], gam[ind], gamc[ind], t1, t2, wofznuc[indp], nuc[indp], nuc2[indp])\
             + hnew(gamc[ind], gam[indp], gamc[indp], t2, t1, wofznuc[ind], nuc[ind], nuc2[ind])

        #When wd and wd' ares real
        if np.any(np.logical_not(wbool)):
            kff[ind3t, ind3tp] = 2.*np.real(kff[ind3t, ind3tp])

        return (K0 * kff).real

    #Prediction
    def predict(self,ts, inds):
        kfsf = self.Kff(ts, inds, self.t, self.ind)
        kfsfs = self.Kff(ts, inds)
        #Posterior for outputs
        ms = np.dot(kfsf,self.alpha)
        #v = np.linalg.solve( self._L, kfsf.T)
        #print 'min var', v.min()
        #Kysys = kfsfs - np.dot(v.T, v)
        var = np.diag(kfsfs) - (kfsf.T*np.linalg.solve(self.K, kfsf.T)).sum(0)
        return ms, var

    def plot_predict(self,ts, inds):
        ms, var = self.predict(ts, inds)
        print var.min()
        #var = np.diag(Kysys)
        f, axarr = plt.subplots(self.D, sharex=True)
        for k in range(self.D):
            indexk = (self.ind == k).reshape((self.ind.size,))
            axarr[k].plot(self.t[indexk], self.y[indexk],'k+', markersize=7, linewidth=3)
            indexk = (inds == k).reshape((inds.size,))
            axarr[k].plot(ts[indexk], ms[indexk],'-k', markersize=7, linewidth=3)
            axarr[k].plot(ts[indexk], ms[indexk] - 2.*np.sqrt(var[indexk] ),'--k')
            axarr[k].plot(ts[indexk], ms[indexk] + 2.*np.sqrt(var[indexk] ),'--k')
	plt.show()

    def lfm_LL_fn(self, p ):
        self.set_params( p )
        return -self.LLeval()

    def lfm_LL_deriv_fn(self, p ):
        self.set_params( p )
        grad = np.asarray(self.LFM2_grad())
        return -grad

    def Optimize(self):
        #results = optimize.minimize(self.lfm_LL_fn, self.params, jac=self.lfm_LL_deriv_fn)
        results = optimize.minimize(self.lfm_LL_fn, self.params)
        print results.fun
        self.set_params(results.x)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    #Simulation

    #System's parameters
    lq=[1.]
    B=[5.,1.]
    C=[3.,3.]
    S=[[1.],[1.]]
    Q=1
    D=2
    t=[0]*D
    tt = [0]*D
    y = [0.]*D
    #Simulation
    def calc_deri(yvec, time, nuc, omc,s):
        return (yvec[1], 1.*s -nuc * yvec[1] - omc * yvec[0])
    t2 = np.linspace(0., 10., 100)
    for d in range(D):
        nu_c = C[d]
        om_c = B[d]
        yarr = integrate.odeint(calc_deri, (0, 0), t2, args=(nu_c, om_c, S[d][0]))
        y[d] = np.asarray(yarr[:,0]).T + 0.01*np.random.randn(np.size(t2))


    t2 = t2.reshape((t2.size,))
    index = np.array([np.zeros((1,t2.size),dtype = np.int8), np.ones((1,t2.size), dtype=np.int8)])
    index = index.reshape((index.size,1))

    t = np.asarray(np.append(t2,t2))
    y = np.asarray(np.append(y[0],y[1]))


    #LFM model creation
    lfm=lfm2(Q,2)
    lfm.set_inputs(t, index)
    lfm.set_outputs(y)
    paraini = lfm.params
    # Optimize hyperparameters
    #params=np.concatenate((np.log(B),np.log(C),np.hstack(S),np.log(lq),np.log([100, 100])))
    lfm.Optimize()
    lfm.plot_predict(t, index)
