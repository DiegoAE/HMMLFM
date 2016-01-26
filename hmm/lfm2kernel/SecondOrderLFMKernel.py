import numpy as np
from scipy.special import wofz

def kffs(B,C,t,index,tp,indexp,lq, noise_var):

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
    alpha = C/2.
    w = np.sqrt(4.*B - C*C + 0j)/2.
    wbool = C*C>4.*B
    wbool = np.logical_or(wbool[:,None],wbool) # here np.logical_or behaves similarly to the bsxfun matlab function.

    # wbool shape (n, n) where n is equal to len(B) and len(C).
    # wbool[index,indexp] this creates a matrix from the column vector index and row vector indexp.

    ind2t, ind2tp = np.where(wbool[index,indexp])
    ind3t, ind3tp = np.where(np.logical_not(wbool[index,indexp])) #TODO: from the original index can be done

    gam = alpha + 1j*w
    gamc = alpha - 1j*w
    # * operator is elementwise multiplication when arrays are used
    W = w*w.reshape((w.size,1))
    K0 = lq*np.sqrt(np.pi)/(8.*W[index, indexp])
    nu = lq*gam/2.
    nu2 = nu**2
    wofznu = wofz(1j*nu)

    kff = np.zeros((t.size, tp.size), dtype = complex)

    # deleting one dimension
    t = t.reshape(t.size,)
    tp = tp.reshape(tp.size,)
    index = index.reshape(index.size,)
    indexp = indexp.reshape(indexp.size,)

    # All indexes of a matrix of shape kff.shape
    indbf, indbc = np.where(np.ones(kff.shape, dtype=bool))


    # To which observation do I belong? in the Kff matrix spacce
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
    # TODO:for now there is only one noise affecting all the outputs.
    # And there should be one for each output.
    return (K0 * kff).real + np.eye(t.size, tp.size) * noise_var


def K(B, C, lq, t, noise_var):
    """ Computes the kernel covariance function for the second order LFM, using t against itself, k(t, t).
    Assumptions:
        *the output functions are evaluated in the same time steps.
        *There is only one input latent force with RBF lengthscale lq
        *Returns the kernel evaluation of t against itself."""
    assert len(B) == len(C)
    D = len(B)
    idx = np.zeros(shape=(0,1), dtype = np.int8)
    time_length = len(t)
    stacked_time = np.zeros(shape=(0,1))
    for d in xrange(D):
        idx = np.vstack((idx, d * np.ones((time_length,1), dtype = np.int8)))
        stacked_time = np.vstack((stacked_time, t))
    return kffs(B, C, stacked_time, idx, stacked_time, idx, lq, noise_var)


def K_pred(B, C, lq, t, t_pred):
    """ Computes the kernel covariance function for the second order LFM, using t against t_pred, k(t, t_pred).
    Assumptions:
        *the output functions are evaluated in the same time steps.
        *There is only one input latent force with RBF lengthscale lq.
        *Returns the kernel evaluation of t against t_pred."""
    assert len(B) == len(C)
    D = len(B)
    idx_t = np.zeros(shape=(0, 1), dtype = np.int8)
    time_length_t = len(t)
    stacked_time_t = np.zeros(shape=(0, 1))
    for d in xrange(D):
        idx_t = np.vstack((idx_t,
                           d * np.ones((time_length_t,1), dtype=np.int8)))
        stacked_time_t = np.vstack((stacked_time_t, t))
    idx_t_pred = np.zeros(shape=(0, 1), dtype = np.int8)
    time_length_t_pred = len(t_pred)
    stacked_time_t_pred = np.zeros(shape=(0, 1))
    for d in xrange(D):
        idx_t_pred = np.vstack(
            (idx_t_pred, d * np.ones((time_length_t_pred, 1), dtype=np.int8)))
        stacked_time_t_pred = np.vstack((stacked_time_t_pred, t_pred))
    # Noise variance is assumed 0 here because this function is intended
    # to return the covariance between training data and prediction data.
    return kffs(B, C, stacked_time_t, idx_t,
                stacked_time_t_pred, idx_t_pred, lq, 0.0)



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    Bd= 1. #spring
    Bdp= 3.
    Cd = 3.
    Cdp = 1. #damper
    lq= 100.
    noise_var = 0.001

    ND1= 50
    ND2= 50
    t1 = np.linspace(0,10.,ND1)
    t1 = t1.reshape((t1.size,))
    t2 = np.linspace(0,10.,ND2)
    t2 = t2.reshape((t2.size,))
    index = np.array([np.zeros((1,t1.size),dtype = np.int8), np.ones((1,t2.size), dtype=np.int8)])
    index = index.reshape((index.size,1))
    indexp = index

    t = np.append(t1,t2)
    tp = t.copy()
    t = t.reshape((t.size,1))
    tp = tp.reshape((tp.size,1))

    B = np.asarray([Bd, Bdp])
    C = np.asarray([Cd, Cdp])

    cov = kffs(B, C, t, index, tp, indexp, lq, noise_var)

    plt.figure(1)
    plt.imshow(cov)
    plt.show()

    realization = np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=np.real(cov))

    plt.figure(2)
    plt.plot(t1, realization[:ND1])
    plt.plot(t1, realization[ND1:])
    plt.show()

    # testing K_pred

    B = np.asarray([Bd])
    C = np.asarray([Cd])
    t_ = np.array([1., 2., 3., 4.]).reshape((-1, 1))
    t_pred = np.array([2]).reshape((-1, 1))
    cov2 = K_pred(B, C, lq, t_, t_pred)
    #print cov2.shape
    plt.figure(3)
    plt.imshow(cov2)
    plt.show()