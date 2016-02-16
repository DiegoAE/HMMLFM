import GPy
from matplotlib import pyplot as plt
import numpy as np

data = GPy.util.datasets.cmu_mocap(subject='43', train_motions=['01'],
                                   sample_every=1)
Y = data['Y']
animation_flag = False
if animation_flag:
    prueba = Y[70:203, :]
    Y[:, 0:3] = 0.   # Make figure walk in place
    visualize = GPy.plotting.matplot_dep.visualize.skeleton_show(
            Y[0, :], data['skel'])
    GPy.plotting.matplot_dep.visualize.data_play(prueba, visualize, 30)
    plt.close()


ltibia_id = 9
rtibia_id = 16

nsamples, nfeatures = Y.shape
channel_id = ltibia_id
plt.plot(np.arange(nsamples), Y[:, channel_id], color='blue')
plt.axvline(x=70, color='red')
plt.axvline(x=203, color='red')
plt.axvline(x=336, color='red')
plt.show()

