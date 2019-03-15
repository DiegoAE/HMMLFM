from hmm.continuous.LFMHMMcontinuousMO import LFMHMMcontinuousMO
from matplotlib import pyplot as plt
import numpy as np
import os

number_lfm = 3
outputs = 4
start_t = 0.1
end_t = 5.1
locations_per_segment = 20
pi = np.array([0.1, 0.3, 0.6])
A = np.array([[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.3, 0.2, 0.5]])

damper_constants = np.asarray(
        [[1., 3., 7.5, 10.],
         [3., 10., 0.5, 0.1],
         [6., 5., 4., 9.]]
)

spring_constants = np.asarray(
        [[3., 1, 2.5, 10.],
         [1., 3.5, 9.0, 5.0],
         [5., 8., 4.5, 1.]]
)

# implicitly assuming there is only one latent force governing the system.
lengthscales = np.asarray([[10.], [2.], [5.]])

noise_var = np.array([0.0005, 0.005, 0.0025, 0.0008])

lfm_hmm = LFMHMMcontinuousMO(outputs, number_lfm, locations_per_segment,
                             start_t, end_t, verbose=True)
lfm_hmm.set_params(A, pi, damper_constants, spring_constants, lengthscales,
                   noise_var)

dummy_model = LFMHMMcontinuousMO(outputs, number_lfm, locations_per_segment,
                             start_t, end_t, verbose=True)
dummy_model.read_params(os.path.realpath("../PretrainedModels/TOY"),
                        "toy_LFM")
#The following expression of relative path also works
#dummy_model.read_params(os.getcwd()+"/../PretrainedModels/TOY","toy_LFM")

#This following expression was for the original arthor's convenience
#dummy_model.read_params("/home/diego/tmp/Parameters/WALKING", "toy_LFM")

# plotting covariances

def transform_covariance(cov):
    ret = cov.copy()
    rows, cols = cov.shape
    lps = locations_per_segment
    for r in range(rows):
        for o in range(outputs):
            ret[r][lps * o:lps * (o + 1)] = cov[r][o::outputs]
    nret = ret.copy()
    for o in range(outputs):
        nret[lps * o:lps * (o + 1)] = ret[o::outputs]
    return nret

plt.figure()
for i in range(lfm_hmm.n):
    if (i == 0):
        plt.xlabel("hola")
    plt.subplot(2, 3, i + 1)
    plt.imshow(transform_covariance(lfm_hmm.get_cov_function(i, False)))
    if i == 1:
        plt.title('(a)')
    plt.axis('off')
    plt.subplot(2, 3, i + 4)
    if (i < 2):
        plt.imshow(transform_covariance(
                dummy_model.get_cov_function((i+1)%2, False)))
    else:
        plt.imshow(transform_covariance(dummy_model.get_cov_function(i, False)))
    if i == 1:
        plt.title('(b)')
    plt.axis('off')
plt.show()
