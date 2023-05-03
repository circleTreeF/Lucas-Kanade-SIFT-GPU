import numpy as np
from KDEpy import NaiveKDE, TreeKDE, FFTKDE

"""
My implementation of KDE
data: images shall be flattened and in size of (n_data_point,n_features)
References:
Silverman, B.W. (1998). Density Estimation for Statistics and Data Analysis (1st ed.). Routledge. https://doi.org/10.1201/9781315140919
"""
def distribution_normalization(input):
    data_std = np.std(input, axis=0)
    data_mean = np.mean(input, axis=0)
    norm = (input - data_mean) / (data_std + 1e-32) + 1e-32
    return norm


def distribution_modifier(data_set, __bandwidth=1):
    norm_x = distribution_normalization(data_set)
    naive_kde = NaiveKDE(kernel='gaussian', bw=__bandwidth).fit(norm_x)
    return naive_kde

class KDE:
    """
    This is a class to manage and evaluate the probability density function with kernel density estimation approach
    """

    def __init__(self, __sample, __h=1):
        """
        Construct an insetane of distribution manager with sample and bandwidth
        params: __sample: in shape of [n_data_point, n_features]
        """
        self.sample = __sample
        self.h = __h

    def __gaussian_kernel__(self, input):
        """
        Calculate the Gaussian kernel given input
        :param input:
        :return: input in Gaussian knernel
        """
        # (1. / np.sqrt(2 * np.pi)) * np.exp(-(Xi - x) ** 2 / (h ** 2 * 2.))
        # K(\mathbf{x})=(2 \pi)^{-d / 2} \exp \left(-\frac{1}{2} \mathbf{x}^{\mathbf{T}} \mathbf{x}\right)
        kernel_out = (1 / np.power(2 * np.pi, self.sample.shape[1] / 2)) * np.exp(-np.matmul(input, input.T) / 2)
        return kernel_out

    def my_kde(self, data_point):
        kernel = 0.
        for X_i in self.sample:
            kernel += self.__gaussian_kernel__((data_point - X_i) / self.h)
        return kernel / (self.sample.shape[0] * np.power(self.h, self.sample.shape[1]))

