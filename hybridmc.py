import numpy as np
from math import exp
import matplotlib.pyplot as plt

'''
有关调参：
    1.似乎leap_frog_loop次数多一些采出来会更准确
    2.但是leap_frog_loop次数多了以后acceptance_rate明显下降
    
    何解？？？
'''

class hybridMonteCarloGaussian(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.sigma_inv = np.linalg.inv(sigma)
        self.samples = list()

    def potential(self, x_t):
        # override this function for sampling some other pdf
        x_center = x_t - self.mu
        potential = x_center.transpose().dot(self.sigma_inv).dot(x_center)[0, 0]
        return 0.5 * potential

    def kinetic(self, p_t):
        kinetic = p_t.transpose().dot(p_t)[0, 0]
        return 0.5 * kinetic

    def acceptance_rate(self, num):
        return float(len(self.samples)) / float(num)

    def leap_frog(self, p_t, x_t, delta=0.3, loop=20):
        p_delta = p_t
        p_half = p_t - (delta/2) * self.sigma_inv.dot(x_t - self.mu)
        x_delta = x_t + delta * p_half
        for i in range(loop):
            p_half = p_t - (delta / 2) * self.sigma_inv.dot(x_t - self.mu)
            x_delta = x_delta + delta * p_half
            p_delta = p_half - (delta/2) * self.sigma_inv.dot(x_delta - self.mu)
        return p_delta, x_delta

    def sample(self, num=3000, leap_frog_loop=20):
        x_0 = np.random.normal(size=(2, 1))
        for t in range(num):
            p_0 = np.random.normal(size=(2, 1))

            # Leap Frog
            p_star, x_star = self.leap_frog(p_0, x_0, loop=leap_frog_loop)

            # Metropolis Acceptance Probability
            alpha = exp(self.potential(x_0) - self.potential(x_star) + self.kinetic(p_0) - self.kinetic(p_star))
            if np.random.uniform(0.0, 1.0) <= alpha:
                self.samples.append(x_star)
                x_0 = x_star
            else:
                continue
        return self.acceptance_rate(num)

    def show(self):
        print('%d samples to show' % len(self.samples))
        ps = np.concatenate(self.samples, axis=1)
        plt.figure()
        plt.scatter(ps[0, :], ps[1, :], marker='x')
        plt.show()


if __name__ == '__main__':
    mu = np.array([[7.0, 7.0]], dtype=np.float32).transpose()
    sigma = np.array([[100.0, 0.0],
                      [0.0, 1.0]], dtype=np.float32)
    gaussian = hybridMonteCarloGaussian(mu=mu, sigma=sigma)

    acc = gaussian.sample(num=5000, leap_frog_loop=15)
    print(acc)
    gaussian.show()