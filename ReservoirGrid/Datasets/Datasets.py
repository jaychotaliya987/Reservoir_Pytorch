import numpy as np
import matplotlib.pyplot as plt

class LorenzAttractor:
    def __init__(self, n=10000, dt=0.01, sigma=10, rho=28, beta=8/3):
        self.n = n
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def generate(self):
        x = np.zeros(self.n)
        y = np.zeros(self.n)
        z = np.zeros(self.n)

        x[0] = 0
        y[0] = 1
        z[0] = 1.05

        for i in range(1, self.n):
            x[i] = x[i-1] + self.dt * self.sigma * (y[i-1] - x[i-1])
            y[i] = y[i-1] + self.dt * (x[i-1] * (self.rho - z[i-1]) - y[i-1])
            z[i] = z[i-1] + self.dt * (x[i-1] * y[i-1] - self.beta * z[i-1])

        return x, y, z

    def plot(self):
        x, y, z = self.generate()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.set_title("Lorenz Attractor")
        plt.show()


class MackeyGlass:
    def __init__(self, n=1000, beta=0.2, gamma=0.1, tau=17, dt=0.1):
        self.n = n
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.dt = dt
        self.x = np.zeros(n)

    def generate(self):
        # Initial condition
        self.x[0] = 1.2

        for t in range(1, self.n):
            if t < self.tau:
                self.x[t] = self.x[t-1]  # No feedback until tau
            else:
                # Mackey-Glass equation
                self.x[t] = self.x[t-1] + self.dt * (self.beta * self.x[t - 1 - int(self.tau)] /
                                                      (1 + self.x[t - 1 - int(self.tau)]**10) -
                                                      self.gamma * self.x[t - 1])
        return self.x

    def plot(self):
        self.generate()
        plt.figure()
        plt.plot(self.x, label='Mackey-Glass')
        plt.title("Mackey-Glass Attractor")
        plt.xlabel('Time Steps')
        plt.ylabel('x(t)')
        plt.legend()
        plt.show()
