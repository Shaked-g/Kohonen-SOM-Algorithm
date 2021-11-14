import math
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.random.mtrand import dirichlet


class SOM:
    def __init__(self, h, w, feat_dim):
        self.shape = (h, w, feat_dim)
        # self.som = np.zeros((h, w, feat_dim))

        xs = np.linspace(0, 1, w)
        array = [[0 for c in range(2)] for r in range(w)]
        i = 0
        for x in xs:
            array[i][0] = x
            i += 1
        nparray = np.array(array)
        self.som = nparray.reshape(h, w, feat_dim)

        self.L0 = 0.0  # the initial learning rate.
        self.lam = 0.0  # a time scaling constant.
        self.sigma0 = 0.0  # the initial sigma.

    def train(self, data, iterations, uniform):
        """
        self: the SOM model
        :param data: the data to be trained on.
        :param iterations: number of iterations.
        :param uniform: if 1 preforms uniform distribution sampling of the data
        if 0 than non-uniform.
        :return:
        """
        self.L0 = 0.8
        self.lam = 1e2
        self.sigma0 = 10  # neighbourhood radius.
        iter_count = 0

        while iter_count < iterations:
            if uniform == 1:
                i_data = np.random.choice(range(len(data)))  # returns a random number.
            elif uniform == 0:
                probability = dirichlet([1] * 5000)  # uses the dirichlet function to distribute probabilities.
                i_data = np.random.choice(range(len(data)), p=probability)

            bmu = self.find_bmu(data[i_data])
            self.update_som(bmu, data[i_data], iter_count)
            iter_count += 1

    def find_bmu(self, input_vec):
        """
            Find the BMU of a given input vector <x,y>.
            Finds the Euclidean distance between the input vector and the som neurons.
            input_vec: an input vector <x,y>.
            :returns The BMU.
        """
        list_bmu = []
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist = np.linalg.norm((input_vec - self.som[y, x]))
                list_bmu.append(((y, x), dist))
        list_bmu.sort(key=lambda x: x[1])
        return list_bmu[0][0]

    def update_som(self, bmu, input_vector, t):
        """
            Calls the update rule on each cell.
            finds the Euclidean distance between the bmu and the som neurons.
            bmu: (y,x) BMU coordinates.
            input_vector: current data vector.
            t: current time.
        """
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist_to_bmu = np.linalg.norm((np.array(bmu) - np.array((y, x))))
                self.update_cell((y, x), dist_to_bmu, input_vector, t)

    def update_cell(self, cell, dist_to_bmu, input_vector, t):
        """
            Computes the update rule on a cell.
            cell: (y,x) cell's coordinates.
            dist_to_bmu: L2 distance from cell to bmu.
            input_vector: current data vector.
            t: current time.
        """
        self.som[cell] += self.N(dist_to_bmu, t) * self.L(t) * (input_vector - self.som[cell])

    def L(self, t):
        """
            Learning rate formula.
            t: current time.
        """
        return self.L0 * np.exp(-t / self.lam)

    def N(self, dist_to_bmu, t):
        """
            Computes the neighbouring penalty.
            dist_to_bmu: L2 distance to bmu.
            t: current time.
        """
        curr_sigma = self.sigma(t)
        return np.exp(-(dist_to_bmu ** 2) / (2 * curr_sigma ** 2))

    def sigma(self, t):
        """
            Neighbouring radius formula.
            t: current time.
        """
        return self.sigma0 * np.exp(-t / self.lam)


def plot_2D_som(som, iterations, neurons, uniform, shape):
    """
    Plots the som on 2d graph and saves it.
    :param som: the som model after training .
    :param iterations: number of iterations that were preformed.
    :return:
    """
    fig = plt.figure()
    x = som.som[:, :, 0].flatten()
    y = som.som[:, :, 1].flatten()
    plt.scatter(x, y)
    plt.title("Number of iterations %d, Neurons %d, Shape: %s, Uniform %d" % (iterations, neurons, shape, uniform))
    plt.show()
    # fig.savefig('iterations %d neurons %d shape %s uniform %d .png' % (iterations, neurons, shape, uniform),
    #             dpi=fig.dpi)


def PointsInDonut(n=500):
    """
    calculates points on a "donut" shape i.e. {<x.y> | 1<= x^2 +y^2 <= 2}.
    :param n: number of data samples .
    :return:an array of points <x,y> inside the donut shape.
    """
    i = 0
    return_array = [[0 for c in range(2)] for r in range(n)]

    while i < n:
        x = random.uniform(-math.sqrt(2), math.sqrt(2))
        y = random.uniform(-math.sqrt(2), math.sqrt(2))
        if 1 <= math.pow(x, 2) + math.pow(y, 2) <= 2:
            return_array[i][0] = x
            return_array[i][1] = y
            i += 1
    return return_array


if __name__ == '__main__':
    # the data set is {(x,y) |  0 <= x <= 1, 0<=y<=1}
    square_data = np.random.rand(5000, 2)  # random samples from a uniform distribution over [0, 1).
    uniform = 1
    # 15 neurons, 50 iterations #
    number_of_flat_neurons = 15
    som_square = SOM(1, number_of_flat_neurons, 2)
    iterations = 50
    som_square.train(square_data, iterations, uniform)
    plot_2D_som(som_square, iterations, number_of_flat_neurons, uniform, "Square")
    #
    # 15 neurons, 250 iterations #
    number_of_flat_neurons = 15
    som_square = SOM(1, number_of_flat_neurons, 2)
    iterations = 250
    som_square.train(square_data, iterations, uniform)
    plot_2D_som(som_square, iterations, number_of_flat_neurons, uniform, "Square")

    # 200 neurons, 50 iterations #
    number_of_flat_neurons = 200
    som_square = SOM(1, number_of_flat_neurons, 2)
    iterations = 50
    som_square.train(square_data, iterations, uniform)
    plot_2D_som(som_square, iterations, number_of_flat_neurons, uniform, "Square")

    # 200 neurons, 300 iterations #
    number_of_flat_neurons = 200
    som_square = SOM(1, number_of_flat_neurons, 2)
    iterations = 300
    som_square.train(square_data, iterations, uniform)
    plot_2D_som(som_square, iterations, number_of_flat_neurons, uniform, "Square")

    # Non-uniform #
    square_data = np.random.rand(5000, 2)
    non_uniform = 0
    # 15 neurons, 50 iterations #
    number_of_flat_neurons2 = 15
    som_square = SOM(1, number_of_flat_neurons2, 2)
    iterations = 50
    som_square.train(square_data, iterations, non_uniform)
    plot_2D_som(som_square, iterations, number_of_flat_neurons2, non_uniform, "Square")

    # 15 neurons, 250 iterations #
    number_of_flat_neurons2 = 15
    som_square = SOM(1, number_of_flat_neurons2, 2)
    iterations = 250
    som_square.train(square_data, iterations, non_uniform)
    plot_2D_som(som_square, iterations, number_of_flat_neurons2, non_uniform, "Square")

    # 200 neurons, 50 iterations #
    number_of_flat_neurons2 = 200
    som_square = SOM(1, number_of_flat_neurons2, 2)
    iterations = 50
    som_square.train(square_data, iterations, non_uniform)
    plot_2D_som(som_square, iterations, number_of_flat_neurons2, non_uniform, "Square")

    # 200 neurons, 300 iterations #
    number_of_flat_neurons2 = 200
    som_square = SOM(1, number_of_flat_neurons2, 2)
    iterations = 300
    som_square.train(square_data, iterations, non_uniform)
    plot_2D_som(som_square, iterations, number_of_flat_neurons2, non_uniform, "Square")

    # #######################################
    # PART 2 #
    # {<x.y> | 1<= x^2 +y^2 <= 2}
    # shape = "donut"
    number_of_data_points = 5000
    donut_data = PointsInDonut(number_of_data_points)

    # 30 neurons, 50 iterations #
    number_of_flat_neurons = 15
    som_line = SOM(1, number_of_flat_neurons, 2)
    iterations = 50
    uniform = 1
    som_line.train(donut_data, iterations, uniform)

    plot_2D_som(som_line, iterations, number_of_flat_neurons, uniform, "donut")

    # 30 neurons, 250 iterations #
    number_of_flat_neurons = 20
    som_line = SOM(1, number_of_flat_neurons, 2)
    iterations = 250
    uniform = 1
    som_line.train(donut_data, iterations, uniform)

    plot_2D_som(som_line, iterations, number_of_flat_neurons, uniform, "donut")

    # 200 neurons, 250 iterations #
    number_of_flat_neurons = 200
    som_line = SOM(1, number_of_flat_neurons, 2)
    iterations = 250
    uniform = 1
    som_line.train(donut_data, iterations, uniform)

    plot_2D_som(som_line, iterations, number_of_flat_neurons, uniform, "donut")
