
# GNU General Public License software (C) Michael Kochen 2017.
# Based on the algorithm published by Sivia and Skilling 2006,
# and translated to Python by Issac Trotts in 2007.
# http://www.inference.org.uk/bayesys/

# A nested sampling algorithm to compute the evidence value
# for high dimensional PySB models.

# Two point selection methods are available
#   - an MCMC algorithm based on the original algorithm
#   - a new kernel density estimate method

from pysb.integrate import Solver
import numpy as np
from copy import deepcopy
import random
from math import *
import csv

class NS:

    def __init__(self, model, likelihood_function, data, alg='NS'):
        self.model = model
        self.likelihood_function = likelihood_function
        self.data = data
        self.model_solver = None
        self.prior_1kf = [-4, 0]
        self.prior_2kf = [-8, -4]
        self.prior_1kr = [-4, 0]
        self.prior_1kc = [-1, 3]
        self.iterations = 10000
        self.scalar = 10.0
        self.scalar_reduction = 0
        self.scalar_limit = .0001
        self.evidence = -1e300
        self.information = 0.0
        self.iteration = 0
        self.useless = 10
        self.N = 1000
        self.time = []
        self.working_set = []
        self.params = []
        self.width_LH = []
        self.processed_data = self.importData()
        self._initiate_log()
        if alg == 'NS':
            self._nested_sampling_KDE()
            self._output()
        if alg == 'MCMC':
            self._nested_sampling_MCMC()
            self._output()


    def _output(self):

        summary_object = self.model.name
        summary = open(summary_object, 'w')
        summary.write(summary_object)
        summary.write('parameters: ' + str(len(self.params)) + '\n')
        summary.write('iteration: ' + str(self.iteration) + '\n')
        summary.write('scalar: ' + str(self.scalar) + '\n')
        summary.write('scalar reduction: ' + str(self.scalar_reduction) + '\n')
        summary.write('evidence: ' + str(self.evidence) + ' +/- ' + str(sqrt(self.information / self.N)) + '\n')
        summary.write('information: ' + str(self.information) + '\n')
        summary.write('stop criteria: ' + self.stop + '\n\n')
        summary.close()

        # print
        # print self.model.name
        # print 'parameters: ' + str(len(self.params))
        # print 'iteration: ' + str(self.iteration)
        # print 'scalar: ' + str(self.scalar)
        # print 'scalar reduction: ' + str(self.scalar_reduction)
        # print 'evidence: ' + str(self.evidence) + ' +/- ' + str(sqrt(self.information / self.N))
        # print 'information: ' + str(self.information)
        # print 'stop criteria: ' + self.stop + '\n\n'

    def importData(self):

        # first column should be time
        # first row should be observable labels
        # this could change depending on the likelihood function

        data_object = []
        with open(self.data) as data_file:
            reader = csv.reader(data_file)
            line = list(reader)
            for each in line:
                data_object.append(each)

        for i, each in enumerate(data_object):
            if i > 0:
                for j, item in enumerate(each):
                    data_object[i][j] = float(data_object[i][j])

        return data_object

    def _initiate_log(self):

        # retrieve time points from data
        for each in self.processed_data[1:]:
            self.time.append(float(each[0]))

        # set parameter values or prior ranges (log-space)
        for each in self.model.parameters:
            if each.name[-2:] == '_0':
                self.params.append(each.value)
            if each.name[-3:] == '1kf':
                self.params.append(self.prior_1kf)
            if each.name[-3:] == '2kf':
                self.params.append(self.prior_2kf)
            if each.name[-3:] == '1kr':
                self.params.append(self.prior_1kr)
            if each.name[-3:] == '1kc':
                self.params.append(self.prior_1kc)

        # create solver object
        self.model_solver = Solver(self.model, self.time, integrator='lsoda', integrator_options={'atol': 1e-12, 'rtol': 1e-12, 'mxstep': 20000}) # , integrator_options={'atol': 1e-12, 'rtol': 1e-12, 'mxstep': 20000}

        # construct the working population of N parameter sets
        k=0
        while k < self.N:

            # randomly choose points from parameter space
            point = []
            for each in self.params:
                if isinstance(each, list):
                    point.append(10**(np.random.uniform(each[0], each[1])))
                else:
                    point.append(each)
            likelihood = self._compute_likelihood(point)
            if not isnan(likelihood):
                self.working_set.append([likelihood, point])
                k += 1

        self.working_set.sort(reverse = True)

    def _compute_likelihood(self, point):

        # simulate a point
        self.model_solver.run(point)

        # construct simulation trajectories
        sim_trajectories = [['time']]
        for each in self.model.observables:
            sim_trajectories[0].append(each.name)

        for i,each in enumerate(self.model_solver.yobs):
            sim_trajectories.append([self.time[i]] + list(each))

        # calculate the cost
        cost = self.likelihood_function(self.processed_data, sim_trajectories)
        if isinstance(cost, float):
            return cost
        else:
            return False # doesn't seem to do anything

    def _nested_sampling_KDE(self):

        useless_samples = 0
        index = 1

        while index <= self.iterations and self.scalar > self.scalar_limit:

            self.iteration = index

            # check number of non-viable samples taken from prior
            # when too many non-viable samples are taken from prior
            # shrink adaptive scalar which shrinks the kernel function

            if useless_samples == self.useless:
                self.scalar *= 0.9
                useless_samples = 0
                self.scalar_reduction += 1

            # sample from the prior using KDE_log
            test_point = self._KDE_sample_log()

            # calculate objective
            test_point_objective = self._compute_likelihood(test_point)

            # check if sample is within cost bound
            if not isnan(test_point_objective):
                if test_point_objective > self.working_set[-1][0]:

                    # update evidence and working set
                    LH_addition = self.working_set.pop()[0]
                    width = log(exp(-((float(index) - 1) / self.N)) - exp(-(float(index) / self.N)))
                    self.width_LH.append([width, LH_addition])
                    old_evi = float(self.evidence)
                    self.evidence = np.logaddexp(float(self.evidence), (LH_addition + width))
                    self.information = exp((LH_addition + width) - self.evidence) * LH_addition + \
                            exp(old_evi - self.evidence) * (self.information + old_evi) - self.evidence
                    self.working_set.append([test_point_objective, list(test_point)])
                    self.working_set.sort(reverse=True)
                    useless_samples = 0

                    index += 1
                else:
                    useless_samples += 1
            else:
                useless_samples += 1

        if index < self.iterations:
            self.stop = 'scalar_limit'
        else:
            self.stop = 'iterations'

        # add the likelihood from the working set
        for each in self.working_set:
            self.width_LH.append([log(exp(-(float(index) / self.N))) - log(self.N), each[0]])
            increment = (each[0] - log(self.N) + log(exp(-(float(index) / self.N))))
            self.evidence = np.logaddexp(float(self.evidence), increment)

    def _KDE_sample_log(self):

        # select data point
        data_point = np.random.randint(0, len(self.working_set))
        coordinates = self.working_set[data_point][1]

        # select parameter values individually from normal with respect to prior boundary
        new_point = []
        for i, each in enumerate(coordinates):
            if isinstance(self.params[i], float):
                new_point.append(self.params[i])
            else:
                accept = False
                log_coord = None

                while not accept:
                    log_coord = np.random.normal(log10(each), self.scalar)
                    if self.params[i][0] <= log_coord <= self.params[i][1]:
                        accept = True
                new_point.append(10**log_coord)

        return new_point

    def _nested_sampling_MCMC(self):

        index = 1
        while index < self.iterations and exp(-((float(index) - 1) / self.N)) - exp(-(float(index) / self.N)) > 5e-324:

            # update evidence and working set
            LH_addition = self.working_set[-1][0]

            width = log(exp(-((float(index) - 1) / self.N)) - exp(-(float(index) / self.N)))
            self.width_LH.append([width, LH_addition])
            self.evidence = np.logaddexp(float(self.evidence), (LH_addition + width))
            self._MCMC_sample()
            index += 1

        # add the likelihood from the working set
        for each in self.working_set:
            self.width_LH.append([log(exp(-(float(index) / self.N))) - log(self.N), each[0]])
            increment = (each[0] - log(self.N) + log(exp(-(float(index) / self.N))))
            self.evidence = np.logaddexp(float(self.evidence), increment)

    def _MCMC_sample(self):

        step = 0.4
        accept = 0
        reject = 0

        coord = deepcopy(self.working_set[-1][1])
        coord_LH = deepcopy(self.working_set[-1][0])

        j = 0
        while j < 20:

            new_coords = []
            for i,each in enumerate(self.model.parameters):
                if each.name[-2:] == '_0':
                    new_coords.append(coord[i])
                else:
                    prior = None
                    if each.name[-3:] == '1kf':
                        prior = self.prior_1kf
                    if each.name[-3:] == '2kf':
                        prior = self.prior_2kf
                    if each.name[-3:] == '1kr':
                        prior = self.prior_1kr
                    if each.name[-3:] == '1kc':
                        prior = self.prior_1kc

                    log_new_coord = None
                    while True:
                        log_new_coord = log10(coord[i]) + step * (2.*random.random() -1.)
                        if prior[0] < log_new_coord < prior[1]:
                            break
                    new_coords.append(10**log_new_coord)

            test_LH = self._compute_likelihood(new_coords)
            if test_LH:
                if test_LH > coord_LH:
                    coord = new_coords
                    coord_LH = test_LH
                    accept+=1
                else:
                    reject+=1
                if accept > reject:
                    step *= exp(1.0/accept)
                if accept < reject:
                    step /= exp(1.0/reject)
                j += 1

        self.working_set[-1] = [coord_LH, coord]
        self.working_set.sort(reverse=True)
