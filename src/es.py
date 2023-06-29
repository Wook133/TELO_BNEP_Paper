import numpy as np
import scipy
import random


def rankedFitness(R):
    """ produce a linear ranking of the fitnesses in R.
    (The highest rank is the best fitness)"""
    res = np.zeros_like(R)
    l = list(zip(R, list(range(len(R)))))
    l.sort()
    for i, (_, j) in enumerate(l):
        res[j] = i
    return res


def HansenRanking(R):
    """ Ranking, as used in CMA-ES """
    ranks = rankedFitness(R)
    return np.array([max(0., x) for x in np.log(len(R) / 2. + 1.0) - np.log(len(R) - np.array(ranks))])


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        del step, theta
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError

    def get_parameters(self):
        return self.pi, self.dim, self.epsilon, self.t


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def set_m(self, m):
        self.m = m

    def set_v(self, v):
        self.v = v

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        del a
        return step


class CMAES:
    '''CMA-ES wrapper.
        Added Rank Based Fitnss
    '''


    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.10,  # initial standard deviation
                 popsize=0,  # population size
                 weight_decay=0.01,# weight decay coefficient
                 rank_fitness=True,  # use rank rather than fitness numbers
                 mu_init=0.0,
                 lower_bound=-1.25,
                 upper_bound=1.25,
                 reset=False,
                 ftarget=-np.inf):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        self.mu_init = mu_init
        self.solutions = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        print("Bounds = [", lower_bound, "; ", upper_bound, "]")
        self.ftarget = ftarget
        import cma
        if popsize == 0:
            if reset:
                self.es = cma.CMAEvolutionStrategy(self.mu_init, self.sigma_init,
                                                   {'bounds': [self.lower_bound, self.upper_bound],
                                                    'ftarget': self.ftarget,})
            else:
                self.es = cma.CMAEvolutionStrategy(self.num_params * [self.mu_init], self.sigma_init,
                                                   {'bounds': [self.lower_bound, self.upper_bound],
                                                    'ftarget': self.ftarget, })
        else:
            if reset:
                self.es = cma.CMAEvolutionStrategy(self.mu_init,
                                                   self.sigma_init,
                                                   {'popsize': self.popsize,
                                                    'bounds': [self.lower_bound, self.upper_bound],
                                                    'ftarget': self.ftarget, })
            else:
                self.es = cma.CMAEvolutionStrategy(self.num_params * [self.mu_init],
                                                   self.sigma_init,
                                                   {'popsize': self.popsize,
                                                    'bounds': [self.lower_bound, self.upper_bound],
                                                    'ftarget': self.ftarget, })


    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self, pop_size=None, xmean=None):
        '''returns a list of parameters'''
        if pop_size is not None:
            if xmean is None:
                self.solutions = np.array(self.es.ask(number=pop_size))
            else:
                self.solutions = np.array(self.es.ask(number=pop_size, xmean=xmean))
        else:
            if xmean is None:
                self.solutions = np.array(self.es.ask())
            else:
                self.solutions = np.array(self.es.ask(xmean=xmean))
        return self.solutions

    def seeder_ask(self, pop_size=None, seed=None, xmean=None):
        print("Seeder Ask")
        '''returns a list of parameters'''
        if pop_size is not None:
            if xmean is None:
                self.solutions = np.array(self.es.ask(number=pop_size))
            else:
                self.solutions = np.array(self.es.ask(number=pop_size, xmean=xmean))
        else:
            if xmean is None:
                self.solutions = np.array(self.es.ask())
            else:
                self.solutions = np.array(self.es.ask(xmean=xmean))
        seeding = []
        if seed is not None:
            for i in range(pop_size):
                seeding.append(int(random.random()))
        else:
            random.seed(seed)
            for i in range(pop_size):
                seeding.append(int(random.random()))
        return self.solutions, seeding

    def tell(self, reward_table_result, solutions=None):
        if self.rank_fitness:
            reward_table_result = compute_centered_ranks(np.array(reward_table_result))
        else:
            reward_table_result = np.array(reward_table_result)

        reward_table = -1.0*reward_table_result
        if solutions is None:
            if self.weight_decay > 0:
                l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
                reward_table += l2_decay
                self.es.tell(self.solutions, (reward_table).tolist())  # convert minimizer to maximizer.
        else:
            self.solutions = solutions
            if self.weight_decay > 0:
                l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
                reward_table += l2_decay
                self.es.tell(self.solutions, reward_table)  # convert minimizer to maximizer.


    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def current_params(self):
        return self.es.result[5], self.es.result[6]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])

    def get_mu(self):
        return self.es.mean

    def get_sigma(self):
        return self.es.sigma


class OpenES:
    ''' Basic Version of OpenAI Evolution Strategies.'''

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.1,  # initial standard deviation
                 sigma_decay=0.999,  # anneal standard deviation
                 sigma_limit=0.01,  # stop annealing if less than this
                 learning_rate=0.01,  # learning rate for standard deviation
                 learning_rate_decay=0.9999,  # annealing the learning rate
                 learning_rate_limit=0.001,  # stop annealing learning rate
                 popsize=256,  # population size
                 antithetic=False,  # whether to use antithetic sampling
                 weight_decay=0.01,  # weight decay coefficient
                 mu=0,
                 rank_fitness=True,  # use rank rather than fitness numbers
                 forget_best=True):  # forget historical best

        self.num_params = num_params
        self.sigma_decay = sigma_decay
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.popsize % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.popsize / 2)

        self.reward = np.zeros(self.popsize)
        if mu == 0:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.ones(self.num_params)*mu
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_interation = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness
        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma * sigma))

    def ask(self, pop_size=None):
        '''returns a list of parameters'''
        # antithetic sampling
        if pop_size is not None:
            self.popsize = pop_size
            self.half_popsize = int(pop_size / 2)
        if self.antithetic:
            self.epsilon_half = np.random.randn(self.half_popsize, self.num_params)
            self.epsilon = np.concatenate([self.epsilon_half, - self.epsilon_half])
        else:
            self.epsilon = np.random.randn(self.popsize, self.num_params)
        self.solutions = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma
        seeding = []
        for i in range(self.popsize):
            seeding.append(int(random.random()))
        return self.solutions, seeding



    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert (len(reward_table_result) == self.popsize), "Inconsistent reward_table size reported."
        reward = np.array(reward_table_result)
        if self.rank_fitness:
            reward = compute_centered_ranks(reward)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward += l2_decay
        idx = np.argsort(reward)[::-1]
        best_reward = reward[idx[0]]
        best_mu = self.solutions[idx[0]]
        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu
        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward
        # main bit:
        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = (1. / (self.popsize * self.sigma)) * np.dot(self.epsilon.T, normalized_reward)
        #normalized_reward = np.divide(np.subtract(reward, np.mean(reward)), np.std(reward))
        #change_mu = np.multiply(np.divide(1.0, np.multiply(self.popsize, self.sigma)),
        #                        np.dot(self.epsilon.T, normalized_reward))
        # self.mu += self.learning_rate * change_mu
        self.optimizer.stepsize = self.learning_rate
        update_ratio = self.optimizer.update(-change_mu)
        # adjust sigma according to the adaptive sigma calculation
        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay
        if (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay
        del update_ratio, change_mu, normalized_reward, best_mu, best_reward, idx, reward

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)

    def current_params(self):
        return self.mu, self.sigma

    def get_parameters(self):
        # self.epsilon_half,
        return self.epsilon, self.solutions, self.curr_best_reward, self.curr_best_mu, self.first_interation, self.best_reward, self.best_mu, self.optimizer, self.mu, self.num_params, self.sigma_init, self.sigma, self.sigma_decay, self.sigma_limit, self.learning_rate, self.learning_rate_decay, self.learning_rate_limit, self.popsize, self.antithetic

    # epsilon_half
    def set_parameters(self, epsilon, solutions, curr_best_reward, curr_best_mu, first_interation, best_reward, best_mu, optimizer, mu, num_params, sigma_init, sigma, sigma_decay, sigma_limit, learning_rate, learning_rate_decay, learning_rate_limit, popsize, antithetic):
        # self.epsilon_half = epsilon_half
        self.epsilon = epsilon
        self.solutions = solutions
        self.curr_best_reward = curr_best_reward
        self.curr_best_mu = curr_best_mu
        self.first_interation = first_interation
        self.best_reward = best_reward
        self.best_mu = best_mu
        self.optimizer = Adam(optimizer.pi, optimizer.stepsize, optimizer.beta1, optimizer.beta2)
        self.optimizer.set_m(optimizer.m)
        self.optimizer.set_v(optimizer.v)
        self.mu = mu
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.antithetic = antithetic


class sepCEMA:

    """
    Cross-entropy methods.
    Diagonal Only?
    """

    def __init__(self,
                 num_params,
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 parents=None,
                 elitism=False,
                 rank_fitness=False,
                 antithetic=False):

        # misc
        self.num_params = num_params

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = -np.inf

        # sampling stuff
        self.pop_size = pop_size
        self.antithetic = antithetic
        self.rank_fitness = rank_fitness
        if self.antithetic:
            assert (self.pop_size % 2 == 0), "Population size must be even"
        if parents is None or parents <= 0:
            self.parents = pop_size // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def ask(self, pop_size):
        """
        Returns a list of candidates parameters
        """
        if self.antithetic and not pop_size % 2:
            epsilon_half = np.random.randn(pop_size // 2, self.num_params)
            epsilon = np.concatenate([epsilon_half, - epsilon_half])
        else:
            epsilon = np.random.randn(pop_size, self.num_params)
        inds = self.mu + epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        return inds

    def tell(self, solutions, scores):
        """
        Updates the distribution
        """

        scores = np.array(scores)
        if self.rank_fitness:
            scores = compute_centered_ranks(scores)
        scores *= -1
        idx_sorted = np.argsort(scores)

        # new and old mean
        old_mu = self.mu
        self.mu = self.weights @ solutions[idx_sorted[:self.parents]]

        # sigma adaptation
        if scores[idx_sorted[0]] > 0.95 * self.elite_score:
            self.sigma *= 0.95
        else:
            self.sigma *= 1.05
        self.elite = solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        z = (solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = self.weights @ (z * z)
        self.cov = self.sigma * self.cov / np.linalg.norm(self.cov)
        print(self.cov)
        print(self.sigma)

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)


class SNES():
    #To Do
    #More Robust Settings
    #E.g. in line with some of the es.py settings
    #If a sigma_i == 0. then expand search out?
    def __init__(self, numParameters, initial_mu=None, initial_variance=None, population_size=None, covLearningRate=None, minimize=False, record_stats=False, antithetic=False):
        # parameters, which can be set but have a good (adapted) default value
        self.numParameters = numParameters
        if initial_mu is None:
            # self._center = scipy.randn(self.numParameters)
            self._center = np.zeros(self.numParameters)
        else:
            assert len(initial_mu) == self.numParameters, "number of parameters does not match"
            self._center = initial_mu
        if initial_variance is None:
            self.initVariance = 1.0
            self._sigmas = np.ones(self.numParameters) * self.initVariance
        else:
            self.initVariance = initial_variance
            self._sigmas = np.ones(self.numParameters) * initial_variance

        self.centerLearningRate = 1.0
        if covLearningRate is not None:
            self.covLearningRate = covLearningRate
        else:
            self.covLearningRate = self._initLearningRate()
        if population_size is not None:
            self.batchSize = population_size
        else:
            self.batchSize = self._initBatchSize()
        self.uniformBaseline = True
        self.shapingFunction = HansenRanking
        self.initVariance = 1.

        # fixed settings
        self._allEvaluated = []
        self._allEvaluations = []
        # for very long runs, we don't want to run out of memory
        self.clearStorage = False

        # minimal setting where to abort the search
        self.varianceCutoff = 1e-20

        self.minimize = minimize
        self.reward = np.zeros(self.batchSize)
        self.best_mu = np.zeros(self.numParameters)
        self.best_sigma = self._sigmas
        self.best_y = float('-inf')
        self.cur_best_y = float('-inf')
        self.best_x = np.zeros(self.numParameters)
        self.cur_best_x = np.zeros(self.numParameters)
        self.popsize = self.batchSize
        self._talliedVariance = self._sigmas
        self.record_stats = record_stats
        self.antithetic = antithetic
        if self.antithetic:
            assert (self.batchSize % 2 == 0), "Population size must be even"

    def get_batch_Size(self):
        return self.batchSize

    def ask(self):
        return [self._sample2base(self._produceSample()) for _ in range(self.batchSize)]

    def _growingCov(self):
        self._talliedVariance = self._talliedVariance * 1.5
        self._sigmas = np.ones(self.numParameters) * self._talliedVariance

    def _reinitialization(self):
        #Assume that all x's are the same length
        self._talliedVariance = self._talliedVariance * 1.0001
        lengthBestMu = len(self.best_mu)
        lengthTempItem = len(self.best_mu[0].id)
        listAllX = np.zeros(lengthBestMu*lengthTempItem)

        for i in range(lengthBestMu):
            item = self.best_mu[i]
            x = item.id
            for j in x:
                listAllX[i] = j
        new_mu = listAllX.reshape(lengthBestMu, lengthTempItem)

        self._center = np.mean(new_mu, axis=0)
        print("New Mu = ", self._center)
        self._sigmas = np.var(new_mu, axis=0) + self._talliedVariance
        print("New _sigmas = ", self._sigmas)

    def _do_tell(self, x, y):
        self._allEvaluated.append(x.copy())
        self._allEvaluations.append(y)
        return y

    def tell(self, x, y):
        if self.record_stats:
            index = np.where(y == np.amax(y))[0][0]
            cur_max = y[index]
            if cur_max > self.best_y:
                #print("Hello there")
                self.best_y = cur_max
                self.best_x = x[index]
                self.best_mu = self._center
                self.best_sigma = self._sigmas

        #Record All => Causes Memory Requirements to Continually Grow
        #self._do_tell(x, y)
        samples = list(map(self._base2sample, x))

        #print("Y = ", y)
        # compute utilities
        utilities = self.shapingFunction(y)
        #print("Y utiled = ", utilities)
        utilities /= sum(utilities)  # make the utilities sum to 1
        #print("utlities = ", utilities)
        if self.uniformBaseline:
            utilities -= 1. / self.batchSize

        #update center
        dCenter = np.dot(utilities, samples)
        self._center += self.centerLearningRate * self._sigmas * dCenter

        # update variances
        covGradient = np.dot(utilities, [s ** 2 - 1 for s in samples])
        dA = 0.5 * self.covLearningRate * covGradient
        self._sigmas = self._sigmas * np.exp(dA)
        self._sigmas = np.nan_to_num(self._sigmas)
        all_zero = np.all(self._sigmas == 0.)
        if all_zero:
            #print("Restarting Sigma")
            if self.record_stats:
                self._reinitialization()
            else:
                self._growingCov()
        del samples, utilities, dCenter, covGradient, dA

    def _produceSample(self):
        return scipy.randn(self.numParameters)

    def _sample2base(self, sample):
        """ How does a sample look in the outside (base problem) coordinate system? """
        #return numpy.add(numpy.multiply(self._sigmas, sample), self._center)
        return self._sigmas * sample + self._center

    def _base2sample(self, e):
        """ How does the point look in the present one reference coordinates? """
        return (e - self._center) / self._sigmas

    def _initLearningRate(self):
        """ Careful, robust default value. """
        return 0.6 * (3 + np.log(self.numParameters)) / 3 / np.sqrt(self.numParameters)

    def _initBatchSize(self):
        """ as in CMA-ES """
        return 4 + int(np.floor(3 * np.log(self.numParameters)))

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_x, self.best_y, self.best_mu, self.best_sigma)

#NEAT
#BNE

#CEM-RL
#BNE-RL
