import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from neural_ga.utils import SharedNoiseTable, rollout
import copy
import logging
logger = logging.getLogger(__name__)


class Individual(nn.Module):
    def __init__(self, s_shape, a_dim, noise_table,
                 initial_genotype=None, sigma=2e-3):
        super(Individual, self).__init__()
        self.s_shape = s_shape
        self.a_dim = a_dim
        self.noise_table = noise_table
        self.sigma = sigma

        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, a_dim)

        # self.distribution = torch.distributions.Categorical
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]

        self.n_param = 0
        self.param_shapes, self.n_params = {}, {}
        for param, key in zip(self.parameters(), self.state_dict().keys()):
            param.requires_grad = False
            self.param_shapes[key] = param.shape
            n = 1
            for s in param.shape:
                n *= s
            self.n_param += n
            self.n_params[key] = n

        self.random_stream = np.random.RandomState()
        self.genotypes = []
        self.initial_genotype = initial_genotype
        self.xavier_initialization(initial_genotype=self.initial_genotype)
        self.fitness = None

    def __repr__(self):
        s = super(Individual, self).__repr__()
        return s + '\n' + json.dumps(self.encode()).replace(', ', ', \n') + '\n'

    def xavier_initialization(self, initial_genotype=None):
        if not initial_genotype:
            self.initial_genotype = self.noise_table.sample_index(
                self.random_stream, self.n_param)
        else:
            self.initial_genotype = initial_genotype
        noise_flattened = self.noise_table.get(self.initial_genotype,
                                               self.n_param)

        input_sizes = {
            'conv1.weight': 4 * 84 * 84,
            'conv2.weight': 16 * 20 * 20,
            'fc1.weight': 32 * 9 * 9,
            'fc2.weight': 256
        }
        idx = 0
        for param, key in zip(self.parameters(), self.state_dict().keys()):
            n = self.n_params[key]
            if 'bias' in key:
                nn.init.constant_(param, 0.0)
            else:
                w = noise_flattened[idx: idx + n]
                w = torch.tensor(w)
                xavier_coef = 1 / input_sizes[key] ** 0.5
                param.data = xavier_coef * w.reshape(self.param_shapes[key])
            idx += n

    def reset(self, sigma=None, initial_genotype=None):
        self.genotypes = []
        if sigma:
            self.sigma = sigma
        if not initial_genotype:
            self.initial_genotype = initial_genotype
            self.xavier_initialization(
                initial_genotype=self.initial_genotype)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        logits = self.fc2(x)
        return logits

    def choose_action(self, s):
        self.eval()
        logits = self.forward(s)
        a = logits.argmax(dim=1).numpy()[0]
        return a
        # prob = F.softmax(logits, dim=1).data
        #
        # m = self.distribution(prob)
        # return m.sample().numpy()[0]

    def update(self, genotype):
        self.genotypes.append(genotype)
        idx = 0
        noise_flattened = self.noise_table.get(genotype, self.n_param)

        for param, key in zip(self.parameters(), self.state_dict().keys()):
            n = self.n_params[key]
            w = noise_flattened[idx: idx + n]
            w = torch.tensor(w)
            param.data += self.sigma * w.reshape(self.param_shapes[key])
            idx += n

    def mutate(self):
        genotype = self.noise_table.sample_index(self.random_stream,
                                                 self.n_param)
        self.update(genotype)
        return genotype

    def set_parameters(self, genotypes, initial_genotype):
        self.xavier_initialization(initial_genotype=initial_genotype)
        for genotype in genotypes:
            self.update(genotype)
        self.initial_genotype = initial_genotype
        self.genotypes = genotypes

    def encode(self):
        return copy.deepcopy(dict(
            genotypes=self.genotypes,
            sigma=self.sigma,
            initial_genotype=self.initial_genotype,
            fitness=self.fitness,
            noise_table_seed=self.noise_table.seed,
            noise_table_size=self.noise_table.size,
        ))

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.encode(), f)

    def load(self, code, noise_table=None):
        if type(code) == str:
            with open(code, 'r') as f:
                code = json.load(f)
        assert type(code) == dict
        code = copy.deepcopy(code)

        self.sigma = code['sigma']

        if self.noise_table and noise_table:
            assert noise_table.seed == code['noise_table_seed']
            assert noise_table.size == code['noise_table_size']
            self.noise_table = noise_table
        elif not self.noise_table and noise_table:
            self.noise_table = SharedNoiseTable(
                seed=code['noise_table_seed'],
                size=code['noise_table_size'])
        elif not self.noise_table and not noise_table:
            logger.error('Noise table is not specified.')

        self.set_parameters(code['genotypes'], code['initial_genotype'])
        return self

    def rollout(self, env, max_steps=1000, rendering=False, episode=1, return_steps=False):
        if not return_steps:
            self.fitness = rollout(self, env, max_steps=max_steps,
                                   rendering=rendering, return_steps=return_steps,
                                   episode=episode)
            return self.fitness
        else:
            self.fitness, steps = rollout(self, env, max_steps=max_steps,
                                   rendering=rendering, return_steps=return_steps,
                                   episode=episode)
            return self.fitness, steps
