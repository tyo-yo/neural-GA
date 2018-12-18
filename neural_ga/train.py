from neural_ga.genetic_algorithm import Individual
from neural_ga.utils import SharedNoiseTable, save_json
import gym
import numpy as np
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

ENV_NAME = 'Frostbite-v0'
MAX_GENERATIONS = 5
POPULATION_SIZE = 1000
TRUNCATE_SIZE = 20
ELITE_CANDIDATES_SIZE = 10
MUTATION_POWER = 2E-3
ELITE_CANDIDATES_ROLLOUT_COUNT = 10

_env = gym.make(ENV_NAME)
STATE_SHAPE = (4, 84, 84)
N_A = _env.action_space.n

noise_table = SharedNoiseTable()

env = gym.make(ENV_NAME)
individual = Individual(STATE_SHAPE, N_A, noise_table, sigma=MUTATION_POWER)
elite = None

for generation in range(MAX_GENERATIONS):
    if generation == 0:
        parents = [
            Individual(STATE_SHAPE, N_A, noise_table, sigma=MUTATION_POWER).encode() \
            for i in range(POPULATION_SIZE)
        ]

    children = []
    # This can be Parallelized
    for i in range(POPULATION_SIZE):
        k = np.random.randint(0, TRUNCATE_SIZE)
        individual.load(parents[k])
        individual.mutate()
        individual.rollout(env)
        children.append(individual.encode())
        if i % 10 == 0:
            logger.info('individual {:d}: {:.1f}'.format(i, individual.fitness))

    children.sort(key=lambda x: x['fitness'], reverse=True)

    elite_candidates = []
    if generation == 0:
        elite_candidates = children[:ELITE_CANDIDATES_SIZE]
    else:
        elite_candidates = children[:ELITE_CANDIDATES_SIZE - 1]
        elite_candidates.append(elite)

    for i, elite_candidate in enumerate(elite_candidates):
        individual.load(elite_candidate)
        individual.rollout(env, episode=ELITE_CANDIDATES_ROLLOUT_COUNT)
        elite_candidates[i] = individual.encode()
        logger.info('elite_candidate {:d}: {:.1f}'.format(i, individual.fitness))

    elite_candidates.sort(key=lambda x: x['fitness'], reverse=True)
    elite = elite_candidates[0]

    for i in range(len(children)):
        if children[i]['genotypes'] == elite['genotypes']:
            children.pop(i)
            break
    parents = [elite] + children

    logger.info('\nelite fitness: {:.1f}\n'.format(elite['fitness']))
    logger.info(elite['genotypes'])
    save_json(elite, './cache')
