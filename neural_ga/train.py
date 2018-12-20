from neural_ga.genetic_algorithm import Individual
from neural_ga.utils import SharedNoiseTable, save_json
import gym
import numpy as np
from multiprocessing import Pool, Value
import multiprocessing as mp
from datetime import datetime
import copy

from logging import getLogger, StreamHandler, DEBUG, FileHandler
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

handler2 = FileHandler(filename="train.log")
handler2.setLevel(DEBUG)
logger.addHandler(handler2)

ENV_NAME = 'Frostbite-v0'
MAX_GENERATIONS = 50
POPULATION_SIZE = 1000
TRUNCATE_SIZE = 20
ELITE_CANDIDATES_SIZE = 10
MUTATION_POWER = 2E-3
ELITE_CANDIDATES_ROLLOUT_COUNT = 30

_env = gym.make(ENV_NAME)
STATE_SHAPE = (4, 84, 84)
N_A = _env.action_space.n


def init_processes(*initargs):
    global worker, env, count, avg_fitness
    noise_table, ENV_NAME, MUTATION_POWER, count, avg_fitness = initargs
    env = gym.make(ENV_NAME)
    STATE_SHAPE = (4, 84, 84)
    N_A = env.action_space.n
    worker = Individual(STATE_SHAPE, N_A, noise_table, sigma=MUTATION_POWER)
    print(mp.current_process().name, 'init')


def run_individual(individual, mutate=False):
    global worker, env, count, avg_fitness
    worker.load(individual)
    worker_name = mp.current_process().name
    if mutate:
        worker.mutate()
        fitness, steps = worker.rollout(env, max_steps=2000, episode=1, return_steps=True)
        avg_fitness.value = worker.fitness * 0.001 + avg_fitness.value * 0.999
    else:
        fitness, steps = worker.rollout(env, max_steps=3000, episode=ELITE_CANDIDATES_ROLLOUT_COUNT, return_steps=True)
    count.value += 1
    if count.value % 50 == 0 or (not mutate and count.value % 2 == 0):
        msg = '{}, count {:04d}, fitness: {:>.0f}, steps: {:>.0f}, avg_fitness: {:.1f}'.format(worker_name, count.value, worker.fitness, steps, avg_fitness.value)
        logger.info(msg)
    return worker.encode()

def run_parent(parent):
    return run_individual(parent, mutate=True)


def run_elite_candidate(elite_candidate):
    return run_individual(elite_candidate, mutate=False)

# if __name__ == '__main__':
noise_table = SharedNoiseTable()

elite = None

NUM_WORKER = mp.cpu_count()
count = Value('i', 0)
avg_fitness = Value('d', 0.0)
initargs = (noise_table, ENV_NAME, MUTATION_POWER, count, avg_fitness)
pool = Pool(processes=NUM_WORKER, initializer=init_processes, initargs=initargs)

elite_fitness_history = []

start = datetime.now()
for generation in range(MAX_GENERATIONS):
    if generation == 0:
        parents = [
            Individual(STATE_SHAPE, N_A, noise_table, sigma=MUTATION_POWER).encode() \
            for i in range(POPULATION_SIZE)
        ]

    children = []

    selected_parents_idx = np.random.randint(0, high=TRUNCATE_SIZE, size=POPULATION_SIZE)
    selected_parents = [copy.deepcopy(parents[idx]) for idx in selected_parents_idx]

    logger.info('Generation {}, Start population rollout.'.format(generation))
    children = pool.map(run_parent, selected_parents)
    logger.info('Generation {}, End population rollout.'.format(generation))
    count.value = 0

    children.sort(key=lambda x: x['fitness'], reverse=True)

    elite_candidates = []
    if generation == 0:
        elite_candidates = children[:ELITE_CANDIDATES_SIZE]
    else:
        elite_candidates = children[:ELITE_CANDIDATES_SIZE - 1]
        elite_candidates.append(elite)

    logger.info('Generation {}, Start elite selection.'.format(generation))
    elite_candidates = pool.map(run_elite_candidate, elite_candidates)
    count.value = 0

    elite_candidates.sort(key=lambda x: x['fitness'], reverse=True)
    elite = elite_candidates[0]

    for i in range(len(children)):
        if children[i]['genotypes'] == elite['genotypes']:
            children.pop(i)
            break
    parents = [elite] + children

    logger.info('\nelite fitness: {:.1f}'.format(elite['fitness']))
    logger.info(elite['genotypes'])
    logger.info('*** ' + str(datetime.now() - start)[:7] + ' ***\n')
    save_json(elite, '.cache/')
    elite_fitness_history.append(elite['fitness'])

pool.terminate()
pool.close()
