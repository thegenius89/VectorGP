from numpy import random, sin, cos, tan, log, exp, average, unique
from time import time
from os import getpid
from multiprocessing import Pool

from Population import Population


# flake8 --max-line-length=100
# python3 -m cProfile -o out Main.py
# python3 -m pstats out
# python3 Main.py


class Algorithm:

    def __init__(self, config) -> None:
        for key, value in config.items():
            setattr(self, key, value)
        self.init_stats(time())

    def init_stats(self, start):
        self.stats = {'best': 1.0, 'repr': '', 'start': start, 'used': 0, 'changed': 0}

    def update_best(self, pop, errors, gen, show) -> bool:
        mini = errors.argmin()
        if errors[mini] < self.stats['best']:
            self.stats['best'] = errors[mini]
            self.stats['repr'] = pop.to_str_at(mini)
            self.stats['changed'] = gen
        best = self.stats['best']
        if (gen % (self.gens // 10) == 0 or best <= self.epsilon) and show:
            print('Gen        :', gen)
            print('Mean       :', round(average(errors[errors != 1]), 5))
            print('Min        :', round(min(errors), 10))
            print('Diversity  :', len(unique(errors)))
            print('Best expr  :', pop.to_str_at(mini))
            print()
        return best <= self.epsilon

    def run(self, show) -> None:
        random.seed(getpid())
        pop = Population(self.max_depth, self.pop_size, self.allow_leafs, self.train)
        for gen in range(self.gens):
            results = pop.eval_pop()
            errors = pop.fitness(results, self.target, self.panelty, self.pan_rate)
            if self.update_best(pop, errors, gen, show) or gen == self.gens - 1:
                break
            if gen - self.stats['changed'] > 25:
                pop = Population(self.max_depth, self.pop_size, self.allow_leafs, self.train)
                self.init_stats(self.stats['start'])
            pop.selection(errors, self.pressure, self.elitism)
            pop.mutate_trees(self.mut_rate)
            pop.crossover(self.cx_rate)
        self.stats['used'] = time() - self.stats['start']
        error, model, used = self.stats['best'], self.stats['repr'], self.stats['used']
        divers = len(unique(errors))
        print('Error: {:.4f}, Used: {:.4f}, Uniques: {}, Gen: {}, Model: {}'.format(
            round(error, 5), round(used, 3), divers, gen, model[0:50]))

def worker_run(config):
    algo = Algorithm(config)
    algo.run(True if config['threads'] <= 1 else False)


def main() -> None:
    print('Test Population')
    x = random.rand(50, 2) * 4 - 2
    y = sin(x[:, 0] * x[:, 0]) * cos(x[:, 1] + 1)
    # y = x[:, 0] ** 4 + x[:, 1] ** 2 + x[:, 2] - x[:, 3] + x[:, 4] * 0.5
    # y = (1 - x[:, 0]) ** 2 + (x[:, 1] - x[:, 0] ** 2) ** 2
    # x = random.rand(50, 5)
    # y = (tan(x[:, 0]) / exp(x[:, 1])) * (log(x[:, 2]) - tan(x[:, 3]))
    config = {
        'max_depth': 4,
        'pop_size': 5000,
        'gens': 500,
        'mut_rate': 0.05,
        'cx_rate': 0.05,
        'pressure': 1.00,
        'epsilon': 1e-9,
        'panelty': True,
        'pan_rate': 0.7,
        'allow_leafs': False,
        'elitism': True,
        'threads': 12,
        'train': x.T,
        'target': y,
    }
    if config['threads'] <= 1:
        worker_run(config)
        return
    with Pool(config['threads']) as pool:
        pool.map(worker_run, [config] * config['threads'])


if __name__ == "__main__":
    main()
