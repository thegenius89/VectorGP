from numpy import ndarray, where, mean, empty_like, copy, concatenate
from numpy import setdiff1d, stack, unique, arange, argsort
from numpy.random import rand, randint, choice


from SearchSpace import SearchSpace


class Population:

    def __init__(self, max_depth, pop_size, allow_leafs, x_val) -> None:
        self.space = SearchSpace(x_val)
        self.max_depth = max_depth
        self.pop_size = pop_size
        self.allow_leafs = allow_leafs
        self.binary_op_map = self.space.binary_op_map
        self.unary_op_map = self.space.unary_op_map
        self.var_map = self.space.var_map
        self.num_funcs = len(self.unary_op_map) + len(self.binary_op_map)
        self.bord = self.num_funcs + len(self.var_map)
        self.all_vars = stack([self.var_map[i][0] for i in sorted(self.var_map.keys())], axis=0)
        self.pop = self.init_pop(self.max_depth)

    # create a population of trees in form of {op: ndarray, left: dict, right: dict}
    def init_pop(self, max_depth) -> dict:
        if max_depth == 0:
            return randint(self.num_funcs, self.bord, size=self.pop_size)

        pop = {}
        if self.allow_leafs:
            pop['op'] = randint(0, self.bord, size=self.pop_size)
        else:
            pop['op'] = randint(0, self.num_funcs, size=self.pop_size)
        pop['left'] = self.init_pop(max_depth - 1)
        pop['right'] = self.init_pop(max_depth - 1)
        return pop

    def eval_pop(self) -> ndarray:
        return self.eval_pop_rec(self.pop)

    def eval_pop_rec(self, pop) -> ndarray:
        if isinstance(pop, ndarray):
            return self.all_vars[pop - self.num_funcs]

        left_vals = self.eval_pop_rec(pop['left'])
        right_vals = self.eval_pop_rec(pop['right'])
        ops = pop['op']
        results = empty_like(left_vals)

        for op_code, operation in self.binary_op_map.items():
            mask = (ops == op_code)
            results[mask] = operation[0](left_vals[mask], right_vals[mask])
        for func_code, func in self.unary_op_map.items():
            mask = (ops == func_code)
            results[mask] = func[0](left_vals[mask])
        for var_code, vari in self.var_map.items():
            mask = (ops == var_code)
            results[mask] = vari[0]
        return results

    def to_str_at(self, index) -> str:
        return self.to_str_at_rec(self.pop, index)

    def to_str_at_rec(self, pop, index) -> str:
        if isinstance(pop, ndarray):
            var_index = pop[index]
            return self.var_map.get(var_index)[1]

        op_code = pop['op'][index]
        if op_code in self.unary_op_map:
            func_str = self.unary_op_map[op_code][1]
            expr = self.to_str_at_rec(pop['left'], index)
            return f'{func_str}({expr})'
        if op_code in self.var_map:
            return self.var_map.get(op_code)[1]

        op_str = self.binary_op_map.get(op_code)[1]
        left_expr = self.to_str_at_rec(pop['left'], index)
        right_expr = self.to_str_at_rec(pop['right'], index)
        return f'({left_expr}{op_str}{right_expr})'

    def mutate_trees(self, mutation_prob) -> None:
        self.mutate_trees_rec(self.pop, mutation_prob)

    def mutate_trees_rec(self, pop, mutation_prob) -> dict:
        if isinstance(pop, ndarray):
            mutate_mask = rand(self.pop_size) < mutation_prob
            pop[mutate_mask] = randint(self.num_funcs, self.bord, size=self.pop_size)[mutate_mask]
            return pop

        ops = pop['op']
        mutate_mask = rand(self.pop_size) < mutation_prob
        if self.allow_leafs:
            new_ops = randint(0, self.bord, size=self.pop_size)
        else:
            new_ops = randint(0, self.num_funcs, size=self.pop_size)
        ops[mutate_mask] = new_ops[mutate_mask]
        pop['op'] = ops
        pop['left'] = self.mutate_trees_rec(pop['left'], mutation_prob)
        pop['right'] = self.mutate_trees_rec(pop['right'], mutation_prob)
        return pop

    def fitness(self, results, y_val, penalty, penalty_rate) -> ndarray:
        row_min = results.min(axis=1, keepdims=True)
        row_max = results.max(axis=1, keepdims=True)
        row_range = where(row_max - row_min == 0, 1, row_max - row_min)
        norm_results = (results - row_min) / row_range
        y_min = y_val.min()
        y_max = y_val.max()
        if y_max - y_min == 0:
            norm_y = y_val - y_min
        else:
            norm_y = (y_val - y_min) / (y_max - y_min)
        errors = mean((norm_results - norm_y) ** 2, axis=1)

        if not penalty:
            return errors

        # panlize trees with same errors with a certain rate (not all trees are panelized)
        _, inverse_indices, counts = unique(errors, return_inverse=True, return_counts=True)
        penalties = counts[inverse_indices] / self.pop_size
        penalize_mask = rand(self.pop_size) < penalty_rate
        penalties *= penalize_mask
        penalized_errors = errors + penalties

        return penalized_errors

    def selection(self, errors, selection_pressure, elitism) -> None:
        fitness_scores = 1 / (errors + 1e-6)
        adjusted_fitness = fitness_scores ** selection_pressure
        selection_probs = adjusted_fitness / adjusted_fitness.sum()
        parent_pairs = choice(
            self.pop_size,
            size=(self.pop_size, 2),
            p=selection_probs,
        )
        selected_indices = parent_pairs[:, 0]

        if elitism:
            best_index = errors.argmin()
            selected_indices[0] = best_index

        self.pop = self.copy_selection(selected_indices, self.pop)

    def copy_selection(self, indices, pop) -> dict:
        if isinstance(pop, ndarray):
            return pop[indices]

        new_pop = {}
        new_pop['op'] = pop['op'][indices]
        new_pop['left'] = self.copy_selection(indices, pop['left'])
        new_pop['right'] = self.copy_selection(indices, pop['right'])
        return new_pop

    def crossover(self, cx_prob_per_depth) -> None:
        cx_from = choice(self.pop_size, size=1, replace=False)
        cx_to = choice(self.pop_size, size=1, replace=False)
        self.crossover_rec(self.pop, cx_from, cx_to, cx_prob_per_depth)

    def crossover_rec(self, pop, cx_from, cx_to, cx_prob_per_depth) -> None:
        if isinstance(pop, ndarray):
            temp = copy(pop[cx_from])
            pop[cx_from] = pop[cx_to]
            pop[cx_to] = temp
            return

        pop['op'][cx_from], pop['op'][cx_to] = pop['op'][cx_to], pop['op'][cx_from]

        new_mask_from = rand(self.pop_size) < cx_prob_per_depth
        new_indices_from = where(new_mask_from)[0]
        new_mask_to = rand(self.pop_size) < cx_prob_per_depth
        new_indices_to = where(new_mask_to)[0]
        new_unique_from = setdiff1d(new_indices_from, cx_from)
        new_unique_to = setdiff1d(new_indices_to, cx_to)

        min_length = min(new_unique_from.size, new_unique_to.size)
        new_unique_from = new_unique_from[:min_length]
        new_unique_to = new_unique_to[:min_length]

        cx_from = concatenate((cx_from, new_unique_from))
        cx_to = concatenate((cx_to, new_unique_to))

        self.crossover_rec(pop['left'], cx_from, cx_to, cx_prob_per_depth)
        self.crossover_rec(pop['right'], cx_from, cx_to, cx_prob_per_depth)
