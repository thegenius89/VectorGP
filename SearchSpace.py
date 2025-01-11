from numpy import sin, cos, tan, add, subtract, multiply, divide, exp, log, sqrt, absolute, pi
from numpy import ndarray, where, full


def pdiv(a: ndarray, b: ndarray) -> ndarray:
    return divide(a, where((absolute(b) <= 0.000001), 1, b))


def psqrt(a: ndarray) -> ndarray:
    return sqrt(absolute(a))


def plog(a: ndarray) -> ndarray:
    return log(absolute(where((a <= 0), 1, a)))


def pexp(a: ndarray) -> ndarray:
    return exp(where((absolute(a) > 5), 0, a))


def sq(a: ndarray) -> ndarray:
    return a * a


class SearchSpace:

    def __init__(self, train):
        self.binary_op_map = {
            0: (add, '+'),
            1: (subtract, '-'),
            2: (multiply, '*'),
            3: (pdiv, '/'),
        }
        self.unary_op_map = {
            4: (sin, 'sin'),
            5: (cos, 'cos'),
            6: (tan, 'tan'),
            7: (psqrt, 'sqrt'),
            8: (plog, 'log'),
            9: (pexp, 'exp'),
            10: (sq, 'sq'),
        }
        self.var_map = {}
        self.num_funcs = len(self.unary_op_map) + len(self.binary_op_map)
        for input in range(train.shape[0]):
            index = self.num_funcs + input
            self.var_map[index] = (train[input], 'x{}'.format(input))
        self.x_size = train[0].size
        index_add = len(self.var_map) + self.num_funcs
        self.var_map.update({
            index_add + 0: (full(self.x_size, 0.5), '0.5'),
            index_add + 1: (full(self.x_size, 1.0), '1.0'),
            index_add + 2: (full(self.x_size, 2.0), '2.0'),
            index_add + 3: (full(self.x_size, pi), 'PI'),
            index_add + 4: (full(self.x_size, pi * 0.5), 'PI/2'),
        })
