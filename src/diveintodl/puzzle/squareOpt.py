from mxnet import nd
import numpy as np
import multiprocessing
import time
import operator as op
from functools import reduce


class Square:
    count = 0

    def __init__(self, min, max, time):
        self.min = min
        self.max = max
        self.start = time

    def matrix_remaining_path(self, length):
        remain = nd.ones(shape=(length, length))
        for i in range(length - 1):
            for j in range(length - 1):
                remain[i + 1][j + 1] = remain[i + 1][j] + remain[i][j + 1]
        return remain

    def matrix_path_per_step(self, length):
        remain = self.matrix_remaining_path(length)
        past = nd.array(np.rot90(np.rot90(remain.asnumpy())))
        total = nd.elemwise_mul(remain, past)
        return total  # total[0][0]  Lower the overall score by computing directly the average, but then work of floats

    def sum_all_paths(self, weight, matrix):
        elem_prod = nd.elemwise_mul(weight, matrix)
        elem_sum = nd.sum(elem_prod)
        return elem_sum.asscalar()

    def increment(self, i, j, l):
        i2, j2 = i, j
        if j == l - 1:
            j2 = 0
            i2 = i + 1
        else:
            j2 = j + 1
        return i2, j2

    def max_potential_remaining(self, weight):
        length = len(weight)
        max_potential = nd.zeros(shape=(length, length))
        for i in reversed(range(length)):
            for j in reversed(range(length)):
                i_prev = i
                j_prev = j + 1
                total_prev = 0
                if j_prev == length:
                    i_prev = i + 1
                    j_prev = 0
                if i_prev < length:
                    total_prev = max_potential[i_prev][j_prev]
                max_potential[i][j] = 9 * weight[i][j] + total_prev
        return max_potential

    def count_square(self, length, target_average):
        weight = self.matrix_path_per_step(length)
        print(weight)
        matrix_init = nd.ones(shape=(length, length))
        target_sum = target_average * weight[0][0]
        max_remaining = self.max_potential_remaining(weight)
        print(max_remaining)
        first = self.min





class Step:
    count = 0
    num_path = 0
    max_remaining = 0
    weight = 0

    def __init__(self, num_path):
        self.num_path = num_path
        self.count = 1

    def combinations(self, occurences):
        return ncr(self.count, occurences)

    def range(self):
        for x in range(self.count * 8):
            yield x + self.count

    def inc_count(self):
        self.count += 1
        self.weight += 1


class ListStep:
    all_steps = []
    total_path = 0

    def __init__(self, matrix):
        length = len(matrix[0])
        self.total_path = matrix[0][0]
        for i in range(length):
            for j in range(length):
                num_path = matrix[i][j]
                element = first(x.num_path == num_path for x in self.all_steps)
                if element is None:
                    self.all_steps.append(Step(num_path))
                else:
                    element.inc_count()
        self.compute_remaining()

    def compute_remaining(self):
        length = len(self.all_steps)
        for i in reversed(range(length)):
            prev_max_value = 0
            if i < length - 1:
                prev = self.all_steps[i+1]
                prev_max_value = prev.max_remaining
            current = self.all_steps[i]
            current.max_remaining = prev_max_value + current.num_path * 9 * current.count


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def first(iterable):
    for element in iterable:
        if element:
            return element
    return None


















class MultiSquare:
    results = nd.zeros(shape=(10, 1))

    def __init__(self, length, target_average):
        self.length = length
        self.target_average = target_average

    def run_a_tenth(self, number, return_dict):
        self.results[number][0] = Square(number, number, time.time()).count_square(self.length, self.target_average)
        total = self.results[number].asscalar()
        print("## TOTAL: %d" % total)
        return_dict[number] = total
        return total

    def run(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []

        for i in range(9):
            p = multiprocessing.Process(target=self.run_a_tenth, args=(i + 1, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        print("## TOTAL OVERALL: %d" % (sum(return_dict.values())))


# MultiSquare(3, 20).run()
count = Square(0, 9, time.time()).count_square(11, 110)
# print("## TOTAL: %d" % count)
