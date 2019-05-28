from mxnet import nd
import numpy as np
import threading
import multiprocessing
import time


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

    def count_square(self, length, target_average):
        weight = self.matrix_path_per_step(length)
        print(weight)
        matrix_init = nd.ones(shape=(length, length))
        target_sum = target_average * weight[0][0]
        first = self.min
        while first <= self.max:
            matrix_init[0][0] = first
            self.recursion_loop(weight, matrix_init, length, target_sum.asscalar(), 0, 1)
            first += 1
        return self.count

    def recursion_loop(self, weight, matrix, length, target_sum, i, j):
        if i < length:
            print('-'.join([''.join(['{:2}'.format(int(item)) for item in row]) for row in matrix.asnumpy()])
                  + '|%d-%d|???/%d-%d|%d'
                  % (i, j, target_sum, self.count, time.time() - self.start))
        for v in range(9):
            matrix[i][j] = v + 1
            current_average = self.sum_all_paths(weight, matrix)
            if current_average == target_sum:
                self.count += 1
                if self.count % 100 == 1:
                    print('-'.join([''.join(['{:2}'.format(int(item)) for item in row]) for row in matrix.asnumpy()])
                          + '|%d-%d|%d/%d-%d|%d'
                          % (i, j, current_average, target_sum, self.count, time.time() - self.start))
                return
            elif current_average < target_sum:
                i2, j2 = self.increment(i, j, length)
                if i2 < length:
                    self.recursion_loop(weight, matrix.copy(), length, target_sum, i2, j2)
            else:
                return
        return


class MultiSquare:

    results = nd.zeros(10)

    def __init__(self, length, target_average):
        self.length = length
        self.target_average = target_average

    def run_a_tenth(self, number, return_dict):
        self.results[number] = Square(number, number, time.time()).count_square(self.length, self.target_average)
        total = self.results[number].asscalar()
        print("## TOTAL: %d" % total)
        return_dict[number] = total
        return total

    def run(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []

        for i in range(10):
            p = multiprocessing.Process(target=self.run_a_tenth, args=(i, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        print("## TOTAL OVERALL: %d" % (sum(return_dict.values())))


MultiSquare(11, 110).run()
# count = Square(0, 9, time.time()).count_square(2, 10)
# print("## TOTAL: %d" % count)

