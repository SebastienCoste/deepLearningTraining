from mxnet import  nd
import numpy as np
import threading
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
        matrix_init = nd.ones(shape=(length, length))
        target_sum = target_average * weight[0][0]
        first = self.min
        while first <= self.max:
            matrix_init[0][0] = first
            self.recursion_loop(weight, matrix_init, length, target_sum.asscalar(), 0, 1)
            first += 1
        return self.count

    def recursion_loop(self, weight, matrix, length, target_sum, i, j):
        for v in range(9):
            matrix[i][j] = v + 1
            current_average = self.sum_all_paths(weight, matrix)
            if current_average == target_sum:
                self.count += 1
                print(' -'.join([''.join(['{:4}'.format(int(item)) for item in row]) for row in matrix.asnumpy()])
                      + ' | %d-%d:%d | %f/%d - %d | %d'
                      % (i, j, v+1, current_average, target_sum, self.count, time.time() - self.start))
                return
            elif current_average < target_sum:
                i2, j2 = self.increment(i, j, length)
                if i2 < length:
                    self.recursion_loop(weight, matrix.copy(), length, target_sum, i2, j2)
            else:
                return
        return


class MultiSquare:
    n_zero_to_two = 0
    n_three_to_five = 0
    n_six_to_seven = 0
    n_eight_to_nine = 0

    def __init__(self, length, target_average):
        self.length = length
        self.target_average = target_average

    def zero_to_two(self):
        self.n_zero_to_two = Square(0, 2, time.time()).count_square(self.length, self.target_average)
        print("## TOTAL: %d" % self.n_zero_to_two)

    def three_to_five(self):
        self.n_three_to_five = Square(3, 5, time.time()).count_square(self.length, self.target_average)
        print("## TOTAL: %d" % self.n_three_to_five)

    def six_to_seven(self):
        self.n_six_to_seven = Square(6, 7, time.time()).count_square(self.length, self.target_average)
        print("## TOTAL: %d" % self.n_six_to_seven)

    def eight_to_nine(self):
        self.n_eight_to_nine = Square(8, 9, time.time()).count_square(self.length, self.target_average)
        print("## TOTAL: %d" % self.n_eight_to_nine)

    def run(self):
        t1 = threading.Thread(target=self.zero_to_two, name='t1')
        t2 = threading.Thread(target=self.three_to_five, name='t2')
        t3 = threading.Thread(target=self.six_to_seven, name='t3')
        t4 = threading.Thread(target=self.eight_to_nine, name='t4')

        # starting threads
        t1.start()
        t2.start()
        t3.start()
        t4.start()

        # wait until all threads finish
        t1.join()
        t2.join()
        t3.join()
        t4.join()

        print("## TOTAL OVERALL: %d" % (self.n_zero_to_two + self.n_three_to_five
                                        + self.n_six_to_seven + self.n_eight_to_nine))


MultiSquare(3, 20).run()
# count = Square(0, 9, time.time()).count_square(2, 20)
# print("## TOTAL: %d" % count)
