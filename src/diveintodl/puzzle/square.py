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
        if i < length -1:
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

    def run_a_tenth(self, number):
        self.results[number] = Square(number, number, time.time()).count_square(self.length, self.target_average)
        print("## TOTAL: %d" % self.results[number].asscalar())

    def run_zero(self):
        self.run_a_tenth(0)

    def run_one(self):
        self.run_a_tenth(1)

    def run_two(self):
        self.run_a_tenth(2)

    def run_three(self):
        self.run_a_tenth(3)

    def run_four(self):
        self.run_a_tenth(4)

    def run_five(self):
        self.run_a_tenth(5)

    def run_six(self):
        self.run_a_tenth(6)

    def run_seven(self):
        self.run_a_tenth(7)

    def run_eight(self):
        self.run_a_tenth(8)

    def run_nine(self):
        self.run_a_tenth(9)


    def run(self):
        t0 = threading.Thread(target=self.run_zero, name='t0')
        t1 = threading.Thread(target=self.run_one, name='t1')
        t2 = threading.Thread(target=self.run_two, name='t2')
        t3 = threading.Thread(target=self.run_three, name='t3')
        t4 = threading.Thread(target=self.run_four, name='t4')
        t5 = threading.Thread(target=self.run_five, name='t5')
        t6 = threading.Thread(target=self.run_six, name='t6')
        t7 = threading.Thread(target=self.run_seven, name='t7')
        t8 = threading.Thread(target=self.run_eight, name='t8')
        t9 = threading.Thread(target=self.run_nine, name='t9')
        t0.start()
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
        t8.start()
        t9.start()

        # wait until all threads finish
        t0.join()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()
        t7.join()
        t8.join()
        t9.join()

        print("## TOTAL OVERALL: %d" % (nd.sum(self.results).asscalar()))


MultiSquare(2, 10).run()
count = Square(0, 9, time.time()).count_square(2, 10)
print("## TOTAL: %d" % count)

