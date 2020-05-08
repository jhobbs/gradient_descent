#!/usr/bin/env python3
from argparse import ArgumentParser
import sympy as sym
import sys
from sympy.abc import x, y
import math

def get_args():
    parser = ArgumentParser("3d gradient descent.")
    parser.add_argument("function", help="the function to work on.")
    parser.add_argument("start_x", type=float, help="starting x-coordinate.")
    parser.add_argument("start_y", type=float, help="starting y-coordinate.")
    parser.add_argument("--precision", type=float, help="target precision.", default=0.00001)
    parser.add_argument("--learning_rate",type=float, help="learning rate.", default=0.1)
    parser.add_argument("--max_iters", type=float, help="maximum iterations", default=10000)
    return parser.parse_args()


class Descender():

    def __init__(self, function, start_x, start_y, precision, learning_rate, max_iters):
        self._dx = sym.diff(function, x)
        self._dy = sym.diff(function, y)
        self._f = sym.sympify(function)
        self._cur_x = start_x
        self._cur_y = start_y
        self._precision = precision
        self._learning_rate = learning_rate
        self._previous_step_size = precision + 1  # just so it's bigger than precision
        self._max_iters = max_iters
        self._iters = 0

    @property
    def complete(self):
        return self._previous_step_size <= self._precision or self._iters >= self._max_iters


    def iterate(self):
        prev_x = self._cur_x
        prev_y = self._cur_y
        dxn = float(self._dx.subs([('x', prev_x), ('y', prev_y)]))
        dyn = float(self._dy.subs([('x', prev_x), ('y', prev_y)]))
        self._cur_x = self._cur_x - self._learning_rate * dxn
        self._cur_y = self._cur_y - self._learning_rate * dyn
        cur_zf = self._f.subs([('x', self._cur_x), ('y', self._cur_y)])
        self._cur_z = float(self._f.subs([('x', self._cur_x), ('y', self._cur_y)]))
        # distance formula
        self._previous_step_size = math.sqrt((self._cur_x - prev_x)**2 + (self._cur_y - prev_y)**2)
        self._iters = self._iters + 1
        if self._iters <= 25:
            print(f"iter: {self._iters}, (x, y): ({prev_x:0.4f},{prev_y:0.4f}) f(x,y): {self._cur_z:0.4f}) ∇f(x,y) = <{dxn:0.4f}, {dyn:0.4f}>")
        elif self._iters == 26:
            print("...")

    def print_final(self):
        print(f"the minimum (after {self._iters} iterations) is {self._cur_z:0.4f} and occurs at ({self._cur_x:0.4f}, {self._cur_y:0.4f})")

        


def main():
    args = get_args()
    descender = Descender(args.function, args.start_x, args.start_y, args.precision, args.learning_rate, args.max_iters)

    while not descender.complete:
        descender.iterate()

    descender.print_final()


if __name__ == '__main__':
    main()
