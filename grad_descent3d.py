#!/usr/bin/env python3
from argparse import ArgumentParser
import sympy as sym
import sys
from sympy.abc import x, y
import math
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from numpy import *
import random


def get_args():
    parser = ArgumentParser("3d gradient descent.")
    parser.add_argument("function", help="the function to work on.")
    parser.add_argument(
        "--precision", type=float, help="target precision.", default=0.01
    )
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate.", default=0.1
    )
    parser.add_argument(
        "--max_iters", type=float, help="maximum iterations", default=100
    )
    parser.add_argument("--min_x", type=float, help="maximum x value", default=-2)
    parser.add_argument("--max_x", type=float, help="minimum x value", default=2)
    parser.add_argument("--min_y", type=float, help="maximum y value", default=-2)
    parser.add_argument("--max_y", type=float, help="minimum y value", default=2)
    parser.add_argument("--min_z", type=float, help="maximum z value", default=-2)
    parser.add_argument("--max_z", type=float, help="minimum z value", default=2)
    return parser.parse_args()


class Descender:
    def __init__(
        self, function, precision, learning_rate, max_iters, min_x, max_x, min_y, max_y
    ):
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._dx = sym.diff(function, x)
        self._dy = sym.diff(function, y)
        self._f = sym.sympify(function)
        self._precision = precision
        self._learning_rate = learning_rate
        self._max_iters = max_iters
        self.reset()

    def reset(self):
        self._cur_x = random.uniform(self._min_x, self._max_x)
        self._cur_y = random.uniform(self._min_y, self._max_y)
        self._previous_step_size = (
            self._precision + 1
        )  # just so it's bigger than precision
        self._iters = 0

    def in_bounds(self, x, y):
        return self._min_x <= x <= self._max_x and self._min_x <= y <= self._max_x

    @property
    def complete(self):
        return (
            self._previous_step_size <= self._precision
            or self._iters >= self._max_iters
            or not self.in_bounds(self.next_x, self.next_y)
        )

    @property
    def _cur_z(self):
        return float(self._f.subs([("x", self._cur_x), ("y", self._cur_y)]))

    @property
    def dxn(self):
        return float(self._dx.subs([("x", self._cur_x), ("y", self._cur_y)]))

    @property
    def dyn(self):
        return float(self._dy.subs([("x", self._cur_x), ("y", self._cur_y)]))

    @property
    def next_x(self):
        return self._cur_x - self._learning_rate * self.dxn

    @property
    def next_y(self):
        return self._cur_y - self._learning_rate * self.dyn

    def iterate(self):
        prev_x = self._cur_x
        prev_y = self._cur_y
        dxn = self.dxn
        dyn = self.dyn
        self._cur_x = self.next_x
        self._cur_y = self.next_y
        # distance formula
        self._previous_step_size = math.sqrt(
            (self._cur_x - prev_x) ** 2 + (self._cur_y - prev_y) ** 2
        )
        self._iters = self._iters + 1
        if self._iters <= 25:
            print(
                f"iter: {self._iters}, (x, y): ({prev_x:0.4f},{prev_y:0.4f}) f(x,y): {self._cur_z:0.4f}) ∇f(x,y) = <{dxn:0.4f}, {dyn:0.4f}>"
            )
        elif self._iters == 26:
            print("...")

    def print_final(self):
        print(
            f"the minimum (after {self._iters} iterations) is {self._cur_z:0.4f} and occurs at ({self._cur_x:0.4f}, {self._cur_y:0.4f})"
        )


def main():
    args = get_args()
    descender = Descender(
        args.function,
        args.precision,
        args.learning_rate,
        args.max_iters,
        args.min_x,
        args.max_x,
        args.min_y,
        args.max_y,
    )

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Make data.
    X = np.arange(args.min_x, args.max_x, 0.25)
    Y = np.arange(args.min_y, args.max_y, 0.25)
    X, Y = np.meshgrid(X, Y)
    numpified = args.function.replace("x", "X").replace("y", "Y")
    Z = eval(numpified)

    def start_drawing():
        ax.set_title(
            f"{args.function} over [(x,y) | {args.min_x} <= x <= {args.max_x}, {args.min_y} <= y <= {args.max_y}]"
        )
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        ax.set_zlabel(r"z")
        ax.set_zlim(args.min_z, args.max_z)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        surf = ax.plot_wireframe(X, Y, Z, antialiased=True, cstride=1, rstride=1)

    # Plot the surface.
    def init():
        start_drawing()
        ax.view_init(elev=40, azim=0)
        return (fig,)

    def animate(i):
        ax.scatter(
            [descender._cur_x], [descender._cur_y], [descender._cur_z], linewidth=5
        )
        if not (descender.complete):
            descender.iterate()
        if descender.complete:
            descender.print_final()
            descender.reset()
            ax.clear()
            start_drawing()
        return (fig,)

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=5)

    plt.show()


if __name__ == "__main__":
    main()
