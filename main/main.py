import math
import random as rand
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


class Line:
    head_x = 10
    head_y = 10
    tail_x = 0
    tail_y = 0
    length = 0
    k = 0
    b = 0

    def convert_to_2d(self):
        return plt.Line2D([self.tail_x, self.head_x], [self.tail_y, self.head_y])

    def get_function(self):
        self.k = (self.head_y - self.tail_y) / (self.head_x - self.tail_x)
        self.b = self.tail_y - self.k * self.tail_x
        return lambda x: self.k * x + self.b

    def contains(self, x):
        if self.tail_x < self.head_x:
            return (self.tail_x - 1 <= x) & (self.head_x + 1 >= x)
        else:
            return (self.head_x - 1 <= x) & (self.tail_x + 1 >= x)

    def __init__(self, tail_x, tail_y, head_x, head_y, length):
        self.head_x = head_x
        self.head_y = head_y
        self.tail_x = tail_x
        self.tail_y = tail_y
        self.length = length
        self.function = self.get_function()


class Sphere:
    x = 0
    y = 0
    r = 0

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


maxX = 100
maxY = 100
maxR = 10
minX = 0
minY = 0
minR = 1

minLineX = 0
minLineY = 0
degree = 0
minLengthLine = 5
maxLengthLine = 30
lengthLine = rand.randrange(minLengthLine, maxLengthLine)
maxLineX = maxX - lengthLine
maxLineY = maxY - lengthLine

spheres = list()


def main():
    training_epochs = 500
    n_neurons_in_h1 = 60
    n_neurons_in_h2 = 60
    learning_rate = 0.01

    # countSphere = 50
    # ax = plt.gca()
    #
    # # buf_sphere = Sphere(10, 10, 10)
    # # spheres.append(buf_sphere)
    # # buf_line = Line(10, 0, 30, 20)
    # #
    # # ax.add_line(buf_line.convert_to_2d())
    #
    # for i in range(countSphere):
    #     spheres.append(add_new_sphere())
    #
    # for sphere in spheres:
    #     print(sphere.x, sphere.y, sphere.r)
    #     ax.add_patch(plt.Circle((sphere.x, sphere.y), sphere.r))
    #
    # line = create_line()
    # ax.add_line(line.convert_to_2d())
    # ax.add_line(create_end_line(line).convert_to_2d())
    # # ax.add_line(plt.Line2D([0, 10], [0, 10]))
    #
    # plt.axis("scaled")
    # plt.show()
    #
    # for sphere in spheres:
    #     print("intersection: ", intersection(sphere, line))




def intersection(sphere, line):
    b = -2 * (sphere.x - line.k * line.b + line.k * sphere.y)
    a = 1 + line.k * line.k
    c = sphere.x * sphere.x + line.b * line.b - 2 * line.b * sphere.y + sphere.y * sphere.y - sphere.r * sphere.r
    d = b * b - 4 * a * c
    print(d)
    if d < 0:
        return False
    elif d == 0:
        x = -b / (2 * a)
        return line.contains(x)
    else:
        x1 = (-b + math.sqrt(d)) / (2 * a)
        x2 = (-b - math.sqrt(d)) / (2 * a)
        return line.contains(x1) | line.contains(x2)


def create_line():
    while True:
        flag = False
        degree = rand.random() * 2 * math.pi
        print(degree)
        length = rand.randrange(minLengthLine, maxLengthLine)
        if (degree > math.pi / 2) & (degree < 2 * math.pi / 3):
            head_x, head_y = (rand.randrange(minLineX + length, maxLineX), rand.randrange(minLineY + length, maxLineY))
        else:
            head_x, head_y = (rand.randrange(minLineX, maxLineX - length), rand.randrange(minLineY, maxLineY - length))
        print(degree)
        tail_x, tail_y = (head_x + length * math.cos(degree), head_y + length * math.sin(degree))
        print(tail_x, tail_y)
        line = Line(tail_x, tail_y, head_x, head_y, length)
        for sphere in spheres:
            flag |= intersection(sphere, line)
            if flag:
                break
        if not flag:
            return line


def create_end_line(start_line):
    while True:
        flag = False
        degree = rand.random() * 360
        print(degree)
        length = start_line.length
        if (degree > math.pi / 2) & (degree < 2 * math.pi / 3):
            head_x, head_y = (rand.randrange(minLineX + length, maxLineX), rand.randrange(minLineY + length, maxLineY))
        else:
            head_x, head_y = (rand.randrange(minLineX, maxLineX - length), rand.randrange(minLineY, maxLineY - length))
        print(degree)
        tail_x, tail_y = (head_x + length * math.cos(degree), head_y + length * math.sin(degree))
        print(tail_x, tail_y)
        line = Line(tail_x, tail_y, head_x, head_y, length)
        for sphere in spheres:
            flag |= intersection(sphere, line)
            if flag:
                break
        if not flag:
            if not intersection_with_line(start_line, line):
                return line


def intersection_with_line(start, end):
    return start.contains((end.b - start.b) / (start.k - end.k))


def add_new_sphere():
    while (True):
        randomSphere = get_random_sphere()
        flag = True
        for sphere in spheres:
            print(distance(randomSphere, sphere), randomSphere.r + sphere.r)
            if distance(randomSphere, sphere) < randomSphere.r + sphere.r:
                flag = False
        if flag:
            break
    return randomSphere


def distance(first, second):
    return math.sqrt(math.pow(first.x - second.x, 2) + math.pow(first.y - second.y, 2))


def get_random_sphere():
    r = rand.randrange(minR, maxR)
    x_0 = rand.randrange(minX + r, maxX - r)
    y_0 = rand.randrange(minY + r, maxY - r)
    return Sphere(x_0, y_0, r)


if __name__ == "__main__":
    main()
