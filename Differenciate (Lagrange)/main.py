from math import factorial
from numpy import cos, sin, pi, linspace

import pandas as pd


def lagrange(num_point: int, points: list[tuple[float, float]], step: float) -> float:
    """
    Неньютоновская реализация производной от полинома Лагранжа первой степени

    :param points: Список точек
    :type points: list[tuple[float, float]]
    :param num_point: Номер точки
    :type num_point: int
    :param step: Шаг сетки
    :type step: float

    :return: Значение полинома Лагранжа в точке bp
    :rtype: float

    """
    count_points = len(points)
    result = 0
    for i, point in enumerate(points):
        point_mult = point[1] / step
        diff_mult = 1

        for j in range(0, i - 1 + 1):
            diff_mult *= (i - j)
        for j in range(i + 1, count_points):
            diff_mult *= (i - j)

        grid_mult = 1
        for j in range(0, count_points):
            for j1 in range(0, min(i, j) - 1 + 1):
                pass
            for j1 in range(min(i, j) + 1, max(i, j) - 1 + 1):
                grid_mult *= (num_point - j)
            for j1 in range(max(i, j) + 1, count_points):
                grid_mult *= (num_point - j)
        result += point_mult * diff_mult * grid_mult

    return result


# В - 15
def func(x: float, deriv_s: int = 0):
    """
    Функция y = 2x− cos(x)

    При вводе аргумента deriv_s можно выбрать степень производной

    :param x: Точка
    :param deriv_s: Степень производной

    :return: Значение функции в точке x
    :rtype: float

    """
    if deriv_s == 0:
        return 2 * x - cos(x)
    elif deriv_s == 1:
        return x + sin(x)
    return cos(x + ((deriv_s - 2) * pi) / 2)


def get_norm(function, rng: tuple[float, float], *args) -> float:
    """
    Получение нормы функции

    :param function: Некоторая функция, норму которой мы хотим получить
    :param rng: Кортеж границ
    :param args: Дополнительные аргументы функции

    :return: Норма функции
    :rtype: float

    """
    return max(abs(function(linspace(*rng, num=10 ** 3), *args)))


def teor_error(count_points: int, rng: tuple[float, float], function) -> tuple[float, float]:
    """
    Получение теоретической ошибки.
    Теоретическая ошибка берётся от максимума и минимума функции.

    :param function: Функция
    :param count_points: Количество точек
    :param rng: Кортеж границ

    :return: Теоретическая ошибка
    :rtype: float

    """
    pts = function(linspace(*rng, num=10 ** 3), count_points + 1)
    fct = factorial(count_points + 1)
    diff = ((rng[1] - rng[0]) ** (count_points + 1))
    return (min(pts) / fct) * diff, (max(pts) / fct) * diff

if __name__ == "__main__":
    k, count_pts, index_point = 1, 5, 5
    range_graph = (0.1, 0.6)
    step_grid = (range_graph[1] - range_graph[0]) / count_pts
    mass_points = [(pt, func(pt)) for pt in linspace(*range_graph, count_pts)]

    res_lagrange = lagrange(index_point, mass_points, step_grid)
    res_func = func(mass_points[index_point - 1][0], k)
    # TODO: Выходит дикий Лагранж. Необходимо пофиксить
    print(res_lagrange, res_func, res_lagrange - res_func)
