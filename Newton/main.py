from enum import IntEnum
from pprint import pprint

from numpy import cos, sin, pi, linspace

import pandas as pd


# from random import randint


class ResponseCode(IntEnum):
    """
    Коды состояний
    """
    TABLE_POINT = 100
    OVERFLOW_FWD = 400
    OVERFLOW_BWD = -OVERFLOW_FWD
    NEWTON_FWD = 200
    NEWTON_BWD = - NEWTON_FWD
    GAUSS_FWD = 202
    GAUSS_BWD = -GAUSS_FWD
    NO_FUNC = 404  # TODO: Что делать, если точка находится не рядом с краями и не рядом с центральным элементом?


# В = 15
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


def newton_bwd(point: float, points: list[float]):
    pass


def newton_fwd(point: float, points: list[float]):
    pass


def gauss_bwd(point: float, points: list[float]):
    pass


def gauss_fwd(point: float, points: list[float]):
    pass


def redirector(point: float, points: list[float], status: tuple[ResponseCode, float | None]) -> float | None:
    """
    Функция-переадресовщик.

    Переадресует запрос в зависимости от статуса.

    :param point: Точка, значение функции в которой требуется узнать.
    :param points: Сетка значений.
    :param status: Статус выполнения подбора функции.
     В зависимости от статуса происходит переадресация

     :return: Значение полинома для точки point.
      None, если функция не была найдена.

    """
    match (status[0]):
        case ResponseCode.TABLE_POINT:
            return func(point)

        case ResponseCode.NEWTON_BWD:
            return newton_bwd(point, points)
        case ResponseCode.NEWTON_FWD:
            return newton_fwd(point, points)

        case ResponseCode.GAUSS_BWD:
            return gauss_bwd(point, points)
        case ResponseCode.GAUSS_FWD:
            return gauss_fwd(point, points)

        case ResponseCode.OVERFLOW_BWD:
            return None
        case ResponseCode.OVERFLOW_FWD:
            return None
        case ResponseCode.NO_FUNC:
            return None
        # TODO: Добавить возвращение функций


def check_pos(point: float, points: list[float]) -> tuple[ResponseCode, float | None]:
    """
    Нахождение функции, которая будет использоваться для интерполяции.

    Выбор делается между Ньютоном и Гауссом. Учитываются обе их вариации.

    Также учитывается возможность выхода функции за пределы списка и совпадение точки point с одной из точек в points.

    :param point: Точка, значение функции в которой требуется узнать.
    :param points: Сетка значений.
    :return: Возвращается набор из двух значений:
     статуса, отвечающего за то, какая функция будет выбрана;
     индекса начальной точки x_0, если таковая применима.

    """
    """Проверка на нахождение точки в массиве"""
    try:
        return ResponseCode.TABLE_POINT, points.index(point)
    except:
        print(f"{point} is not in the list")

    """Проверка на выход за пределы"""
    if point < points[0]:
        print(f"{point} is too small")
        return ResponseCode.OVERFLOW_BWD, None
    elif point > points[-1]:
        print(f"{point} is too large")
        return ResponseCode.OVERFLOW_FWD, None

    """Проверка на Ньютона"""
    if point < points[1]:
        return ResponseCode.NEWTON_BWD, 0
    elif point > points[-2]:
        return ResponseCode.NEWTON_FWD, len(points) - 1

    """Проверка на Гаусса"""
    if len(points) % 2 == 0:
        # TODO: Разобраться, почему Гаусс берётся только от центра, а Ньютон только от краёв
        return ResponseCode.NO_FUNC, None
    index = (len(points) - 1) // 2
    diff_1 = (points[index] + points[index - 1]) / 2
    diff_2 = (points[index + 1] + points[index]) / 2
    if diff_1 <= point < points[index]:
        return ResponseCode.GAUSS_FWD, index
    elif diff_2 >= point > points[index]:
        return ResponseCode.GAUSS_BWD, index

    return ResponseCode.NO_FUNC, None

    # for index in range(2, len(points) - 2 + 1):
    #     # Исключаем первую и последнюю точки
    #     diff_1 = point - points[index - 1]
    #     diff_2 = points[index] - point
    #     if diff_1 < 0 or diff_2 < 0:
    #         continue
    #     if diff_1 > diff_2:
    #         return ResponseCode.GAUSS_BWD, index - 1
    #     elif diff_1 < diff_2:
    #         return ResponseCode.GAUSS_FWD, index
    #     # Если находится посередине между точками, то выбирается случайный из двух Гауссов
    #     step = randint(0, 1)
    #     return [ResponseCode.GAUSS_BWD, ResponseCode.GAUSS_FWD][1 - step], [index - 1, index][1 - step]


def get_fin_diff(function, points: list[float]) -> list[list[float]]:
    count_pts = len(points)
    mass = [[0 for _ in range(10 - i)] for i in range(count_pts)]
    for i in range(count_pts):
        mass[0][i] = (function(points[i]))
    for i in range(1, count_pts - 1 + 1):
        for j in range(0, 10 - i - 1 + 1):
            mass[i][j] = (function(points[j]) - function(points[j + 1]))
    return mass


mass_points = [0.1, 0.13, 0.58, 0.37]  # 0.1 тестовая точка
range_graph = (0.1, 0.6)

"""Поскольку по условию сказано взять максимальное количество точек,
 то необходимо сгенерировать конечные разности для каждого случая"""

mass_fin_diff = []

for i in range(10, 3 - 1, -1):
    mass_fin_diff.append(get_fin_diff(func, (linspace(*range_graph, i)).tolist()))

pprint(mass_fin_diff, compact=True)

for cur_point in mass_points:
    # <=10
    for cur_count_points in range(10, 3 - 1, -1):
        step = (range_graph[1] - range_graph[0]) / (cur_count_points - 1)
        """Шаг сетки"""
        x_points = linspace(*range_graph, cur_count_points).tolist()
        """Точки сетки"""

        state = check_pos(cur_point, x_points)
        result = redirector(cur_point, x_points, state)
        if result is None:
            print(f"Невозможно подобрать функцию для точки {cur_point} при количестве точек = {cur_count_points}",
                  f"Ошибка: {state[0].name}", sep="\t")
            continue
        else:
            print(f"Удалось подобрать функцию для точки {cur_point} при количестве точек = {cur_count_points}",
                  f"Функция: {state[0].name}", f"Результат: {result}", sep="\t")
            break
