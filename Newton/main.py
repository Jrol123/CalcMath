from random import randint
from enum import IntEnum

from numpy import cos, sin, pi, linspace


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
    NOT_NG = 404  # TODO: Что делать, если точка находится не рядом с краями и не рядом с центральным элементом?


# 0.13, 0.56, 0.37
# (0.1, 0.6)
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


def redirector(point: float, points: list[float], status: tuple[ResponseCode, float | None]):
    match (status[0]):
        case ResponseCode.TABLE_POINT:
            return func(point)

        case ResponseCode.NEWTON_BWD:
            pass
        case ResponseCode.NEWTON_FWD:
            pass

        case ResponseCode.GAUSS_BWD:
            pass
        case ResponseCode.GAUSS_FWD:
            pass

        case ResponseCode.OVERFLOW_BWD:
            return None
        case ResponseCode.OVERFLOW_FWD:
            return None
        case ResponseCode.NOT_NG:
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
        return ResponseCode.NOT_NG, None
    index = (len(points) - 1) // 2
    diff_1 = (points[index] + points[index - 1]) / 2
    diff_2 = (points[index + 1] + points[index]) / 2
    if diff_1 <= point < points[index]:
        return ResponseCode.GAUSS_FWD, index
    elif diff_2 >= point > points[index]:
        return ResponseCode.GAUSS_BWD, index

    return ResponseCode.NOT_NG, None

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


# <=10
max_count_points = 5
range_graph = (0.1, 0.6)
x_points = linspace(*range_graph, max_count_points).tolist()
pt = 0.4125
pt = 0.7
state = check_pos(pt, x_points)
# print(x_points)
# print(state, x_points[state[1]])
result = redirector(pt, x_points, state)
if result is None:
    print(f"Невозможно подобрать функцию при количестве точек = {max_count_points}")
