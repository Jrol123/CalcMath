import random

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
    LAGRANGE = 500 # TODO: Что делать, если точка находится не рядом с краями и не рядом с центральным элементом?



# 0.13, 0.56, 0.37
# (0.1, 0.6)
# В = 15
def func(x: float, deriv_s: int = 0):
    """
    Функция y = 2x− cos(x)

    При вводе аргумента deriv_s можно выбрать степень производной

    :param x: Точка x
    :param deriv_s: Степень производной

    :return: Значение функции в точке x
    :rtype: float

    """
    if deriv_s == 0:
        return 2 * x - cos(x)
    elif deriv_s == 1:
        return x + sin(x)
    return cos(x + ((deriv_s - 2) * pi) / 2)


def newton(rng: tuple[float, float], step: float, count_pts: int, is_fwd: bool = True):
    pass


def check_pos(point: float, points: list[float]) -> tuple[int, float | None]:
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
        return 100, points.index(point)
    except:
        print(f"{point} is not in the list")

    """Проверка на выход за пределы"""
    if point < points[0]:
        return -400, None
    elif point > points[-1]:
        return 400, None

    """Проверка на Ньютона"""
    if point < points[1]:
        return -200, 0
    elif point > points[-2]:
        return 200, len(points) - 1

    """Проверка на Гаусса"""
    for index in range(2, len(points) - 2 + 1):
        # Исключаем первую и последнюю точки
        diff_1 = point - points[index - 1]
        diff_2 = points[index] - point
        if diff_1 < 0 or diff_2 < 0:
            continue
        if diff_1 > diff_2:
            return -2, index - 1
        elif diff_1 < diff_2:
            return 2, index
        # Если находится посередине между точками, то выбирается случайный из двух Гауссов
        step = random.randint(0, 1)
        return 202 * ((-1) ** step), [index - 1, index][1 - step]


# 10
count_points = 5
range_graph = (0.1, 0.6)
x_points = linspace(*range_graph, count_points).tolist()
res = check_pos(0.4125, x_points)
print(x_points)
print(res, x_points[res[1]])
