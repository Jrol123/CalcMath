import random

from numpy import cos, sin, pi, linspace


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
        # Если находится посередине между точками
        step = random.randint(0, 1)
        return 202 * ((-1) ** step), [index - 1, index][1 - step]


# 10
count_points = 5
range_graph = (0.1, 0.6)
full_points = linspace(*range_graph, count_points).tolist()
print(full_points)
print(check_pos(0.4125, full_points))
