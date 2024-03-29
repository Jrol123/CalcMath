from numpy import cos, sin, pi, linspace


# 0.44, 0.13, 0.56, 0.37
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


def check_pos(point: float, points: list[float]) -> tuple:
    if point in points:
        # Если точка находится в массиве точек
        # TODO: Возвращать индекс в массиве
        return 200, point

    if point < points[0]:
        return -500, None
    elif point > points[-1]:
        return 500, None

    if point < points[1]:
        return -1
    elif point > points[-2]:
        return 1

    # TODO: Сделать проверку на нахождение рядом с определённой точкой



count_points = 10
range_graph = (0.1, 0.6)
full_points = linspace(*range_graph, count_points)
print(check_pos(range_graph, 0.1))
if isinstance(1, float):
    print("flt")
