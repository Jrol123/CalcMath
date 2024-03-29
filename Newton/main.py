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


def check_pos(point: float, points: list[float]) -> tuple:
    try:
        return 200, points.index(point)
    except:
        print(f"{point} is not in the list")

    """Проверка на выход за пределы"""
    if point < points[0]:
        return -500, None
    elif point > points[-1]:
        return 500, None

    """Проверка на Ньютона"""
    if point < points[1]:
        return -1, 0
    elif point > points[-2]:
        return 1, len(points) - 1

    """Проверка на Гаусса"""



    # TODO: Сделать проверку на нахождение рядом с определённой точкой


count_points = 10
range_graph = (0.1, 0.6)
full_points = linspace(*range_graph, count_points).tolist()
print(check_pos(0.1, full_points))
if isinstance(1, float):
    print("flt")
