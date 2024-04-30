from math import cos, sin, pi


# В - 15
def func(x: float, deriv_s: int = 0):
    """
    Функция y = 2x− cos(x)

    При вводе аргумента deriv_s можно выбрать степень производной (в том числе и отрицательную степень, давая интеграл)

    :param x: Точка
    :param deriv_s: Степень производной

    :return: Значение функции в точке x
    :rtype: float

    """
    if deriv_s == 0:
        return 2 * x - cos(x)
    elif deriv_s == 1:
        return 2 + sin(x)
    elif deriv_s == -1:
        return x ** 2 - sin(x)
    return cos(x + ((deriv_s - 2) * pi) / 2)


def simpson(rng_graph, count_pts, fin_diff_mass: list[list[float]]):
    pass


def get_fin_diff(function, points: list[float]) -> list[list[float]]:
    """
    Вычисление конечных разностей.

    :param function: Функция
    :type function: function
    :param points: Список узлов

    :return: Список списков конечных разностей.
     Список конечных разностей поделён на подсписки степеней конечных разностей.

    """
    count_pts = len(points)
    mass = [[0 for _ in range(count_pts - i)] for i in range(count_pts)]
    for i in range(count_pts):
        mass[0][i] = (function(points[i]))
    for i in range(1, count_pts - 1 + 1):
        for j in range(0, count_pts - i - 1 + 1):
            mass[i][j] = (mass[i - 1][j + 1] - mass[i - 1][j])
    return mass


# 15 % 5 == 0 => Формула Симпсона

if __name__ == '__main__':
    range_graph = (0.1, 0.6)
    for j in range(1, 15 + 1):
        n = 2 ** j

