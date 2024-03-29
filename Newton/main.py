from numpy import cos, sin, pi, linspace

COUNT_POINTS = 10
RANGE = (0.1, 0.6)


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
    else:
        return cos(x + ((deriv_s - 2) * pi) / 2)


def newton(x: float, deriv_s: int = 0):
