"""
Поиск корней необходимо осуществлять с помощью трех методов: хорд, касательных, дихотомии.
Алгоритм пристрелки для каждого метода должен быть одинаков.
Для каждого из методов измерить время решения задачи.
Сделать вывод.
"""
from math import cos, sin, pi


def secant(f, x_prev: float) -> float:
    """
    Метод хорд

    :param f: Функция
    :type f: function
    :param x_prev: Предыдущее значение x

    :return:

    """
    pass


def newton(f, x_prev: float) -> float:
    """

    :param f: Функция
    :type f: function
    :param x_prev: Предыдущее значение x

    :return:

    """
    return x_prev - f(x_prev) / f(x_prev, 1)


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
        return 2 + sin(x)
    return cos(x + ((deriv_s - 2) * pi) / 2)


if __name__ == '__main__':
    # Start here
    range_mass = [0.1, 0.6]  # Массив границ
    eps = 1e-5
