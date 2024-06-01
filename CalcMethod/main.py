"""
Поиск корней необходимо осуществлять с помощью трех методов: хорд, касательных, дихотомии.
Алгоритм пристрелки для каждого метода должен быть одинаков.
Для каждого из методов измерить время решения задачи.
Сделать вывод.
"""
from numpy import exp
from datetime import datetime, timedelta


class OutputEntity:
    def __init__(self, name_function: str, count_iterations: int, precision: float,  time: timedelta):
        self.name_function = name_function
        self.count_iterations = count_iterations
        self.precision = precision
        self.time = time

    def __str__(self):
        return (f"\nФункция:\t{self.name_function}\nКоличество итераций:\t{self.count_iterations}\n"
                f"Использованная точность:\t{self.precision}\nЗатраченное время:\t{self.time}")


def secant(f, x_prev: float, x_main: float) -> float:
    """
    Метод хорд

    :param f: Функция
    :type f: function
    :param x_prev: Предыдущее значение x
    :param x_main: Значение x, откуда идёт счёт

    :return: Значение x

    """
    return x_main - (f(x_main) * (x_prev - x_main)) / (f(x_prev) - f(x_main))


def newton(f, x_prev: float, x_main) -> float:
    """

    :param f: Функция
    :type f: function
    :param x_prev: Предыдущее значение x

    :return: Значение x

    """
    return x_prev - f(x_prev) / f(x_prev, 1)


def get_result(function, mass_borders, eps):
    """
    Получение результатов из функций

    :param function:
    :type function: function
    :param mass_borders:
    :param eps: Точность

    :return:

    """
    x_main = mass_borders[0]
    x_prev = 0
    if function.__name__ == secant.__name__:
        x_prev = mass_borders[1]
    else:
        x_prev = x_main
    delta = abs(mass_borders[0] - mass_borders[1])
    i = 0
    date_start = datetime.now()
    while delta > eps:
        x_cur = function(func, x_prev, x_main)
        print(i := i + 1, x_prev, x_cur)
        delta = abs(x_cur - x_prev)
        x_prev = x_cur
    date_end = datetime.now()
    # print((date_end - date_start))
    return OutputEntity(function.__name__, i, eps, (date_end - date_start))


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
    if deriv_s == 1:
        return -exp(-x) + 2 * x
    return exp(-x) + x ** 2 - 2


if __name__ == '__main__':
    # Start here
    range_mass = [-0.75, -0.25]  # Массив границ
    eps = 1e-100
    print(get_result(secant, range_mass, eps))
