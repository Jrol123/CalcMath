from math import factorial
from numpy import cos, sin, pi, linspace

import pandas as pd



def lagrange(bp: float, points: list[tuple[float, float]]) -> float:
    """
    Неньютоновская реализация полинома Лагранжа

    :param points: Список точек
    :type points: list[tuple[float, float]]
    :param bp: Точка, значение функции в которой нужно получить
    :type bp: float

    :return: Значение полинома Лагранжа в точке bp
    :rtype: float

    """
    pass

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
        return x + sin(x)
    return cos(x + ((deriv_s - 2) * pi) / 2)


def get_norm(function, rng: tuple[float, float], *args) -> float:
    """
    Получение нормы функции

    :param function: Некоторая функция, норму которой мы хотим получить
    :param rng: Кортеж границ
    :param args: Дополнительные аргументы функции

    :return: Норма функции
    :rtype: float

    """
    return max(abs(function(linspace(*rng, num=10 ** 3), *args)))


def teor_error(count_points: int, rng: tuple[float, float], function) -> tuple[float, float]:
    """
    Получение теоретической ошибки.
    Теоретическая ошибка берётся от максимума и минимума функции.

    :param function: Функция
    :param count_points: Количество точек
    :param rng: Кортеж границ

    :return: Теоретическая ошибка
    :rtype: float

    """
    pts = function(linspace(*rng, num=10 ** 3), count_points + 1)
    fct = factorial(count_points + 1)
    diff = ((rng[1] - rng[0]) ** (count_points + 1))
    return (min(pts) / fct) * diff, (max(pts) / fct) * diff


k, n, m = 1, 5, 5
