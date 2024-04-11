from enum import IntEnum
from math import factorial

from numpy import cos, sin, pi, linspace

import pandas as pd


# from random import randint


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
    NO_FUNC = 404  # TODO: Что делать, если точка находится не рядом с краями и не рядом с центральным элементом?

    BAD_PARAMETER = -404


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


def newton_fwd(point: float, points: list[float], step: float, mass_fin_dir: list[list[float]]) -> float | ResponseCode:
    """
    Функция Ньютона вперёд.

    Берётся от первой точки и вычисляет полином для точки, находящейся между первой и второй.
    Имеется проверка на границы по t, хотя это не нужно в рамках функции check_pos.

    :param point: Точка, значение функции в которой требуется узнать.
    :param points: Сетка значений.
    :param step: Шаг сетки.
    :param mass_fin_dir: Массив конечных значений.

    :return: Значение функции в точке.

    """
    t = (point - points[0]) / step  # TODO: Вытащить вычисление t в функцию redirector. Хотя, а стоит ли?
    if not 1 > t > 0:
        print("ALARM!", f"{point} IS BROKEN FOR NEWTON_FWD!", f"t = {t}!", sep="\t")
        return ResponseCode.BAD_PARAMETER

    result = 0
    for i in range(len(points)):
        mult = mass_fin_dir[i][0]
        for j in range(0, i - 1 + 1):
            mult *= (t - j)
        result += (mult / factorial(i))
    return result


def newton_bwd(point: float, points: list[float], step: float, mass_fin_dir: list[list[float]]) -> float | ResponseCode:
    """
    Функция Ньютона назад.

    Берётся от последней точки и вычисляет полином для точки, находящейся между предпоследней и последней.
    Имеется проверка на границы по t, хотя это не нужно в рамках функции check_pos.

    :param point: Точка, значение функции в которой требуется узнать.
    :param points: Сетка значений.
    :param step: Шаг сетки.
    :param mass_fin_dir: Массив конечных значений.

    :return: Значение функции в точке.

    """
    t = (point - points[-1]) / step  # TODO: Вытащить вычисление t в функцию redirector. Хотя, а стоит ли?
    if not 0 > t > -1:
        print("ALARM!", f"{point} IS BROKEN FOR NEWTON_BWD!", f"t = {t}!", sep="\t")
        return ResponseCode.BAD_PARAMETER

    result = 0
    for i in range(len(points)):
        mult = mass_fin_dir[i][-1]
        for j in range(0, i - 1 + 1):
            mult *= (t + j)
        result += (mult / factorial(i))
    return result


def gauss_fwd(point: float, points: list[float], step: float, mass_fin_dir: list[list[float]]) -> float | ResponseCode:
    """
    Функция Гаусса вперёд/назад.

    Берётся от центральной точки и вычисляет полином для точки, находящейся между n/2 и n/2+1 -й.
    Имеется проверка на границы по t, хотя это не нужно в рамках функции check_pos.

    :param point: Точка, значение функции в которой требуется узнать.
    :param points: Сетка значений.
    :param step: Шаг сетки.
    :param mass_fin_dir: Массив конечных значений.

    :return: Значение функции в точке.

    """
    t = (point - points[
        (len(points) - 1) // 2]) / step  # TODO: Вытащить вычисление t в функцию redirector. Хотя, а стоит ли?
    if not 0 < t <= 0.5:
        print("ALARM!", f"{point} IS BROKEN FOR GAUSS_FWD!", f"t = {t}!", sep="\t")
        return ResponseCode.BAD_PARAMETER

    result = 0
    for i in range(len(points)):

        fin_diff_step = mass_fin_dir[i]
        if i % 2 == 0:
            diff_index = (len(fin_diff_step) - 1) // 2
        else:
            """Идти наверх"""
            diff_index = (len(fin_diff_step)) // 2 - 1

        mult = fin_diff_step[diff_index]
        for j in range(0, i - 1 + 1):
            if j % 2 != 0:
                mult *= (t - (j + 1) // 2)
                # mult *= (t - j)
            else:
                mult *= (t + (j + 1) // 2)
                # mult *= (t + j)
                # TODO: Разобраться, какой способ более подходящий
        result += (mult / factorial(i))

    return result


def gauss_bwd(point: float, points: list[float], step: float, mass_fin_dir: list[list[float]]) -> float | ResponseCode:
    """
    Функция Гаусса назад/вперёд.

    Берётся от центральной точки и вычисляет полином для точки, находящейся между n/2 и n/2-1 -й.
    Имеется проверка на границы по t, хотя это не нужно в рамках функции check_pos.

    :param point: Точка, значение функции в которой требуется узнать.
    :param points: Сетка значений.
    :param step: Шаг сетки.
    :param mass_fin_dir: Массив конечных значений.

    :return: Значение функции в точке.

    """
    t = (point - points[
        (len(points) - 1) // 2]) / step  # TODO: Вытащить вычисление t в функцию redirector. Хотя, а стоит ли?
    if not -0.5 <= t < 0:
        print("ALARM!", f"{point} IS BROKEN FOR GAUSS_BWD!", f"t = {t}!", sep="\t")
        return ResponseCode.BAD_PARAMETER

    result = 0
    for i in range(len(points)):

        fin_diff_step = mass_fin_dir[i]
        if i % 2 == 0:
            diff_index = (len(fin_diff_step) - 1) // 2
        else:
            """Идти вниз"""
            diff_index = len(fin_diff_step) // 2

        mult = fin_diff_step[diff_index]
        for j in range(0, i - 1 + 1):
            if j % 2 != 0:
                mult *= (t + (j + 1) // 2)
                # mult *= (t - j)
            else:
                mult *= (t - (j + 1) // 2)
                # mult *= (t + j)
                # TODO: Разобраться, какой способ более подходящий
        result += (mult / factorial(i))

    return result


def redirector(point: float, points: list[float], step: float, mass_fin_dir: list[list[float]],
               status: tuple[ResponseCode, float | None]) -> float | None:
    """
    Функция-переадресовщик.

    Переадресует запрос в зависимости от статуса.

    :param point: Точка, значение функции в которой требуется узнать.
    :param points: Сетка значений.
    :param step: Шаг сетки.
    :param mass_fin_dir: Массив конечных разностей
    :param status: Статус выполнения подбора функции.
     В зависимости от статуса происходит переадресация

     :return: Значение полинома для точки point.
      None, если функция не была найдена.

    """
    match (status[0]):
        case ResponseCode.TABLE_POINT:
            return func(point)

        case ResponseCode.NEWTON_BWD:
            return newton_bwd(point, points, step, mass_fin_dir)
        case ResponseCode.NEWTON_FWD:
            return newton_fwd(point, points, step, mass_fin_dir)

        case ResponseCode.GAUSS_BWD:
            return gauss_bwd(point, points, step, mass_fin_dir)
        case ResponseCode.GAUSS_FWD:
            return gauss_fwd(point, points, step, mass_fin_dir)

        case ResponseCode.OVERFLOW_BWD:
            return None
        case ResponseCode.OVERFLOW_FWD:
            return None
        case ResponseCode.NO_FUNC:
            return None


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
        return ResponseCode.NEWTON_FWD, 0
    elif point > points[-2]:
        return ResponseCode.NEWTON_BWD, len(points) - 1

    """Проверка на Гаусса"""
    if len(points) % 2 == 0:
        # TODO: Разобраться, почему Гаусс берётся только от центра, а Ньютон только от краёв
        return ResponseCode.NO_FUNC, None
    index = (len(points) - 1) // 2
    diff_1 = (points[index] + points[index - 1]) / 2
    diff_2 = (points[index + 1] + points[index]) / 2
    if diff_1 <= point < points[index]:
        return ResponseCode.GAUSS_BWD, index
    elif diff_2 >= point > points[index]:
        return ResponseCode.GAUSS_FWD, index

    return ResponseCode.NO_FUNC, None

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


mass_points = [0.1, 0.13, 0.58, 0.37]  # 0.1 тестовая точка
range_graph = (0.1, 0.6)

mass_fin_diff = []
"""Массив конечных разностей"""
mass_grid = []
"""Массив сеток"""

mass_ter_e = []
"""Массив теоретических ошибок"""

MAX_COUNT_POINTS = 13
"""Максимальное количество точек"""
MIN_COUNT_POINTS = 3
"""Минимальное количество точек"""

"""Поскольку по условию сказано взять максимальное количество точек,
 то необходимо сгенерировать конечные разности для каждого случая"""
for index in range(MAX_COUNT_POINTS, MIN_COUNT_POINTS - 1, -1):
    fin_diff = get_fin_diff(func, (linspace(*range_graph, index)).tolist())
    mass_fin_diff.append(fin_diff)
    step_grid = (range_graph[1] - range_graph[0]) / (index - 1)
    """Шаг сетки"""
    x_points = linspace(*range_graph, index).tolist()
    """Точки сетки"""
    mass_ter_e.append(teor_error(index, range_graph, func))
    mass_grid.append((step_grid, x_points))
    pd.DataFrame(fin_diff, index=x_points).to_csv(f"Finite_Differences/FiniteDifference_{index}.csv")

for index, cur_point in enumerate(mass_points):
    print(f"\nПроверка точки {cur_point}")
    for sub_index, grid in enumerate(mass_grid):
        print(f"Теоретическая ошибка минимума: {mass_ter_e[sub_index][0]}",
              f"Теоретическая ошибка максимума: {mass_ter_e[sub_index][1]}", sep="\t")

        step_grid, x_points = grid
        # <=10
        cur_count_points = MAX_COUNT_POINTS - sub_index

        state = check_pos(cur_point, x_points)
        result = redirector(cur_point, x_points, step_grid, mass_fin_diff[sub_index], state)
        if result is None:
            print(f"Невозможно подобрать функцию для точки {cur_point} при количестве точек = {cur_count_points}",
                  f"Ошибка: {state[0].name}", sep="\t")
            continue
        elif isinstance(result, ResponseCode):
            print(f"Ошибка в ходе вычислений для точки {cur_point} при количестве точек = {cur_count_points}!",
                  "Проблема с параметром t!",
                  f"Используемая функция: {state[0].name}", sep="\t")
        else:
            abs_e = result - func(cur_point)
            print(f"Удалось подобрать функцию для точки {cur_point} при количестве точек = {cur_count_points}",
                  f"Функция: {state[0].name}", f"Результат: {result}", sep="\t")
            print("Абсолютная ошибка:", f"{abs_e}", sep="\t")
            print("Попадает ли ошибка в промежуток?", f"{mass_ter_e[sub_index][0] < abs_e < mass_ter_e[sub_index][1]}",
                  sep="\t")
            break
