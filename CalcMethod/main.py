"""
Поиск корней необходимо осуществлять с помощью трех методов: хорд, касательных, дихотомии.
Алгоритм пристрелки для каждого метода должен быть одинаков.
Для каждого из методов измерить время решения задачи.
Сделать вывод.
"""
from numpy import exp, log, sign
from datetime import datetime, timedelta


class OutputEntity:
    def __init__(self, name_function: str, true_result: float, result: float, count_iterations: int, precision: float,
                 time: timedelta, mass_results: list[float]):
        self.name_function = name_function
        self.true_result = true_result
        self.result = result
        self.count_iterations = count_iterations
        self.precision = precision
        self.time = time
        self.results = mass_results

    def __str__(self):
        return (
            f"\nФункция:\t{self.name_function}\n"
            f"Desmos результат:\t{self.true_result}\nПолученный результат:\t{self.result}\n"
            f"Разница:\t{abs(self.true_result - self.result)}\n"
            f"Количество итераций:\t{self.count_iterations}\nИспользованная точность:\t{self.precision}\n"
            f"Затраченное время:\t{self.time}\n")


def dichotomy(f, x_mass: list[float, float, float]) -> list[float, float, float]:
    mid = (x_mass[1] + x_mass[0]) / 2
    x_mass[2] = mid
    if f(mid) * f(x_mass[0]) < 0:
        x_mass[1] = mid
    else:
        x_mass[0] = mid
    return x_mass


def secant(f, x_mass: tuple[float, float, float]) -> float:
    """
    Метод хорд

    :param f: Функция
    :type f: function
    :param x_mass: Массив значений x, состоящий из левого края [0], правого края [1], и последнего значения [-1]

    :return: Значение x

    """
    x_prev = x_mass[-1]
    x_main = x_mass[0]
    return x_main - (f(x_main) * (x_prev - x_main)) / (f(x_prev) - f(x_main))


def newton(f, x_mass: tuple[float, float, float]) -> float:
    """

    :param f: Функция
    :type f: function
    :param x_mass: Массив значений x. Состоит из левого края [0], правого края [1], и последнего значения [-1]

    :return: Значение x

    """
    if x_mass[-1] == x_mass[1]:
        if sign(f(x_mass[1])) == sign(f(x_mass[1], 2)):
            x_prev = x_mass[1]
        else:
            x_prev = x_mass[0]
    else:
        x_prev = x_mass[-1]
    return x_prev - f(x_prev) / f(x_prev, 1)


def get_result(function, f, real_res, mass_borders, eps) -> OutputEntity:
    """
    Получение результатов из функций

    :param function:
    :type function: function
    :param mass_borders:
    :param eps: Точность измерений

    :return: Различные параметры, полученные при вычислении корня.

    """
    mass_val = [mass_borders[0], mass_borders[1], mass_borders[-1]]
    end_mass = []
    delta = abs(mass_borders[0] - mass_borders[1])
    i = 0
    date_start = datetime.now()
    while delta > eps:
        x_cur = function(f, mass_val)
        i += 1
        # print(i := i + 1, mass_val[-1], x_cur)
        delta = abs(x_cur - mass_val[-1])
        mass_val[-1] = x_cur
        end_mass.append(x_cur)
    date_end = datetime.now()
    # print((date_end - date_start))
    return OutputEntity(function.__name__, real_res, mass_val[-1], i, eps, (date_end - date_start), end_mass)


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
        return 1 / (x + 2) - 1
    elif deriv_s == 2:
        return -1 / (x ** 2 + 4 * x + 4)
    return log(x + 2) - x


if __name__ == '__main__':
    # Start here
    range_mass = [1, 1.5]  # Массив границ
    eps = 1e-10
    # 1.1462
    #-1.8414
    real_result = 1.1462
    sec_res = get_result(secant, func, real_result, range_mass, eps)
    new_res = get_result(newton, func, real_result, range_mass, eps)

    delta = abs(range_mass[0] - range_mass[1])
    mass_val = [range_mass[0], range_mass[1], range_mass[-1]]
    mass_res = [(range_mass[1] + range_mass[0]) / 2]
    prev_stat = 1
    i = 0
    date_start = datetime.now()
    while delta > eps:
        mass_val = dichotomy(func, mass_val)
        delta = abs(mass_val[-1] - prev_stat)
        print(i := i + 1, mass_val, delta)
        prev_stat = mass_val[-1]
        if func(prev_stat) == 0:
            break
    date_end = datetime.now()
    dich_res = OutputEntity(dichotomy.__name__, real_result, prev_stat, i, eps, (date_end - date_start), mass_val)

    print(sec_res, new_res, dich_res)
    print(new_res.results)
