from math import factorial
from numpy import cos, sin, pi, linspace


def lagrange(num_point: int, points: list[tuple[float, float]], step: float) -> float:
    """
    Неньютоновская реализация производной от полинома Лагранжа первой степени

    :param points: Список точек
    :type points: list[tuple[float, float]]
    :param num_point: Номер точки
    :type num_point: int
    :param step: Шаг сетки
    :type step: float

    :return: Значение полинома Лагранжа в точке bp
    :rtype: float

    """
    count_points = len(points) - 1
    result = 0
    for i, point in enumerate(points):
        point_mult = point[1]

        def diff_mult_part(a: int, b: int) -> float:
            sub_mult = 1
            for j in range(a, b):
                sub_mult *= (i - j)
            return sub_mult

        diff_mult = diff_mult_part(0, i - 1 + 1) * diff_mult_part(i + 1, count_points + 1)

        def grid_mult_part(a: int, b: int) -> float:
            alt_mult = 0
            for j in range(a, b):
                sub_mult = 1
                for j1 in range(0, min(i, j) - 1 + 1):
                    sub_mult *= (num_point - j1)
                for j1 in range(min(i, j) + 1, max(i, j) - 1 + 1):
                    sub_mult *= (num_point - j1)
                for j1 in range(max(i, j) + 1, count_points + 1):
                    sub_mult *= (num_point - j1)
                alt_mult += sub_mult
            return alt_mult

        grid_mult = grid_mult_part(0, i - 1 + 1) + grid_mult_part(i + 1, count_points + 1)

        result += point_mult / diff_mult * grid_mult

    return result / step


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


def teor_error(num_point: int, step: float,
               count_points: int, rng: tuple[float, float], function) -> tuple[float, float]:
    """
    Получение первой производной от теоретической ошибки.
    Теоретическая ошибка берётся от максимума и минимума функции.

    :param function: Функция
    :type function: function
    :param num_point: Номер точки
    :type num_point: int
    :param step: Шаг сетки
    :type step: float
    :param count_points: Количество точек
    :param rng: Кортеж границ

    :return: Теоретическая ошибка
    :rtype: float

    """
    pts = function(linspace(*rng, num=10 ** 3), count_points + 1)
    fct = factorial(count_points + 1)

    result: tuple[float, float] = min(pts), max(pts)

    def everything_else_done(value: float) -> float:
        sub_res = 0
        for j in range(count_points + 1):
            sub_mult = 1
            for j1 in range(count_points + 1):
                if j1 != j:
                    sub_mult *= (num_point - j1)
            sub_res += sub_mult
        return value * sub_res * step ** count_points / fct

    return map(everything_else_done, result)


if __name__ == "__main__":
    koef_round = 10
    k, count_pts, index_point = 1, 5, 5
    range_graph = (0.1, 0.6)
    step_grid = (range_graph[1] - range_graph[0]) / count_pts
    mass_points = [(pt, func(pt)) for pt in linspace(*range_graph, count_pts + 1)]

    res_lagrange = lagrange(index_point, mass_points, step_grid)
    res_func = func(mass_points[index_point][0], k)

    min_ter, max_ter = map(lambda num: round(num, koef_round), teor_error(index_point, step_grid, count_pts, range_graph, func))

    print(f"Лагранж:\t{round(res_lagrange, koef_round)}",
          f"Значение производной функции:\t{round(res_func, koef_round)}",
          f"Разница:\t{round(abs(res_lagrange - res_func), koef_round)}",
          sep='\n')
    print()
    print(f"Минимальная ошибка:\t{min_ter}",
          f"Максимальная ошибка:\t{max_ter}",
          sep='\n')
    print()
    print(f"Попадает ли ошибка в промежуток?:\t{min_ter < abs(res_lagrange - res_func) < max_ter}",
          sep='\n')
