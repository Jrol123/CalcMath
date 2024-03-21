from numpy import cos, linspace, pi
from math import factorial
import matplotlib.pyplot as plt


def lagrange(points: list[tuple[float, float]], bp: float) -> float:
    """
    Неньютоновская реализация полинома Лагранжа

    :param points: Список точек
    :type points: list[tuple[float, float]]
    :param bp: Точка, значение функции в которой нужно получить
    :type bp: float

    :return: Значение полинома Лагранжа в точке bp
    :rtype: float

    """
    count_points = len(points)
    result = 0
    for k, point in enumerate(points):
        multiply = point[1]
        for j in range(0, k - 1 + 1):
            x = points[j][0]
            multiply *= ((bp - x) / (point[0] - x))
        for i in range(k + 1, count_points):
            x = points[i][0]
            multiply *= ((bp - x) / (point[0] - x))
        result += multiply
    return result


def generate_points(rng: tuple[float, float], count_points: int, function) -> list[tuple[float, float]]:
    """
     Генерирование точек с некоторым постоянным шагом

    :param rng: Кортеж границ
    :type rng: tuple[float, float]
    :param count_points: Количество точек, которое нужно сгенерировать
    :type count_points: int
    :param function: Функция
    :type function: function

    :return: Список точек
    :rtype: list[tuple[float, float]]

    """
    step = (rng[1] - rng[0]) / (count_points - 1)
    res = [(rng[0], function(rng[0]))]
    for i in range(1, count_points - 1):
        x = rng[0] + step * i
        res.append((x, function(x)))
    res.append((rng[1], function(rng[1])))
    return res


# В = 15

def func(x: float):
    """
    Функция y = 2x− cos(x)

    :param x: Точка x
    :return: Значение функции в точке x

    :rtype: float

    """
    return 2 * x - cos(x)


def derv_func(x: float, n: int = 2) -> float:
    """
    Вторая производная функции y = 2x− cos(x)

    От второй производной далее, по тригонометрическим свойствам, можно считать n-ю производную

    :param x: Точка x
    :param n: Степень производной. Считается от 2 для удобства.
    :return: Значение функции в точке x

    :rtype: float

    """
    return cos(x + ((n - 2) * pi) / 2)


def get_norm(function, rng: tuple[float, float], *args) -> float:
    return max(abs(function(linspace(*rng), *args)))


def abs_error(norm: float, lagr: float) -> float:
    return abs(norm - lagr)


def rel_error(abs_er: float, norm: float) -> float:
    return (abs_er / norm) * 100


def teor_error(count_pts: int, rng: tuple[float, float], der_norm: float):
    return (get_norm(derv_func, rng, count_pts + 1) / factorial(count_pts + 1)) * ((rng[1] - rng[0])**(count_pts + 1))


count_pts = 20
"""
Количество точек, которое нужно сгенерировать.

Должно быть >= 2, так как по условию берутся граничные точки тоже.
"""

range_graph = (-30.0, 30.0)

full_points = generate_points(range_graph, count_pts, func)

keypoint = 0.0

# print(f"{keypoint:.10f}", f"{lagrange(full_points, keypoint):.10f}", f"{func(keypoint):.10f}")

norm = get_norm(func, range_graph)
lag_norm = max(abs(lagrange(full_points, linspace(*range_graph))))
abs_e = max(abs(lagrange(full_points, linspace(*range_graph)) - func(linspace(*range_graph))))
der_e = max(abs(derv_func(linspace(*range_graph))))
rel_e = rel_error(abs_e, lag_norm)
ter_e = teor_error(count_pts, range_graph, der_e)

print(norm, lag_norm, der_e, abs_e, rel_e, ter_e)

# TODO: Сделать цикл для построения графика
# TODO: Сделать вывод (q)

plt.plot(linspace(range_graph))
