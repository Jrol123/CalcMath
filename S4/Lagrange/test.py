from numpy import cos, linspace, pi
from math import factorial
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
    res = []
    for pos in linspace(*rng, count_points):
        res.append((pos, function(pos)))
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
    Вторая-n производная функции y = 2x− cos(x)

    От второй производной далее, по тригонометрическим свойствам, можно считать n-ю производную

    :param x: Точка x
    :param n: Степень производной. Считается от 2 для удобства.
        + Потому что именно на 2-й производной убирается константа
    :return: Значение функции в точке x

    :rtype: float

    """
    return cos(x + ((n - 2) * pi) / 2)


def get_norm(function, rng: tuple[float, float], *args) -> float:
    """
    Получение нормы функции

    :param function: Некоторая функция, норму которой мы хотим получить
    :param rng: Кортеж границ
    :param args: Дополнительные аргументы функции

    :return: Норма функции
    :rtype: float

    """
    return max(abs(function(linspace(*rng, num=10**3), *args)))


def rel_error(abs_er: float, norm_f: float) -> float:
    """
    Получение относительной ошибки

    :param abs_er: Абсолютная ошибка
    :param norm_f: Норма функции

    :return: Относительная ошибка
    :rtype: float

    """
    return (abs_er / norm_f) * 100


def teor_error(count_points: int, rng: tuple[float, float]) -> float:
    """
    Получение теоретической ошибки

    :param count_points: Количество точек
    :param rng: Кортеж границ

    :return: Теоретическая ошибка
    :rtype: float

    """
    return (get_norm(derv_func, rng, count_points + 1) / factorial(count_points + 1)) * (
            (rng[1] - rng[0]) ** (count_points + 1))


rng_pos = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
abs_e_mass = []
rel_e_mass = []
ter_e_mass = []
print("n", "", "Абсолютная норма", "Относительная норма", "Теоретическая норма", sep='\t')
for count_pts in rng_pos:
    range_graph = (0.1, 0.6)

    full_points = generate_points(range_graph, count_pts, func)

    norm = get_norm(func, range_graph)
    lag_norm = get_norm(lagrange, range_graph, full_points)
    # lag_norm = max(abs(lagrange(linspace(*range_graph), full_points)))
    # print(sub_lag_norm, lag_norm)
    abs_e = max(abs(lagrange(linspace(*range_graph, num=10**3), full_points) - func(linspace(*range_graph, num=10**3))))
    # der_e = max(abs(derv_func(linspace(*range_graph))))
    der_e = get_norm(derv_func, range_graph)
    rel_e = rel_error(abs_e, lag_norm)
    ter_e = teor_error(count_pts, range_graph)

    abs_e_mass.append(abs_e)
    rel_e_mass.append(rel_e)
    ter_e_mass.append(ter_e)

    print(count_pts, "", abs_e, rel_e, ter_e, sep='\t')

abs_df = pd.DataFrame(abs_e_mass, rng_pos, columns=["Value"])
rel_df = pd.DataFrame(rel_e_mass, rng_pos, columns=["Value"])
ter_df = pd.DataFrame(ter_e_mass, rng_pos, columns=["Value"])
