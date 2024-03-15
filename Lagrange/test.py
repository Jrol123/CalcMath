from numpy import pi, cos


def lagrange(points: list[tuple[float, float]], bp: float) -> float:
    count_points = len(points)
    result = 0
    for k, point in enumerate(points):
        multiply = 1
        for j in range(k - 1):
            x = points[j][0]
            multiply *= ((bp - x) / (point[0] - x))
        for i in range(k + 1, count_points):
            x = points[i][0]
            multiply *= ((bp - x) / (point[0] - x))
        result += multiply * point[1]
    return result


def generate_points(rng: tuple[float, float], count_points: int, function) -> list[tuple[float, float]]:
    step = (rng[1] - rng[0]) / (count_points - 1)
    res = [(rng[0], function(rng[0], count_points))]
    for i in range(1, count_points - 1):
        x = rng[0] + step * i
        res.append((x, function(x, count_points)))
    res.append((rng[1], function(rng[1], count_points)))
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


count_pts = 800

range_graph = (-pi, pi)

points = generate_points(range_graph, count_pts, func)

bp = pi/2

print(f"{bp:.10f}", f"{lagrange(points, bp):.10f}", f"{func(bp, count_pts):.10f}")
