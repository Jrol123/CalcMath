from math import log, factorial, prod
from numpy import linspace


def lagrange(m, f_n, interval=(0.5, 1)):
    l_n = 0
    n = len(f_n)
    a, b = interval
    h = (b - a) / (n - 1)
    for i in range(n):
        s1 = 0
        for j1 in range(n - 1):
            s2 = 0
            for j2 in range(j1 + 1, n):
                p = 1
                for j in range(n):
                    if j != i and j != j1 and j != j2 and j1 != i and j2 != i:
                        p *= (m - j)
                s2 += p
            s1 += s2

        for j in range(n):
            if j != i:
                s1 /= (i - j)

        l_n += f_n[i][1] * s1
    l_n = l_n * 2 / h ** 2
    return l_n


def R_n(m, n, f, interval=(0.5, 1)):
    a, b = interval
    h = (b - a) / n

    M_min_prev = min(f(a + (b - a) * i / 1000, n + 1) for i in range(1001))
    M_max_prev = max(f(a + (b - a) * i / 1000, n + 1) for i in range(1001))

    M_min = min(f(a + (b - a) * i / 1000, n + 2) for i in range(1001))
    M_max = max(f(a + (b - a) * i / 1000, n + 2) for i in range(1001))

    def result_giver(M, M_prev):
        return 2 * M * sum(
            prod(m - j for j in range(n + 1) if j != j1)
            for j1 in range(n + 2)) \
            * h ** n / factorial(n + 2) \
            + M_prev * sum(
                sum(
                    prod(m - j for j in range(n + 1) if j != j1 and j != j2)
                    for j2 in range(j1 + 1, n + 1))
                for j1 in range(n)) \
            * h ** (n + 1) / factorial(n + 1)

    return result_giver(M_min, M_min_prev), result_giver(M_max, M_max_prev)

def f(x):
    return x ** 2 + log(x + 5)


def f_dir(x, k):
    if k == 0:
        return f(x)
    elif k == 1:
        return 2 * x + 1 / (x + 5)
    elif k == 2:
        return 2 - 1 / pow(x + 5, 2)
    else:
        return pow(-1, k + 1) * factorial(k - 1) / pow(x + 5, k)



range_graph = [0.5, 1]
n = 3
m = 0

flist = [(pt, f_dir(pt, 0)) for pt in linspace(*range_graph, n + 1)]

res_lagrange = lagrange(m, flist, range_graph)
res_func = f_dir(flist[m][0], 2)

min_ter, max_ter = R_n(m, n, f_dir, range_graph)

print(f"Лагранж:\t{res_lagrange}",
      f"Значение производной функции:\t{res_func}",
      f"Разница:\t{abs(res_lagrange - res_func)}",
      sep='\n')
print()
print(f"Минимальная ошибка:\t{min_ter}",
      f"Максимальная ошибка:\t{max_ter}",
      sep='\n')
print()
print(f"Попадает ли ошибка в промежуток?:\t{min_ter < abs(res_lagrange - res_func) < max_ter}",
      sep='\n')

