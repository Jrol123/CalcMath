import numpy as np
import pandas as pd
from time import time
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning)


FIRST_SIZE = 2
STEP_SIZE = 8
LAST_SIZE = 256 + 1 + FIRST_SIZE

COUNT_TRIES = 10

MAX_DEF_COUNT_ITER = 100000
"""Максимальное количество итераций по-умолчанию"""
DEF_EPS = 0.0001
"""Точность по-умолчанию"""


df = pd.DataFrame(
    columns=[
        "size",
        "true_value_max",
        "true_value_min",
        "value_max",
        "value_max_time",
        "value_min_def",
        "value_min_def_time"
    ],
    index=range(1, ((LAST_SIZE - FIRST_SIZE) // STEP_SIZE + 1) * COUNT_TRIES + 1),
)
df


# # Подготовка


def calc(A: np.matrix, b: np.matrix) -> float:
    """Вычисление собственного значения
    
    Вычисляется с помощью отношения Рэлея

    Args:
        A (np.matrix): Матрица
        b (np.matrix): Собственный вектор

    Returns:
        float: Собственное значение
    """
    if b is None:
        print("Вектор собственного значения не был посчитан")
        return None
    return float((b.T @ A @ b) / (b.T @ b))


# # Прямой метод


# $x^{(k)} = \dfrac{Ax^{(k - 1)}}{\alpha_{k - 1}}$
# 
# $\lambda_{1}(A) = \dfrac{\left( Ax^{(k)}, x^{(k)} \right)}{(x^{(k)}, x^{(k)})} = \dfrac{(x^{(k)})^{T}A^{(k)}x^{(k)}}{(x^{(k)})^{T}x^{(k)}}$, где $|\lambda_1(A)|$ - наибольшее по модулю СЗ


def power_iteration(
    A: np.matrix,
    epsilon: float = DEF_EPS,
    num_iterations: int = MAX_DEF_COUNT_ITER,
    b_k: np.matrix = None,
    is_numpy: bool = True,
) -> np.matrix:
    """Метод прямых итераций

    Args:
        A (np.matrix): Матрица
        epsilon (float, optional): Точность. Defaults to DEF_EPS.
        num_iterations (int, optional): Количество итераций, если функция настроена работать итеративно. Defaults to MAX_DEF_COUNT_ITER.
        b_k (np.matrix, optional): Начальное приближение. Defaults to None.
        is_numpy (bool, optional): Использовать ли инверсию от numpy. Defaults to True.

    Returns:
        np.matrix: Собственный вектор
    """
    if b_k is None:
        """
        Генерировать ли начальное приближение
        """
        b_k = np.random.rand(A.shape[1], 1)

    alpha_old = None

    while True:
        b_k1 = np.dot(A, b_k)

        b_k1_norm = np.linalg.norm(b_k1)

        b_k = b_k1 / b_k1_norm

        alpha_new = calc(A, b_k)
        if alpha_old is not None and abs(alpha_new - alpha_old) < epsilon:
            break
        alpha_old = alpha_new

    return b_k, alpha_new


# # Обратный метод


def calc_LU(A: np.matrix) -> tuple[np.matrix, np.matrix]:
    """Вычисление LU разложения

    Args:
        original (np.matrix): Исходная матрица

    Returns:
        tuple[np.matrix, np.matrix]: L и U матрица
    """

    matrixU = np.matrix(np.zeros(A.shape))

    matrixU[0] = A[0]


    matrixL = np.diag(np.ones((A.shape[0])))

    for i in range(A.shape[1]):

        matrixL[i, 0] = A[i, 0] / A[0, 0]


    for row in range(1, A.shape[0]):

        for column in range(1, A.shape[1]):

            if row <= column:

                elem_sum = sum(
                    matrixL[row, i] * matrixU[i, column] for i in range(0, row - 1 + 1)
                )


                matrixU[row, column] = A[row, column] - elem_sum

                continue

            elem_sum = sum(
                matrixL[row, i] * matrixU[i, column] for i in range(0, column - 1 + 1)
            )


            matrixL[row, column] = (A[row, column] - elem_sum) / matrixU[column, column]


    return matrixL, matrixU



def get_inverse_LU(matrix: np.matrix) -> np.matrix:
    """Получение обратной матрицы с помощью LU разложения

    Args:
        matrix (np.matrix): Исходная матрица.

    Returns:
        np.matrix: Обратная матрица.
    """

    ML, MU = calc_LU(matrix)

    out_mt = np.zeros(matrix.shape)

    for iter in range(matrix.shape[0]):

        b = np.zeros((matrix.shape[0], 1))

        b[iter] = 1

        y = np.linalg.solve(ML, b)

        x = np.linalg.solve(MU, y)

        x = x.reshape((1, matrix.shape[0]))

        out_mt[:, iter] = x
    return out_mt



def inverse_power_def(
    A: np.matrix,

    epsilon: float = DEF_EPS,
    num_iterations: int = MAX_DEF_COUNT_ITER,
    b_k: np.matrix = None,
    is_numpy: bool = True,
) -> tuple[np.matrix, float]:
    """Метод обратных итераций

    Args:
        A (np.matrix): Начальная матрица
        epsilon (float, optional): Точность. Defaults to DEF_EPS.
        num_iterations (int, optional): Число итераций. Defaults to MAX_DEF_COUNT_ITER.
        b_k (np.matrix, optional): Начальное приближение. Defaults to None.
        is_numpy (bool, optional): Использовать ли инверсию от numpy. Defaults to True.

    Returns:
        tuple[np.matrix, float]: Собственный вектор и собственное значение
    """

    try:

        if is_numpy:

            inv_A = np.linalg.inv(A)
        else:


            inv_A = get_inverse_LU(A)

    except np.linalg.LinAlgError:

        print("Вырожденная матрица")

        return None

    res = power_iteration(
        A=inv_A, epsilon=epsilon, num_iterations=num_iterations, b_k=b_k
    )


    return res[0], 1 / res[1]


A: np.matrix = np.matrix("1 3 -2 0;"
                         "1 1 4 7;"
                         "4 7 11 23;"
                         "52 66 2 0")  # -0.65
eigvec, eigval = inverse_power_def(A)
calc(A, eigvec), eigval


# # Тестирование


# ## Подготовка


def output(
    A: np.matrix,
    funcs_names: list[str],
    funcs: list,
    is_numpy: bool = True,
    num_iterations: int = MAX_DEF_COUNT_ITER,
    b_k: np.matrix = None,
) -> tuple[np.matrix, list[float | None]]:
    """Функция-тестировщик

    Args:
        A (np.matrix): Исходная матрица
        funcs_names (list[str]): Названия функций
        funcs (list): функции
        is_numpy (bool, optional): Использовать ли инверсию от numpy. Defaults to True. Defaults to True.
        num_iterations (int, optional): Количество итераций. Defaults to MAX_DEF_COUNT_ITER.
        b_k (np.matrix, optional): Начальное приближение. Defaults to None.

    Returns:
        tuple[np.matrix, list[float | None]]: Выходные данные. Размер, максимальное СЗ, минимальное СЗ и list[float, float], состоящий из полученного приближения и затраченного времени
    """

    eigs = np.linalg.eigvals(A)
    if any(isinstance(eig, np.complex128) for eig in eigs):
        print(eigs)

    abs_v = np.abs(np.linalg.eigvals(A))

    abs_v_max = np.max(abs_v)

    abs_v_min = np.min(abs_v)

    mass_eval = []

    for func_name, f in zip(funcs_names, funcs):
        start_time = time()
        evec, eval = f(A=A, b_k=b_k, num_iterations=num_iterations, is_numpy=is_numpy)
        stop_time = time()
        mass_eval.append(abs(eval))
        mass_eval.append(stop_time - start_time)

    return A.shape[0], abs_v_max, abs_v_min, *mass_eval


def generate_non_singular(n: int) -> np.matrix:
    """Генерация невырожденной матрицы

    Args:
        n (int): Размерность матрицы

    Returns:
        np.matrix: Невырожденная матрица
    """
    while True:
        matrix = np.random.rand(n, n)
        if np.linalg.det(matrix) != 0:
            return matrix


funcs_names = ["Прямые итерации", "Обратные итерации (обычные)"]
funcs = [power_iteration, inverse_power_def]


# ## Тестирование на заранее заготовленных матрицах


A = np.array([[11, 2, 5], [2, 7, 3], [5, 3, 11]])
size, true_value_max, true_value_min, value_max, _, value_min_def, _ = output(A, funcs_names=funcs_names, funcs=funcs)
true_value_max, value_max, true_value_min, value_min_def


A = np.array([[17, -3, 7], [-3, 22, 8], [7, 8, 11]])
size, true_value_max, true_value_min, value_max, _, value_min_def, _ = output(A, funcs_names=funcs_names, funcs=funcs)
true_value_max, value_max, true_value_min, value_min_def


A = np.array([[1, 2, 4], [4, 5, 6], [7, 8, 9]])
size, true_value_max, true_value_min, value_max, _, value_min_def, _ = output(A, funcs_names=funcs_names, funcs=funcs)
true_value_max, value_max, true_value_min, value_min_def


# ## Тестирование на случайных симметричных матрицах
# 
# Симметричные матрицы необходимы для вещественных корней


for i in range(FIRST_SIZE, LAST_SIZE + 1, STEP_SIZE):
    print(f"Current size:{i}")
    for j in range(COUNT_TRIES):
        print(j + 1)
        matr = generate_non_singular(i)
        matr = matr @ matr.T
        # Пришлось ввести симметричную матрицу из-за комплексных корней
        df.iloc[((i - FIRST_SIZE) // STEP_SIZE) * COUNT_TRIES + j] = output(A=matr, funcs_names=funcs_names, funcs=funcs, is_numpy=False)
    print()

df["delta_max"] = abs(df["true_value_max"] - df["value_max"])
df["delta_min_def"] = abs(df["true_value_min"] - df["value_min_def"])

res_df = df[["size", "value_max_time", "value_min_def_time", "delta_max", "delta_min_def"]]

df


df


df.sort_values("delta_max", ascending=False).head(10)


df.sort_values("delta_min_def", ascending=False).head(10)


max(res_df["delta_min_def"])


mean_df = res_df.groupby("size").mean()
mean_df


max(mean_df['delta_min_def'])


# # Графическая часть


import matplotlib.pyplot as plt


figsize = (18, 6)
b_k = range(FIRST_SIZE, LAST_SIZE + 1, STEP_SIZE * 1)
x_lt = np.linspace(FIRST_SIZE, LAST_SIZE)
res_df


plt.figure(figsize=figsize)

fig, ax1 = plt.subplots(figsize=figsize)
ax1.set_xticks(ticks=b_k)
ax1.set_xlabel('Размер матрицы')

plt.plot(mean_df["delta_min_def"], label="Разница с numpy минимальным значением")
plt.legend()

plt.savefig('img/diff_min.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=figsize)

fig, ax1 = plt.subplots(figsize=figsize)
ax1.set_xticks(ticks=b_k)
ax1.set_xlabel('Размер матрицы')

plt.plot(mean_df["delta_max"], label="Разница с numpy максимальным значением")
plt.legend()

plt.savefig('img/diff_max.png', dpi=300, bbox_inches='tight')
plt.show()


# Create a single figure with two subplots (one on top of the other)
fig, (ax_min, ax_max) = plt.subplots(2, 1, figsize=figsize)

# Plot for minimal values in the top half
ax_min.set_xticks(b_k)
ax_min.set_xlabel("Размер матрицы")
ax_min.plot(
    mean_df["delta_min_def"],
    label="Разница с numpy минимальным значением",
    marker="o",
    markersize=4,
    color="blue",
)
ax_min.legend()
ax_min.set_title("Минимальные значения")

# Plot for maximal values in the bottom half
ax_max.set_xticks(b_k)
ax_max.set_xlabel("Размер матрицы")
ax_max.plot(
    mean_df["delta_max"],
    label="Разница с numpy максимальным значением",
    marker="o",
    markersize=4,
    color="red",
)
ax_max.legend()
ax_max.set_title("Максимальные значения")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined figure
plt.savefig("img/diff_combined.png", dpi=300, bbox_inches="tight")

# Show the combined figure
plt.show()


plt.figure(figsize=figsize)

fig, ax1 = plt.subplots(figsize=figsize)

ax1.plot(
    mean_df["value_max_time"],
    marker="o",
    markersize=4,
    label="График времени выполнения для максимума",
)
ax1.set_xlabel("Размер матрицы")
ax1.set_xticks(b_k)

ax2 = ax1.twinx()

ax2.plot(
    x_lt,
    (x_lt**2) / 1e6,
    color="green",
    linestyle="--",
    label="График асимптотики алгоритма для максимума",
)

ax1.legend(loc="upper left")
ax2.legend(loc="lower left")

plt.savefig('img/time_max.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=figsize)

fig, ax1 = plt.subplots(figsize=figsize)

ax1.plot(
    mean_df["value_min_def_time"],
    marker="o",
    markersize=4,
    label="График времени выполнения для минимума",
)
ax1.set_xlabel("Размер матрицы")
ax1.set_xticks(b_k)

ax2 = ax1.twinx()

ax2.plot(
    x_lt,
    (x_lt**3) / 1e6,
    color="green",
    linestyle="--",
    label="График асимптотики алгоритма для минимума",
)


ax1.legend(loc="upper left")
ax2.legend(loc="lower left")

plt.savefig('img/time_min.png', dpi=300, bbox_inches='tight')
plt.show()


# Assuming b_k, mean_df, and x_lt are already defined.
plt.figure(figsize=figsize)

# Create the first subplot for time execution
ax1 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
ax1.plot(
    mean_df["value_min_def_time"],
    marker="o",
    markersize=4,
    label="График времени выполнения для минимума",
)
ax1.set_xlabel("Размер матрицы")
ax1.set_xticks(b_k)
ax1.set_title("Время выполнения")
ax1.legend(loc="upper left")

ax2_u = ax1.twinx()

ax2_u.plot(
    x_lt,
    (x_lt**3) / 1e6,
    color="green",
    linestyle="--",
    label="График асимптотики алгоритма для минимума",
)
ax2_u.legend(loc="lower left")


# Create the second subplot for the difference calculation
ax2 = plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
ax2.plot(
    mean_df["delta_min_def"],
    label="Разница с numpy минимальным значением"
)
ax2.set_xlabel("Размер матрицы")
ax2.set_xticks(b_k)
ax2.set_title("Разница с numpy")
ax2.legend(loc="upper left")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined figure with the two plots
plt.savefig('img/time_min_combined.png', dpi=300, bbox_inches='tight')
plt.show()


# Assuming b_k, mean_df, and x_lt are already defined.
plt.figure(figsize=figsize)

# Create the first subplot for time execution
ax1 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
ax1.plot(
    mean_df["value_max_time"],
    marker="o",
    markersize=4,
    label="График времени выполнения для максимума",
)
ax1.set_xlabel("Размер матрицы")
ax1.set_xticks(b_k)
ax1.set_title("Время выполнения")
ax1.legend(loc="upper left")

ax2_u = ax1.twinx()

ax2_u.plot(
    x_lt,
    (x_lt**3) / 1e6,
    color="green",
    linestyle="--",
    label="График асимптотики алгоритма для максимума",
)
ax2_u.legend(loc="lower left")


# Create the second subplot for the difference calculation
ax2 = plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
ax2.plot(
    mean_df["delta_max"],
    label="Разница с numpy максимальным значением"
)
ax2.set_xlabel("Размер матрицы")
ax2.set_xticks(b_k)
ax2.set_title("Разница с numpy")
ax2.legend(loc="upper left")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined figure with the two plots
plt.savefig('img/time_max_combined.png', dpi=300, bbox_inches='tight')
plt.show()


# # Дополнительная часть


# ## Вычисление


ddf = pd.DataFrame(
    columns=[
        "size",
        "true_value_max",
        "true_value_min",
        "value_min_def",
        "value_min_def_time"
    ],
    index=range(1, ((LAST_SIZE - FIRST_SIZE) // STEP_SIZE + 1) * COUNT_TRIES + 1),
)
ddf


ffuncs_names = ["Обратные итерации (обычные)"]
ffuncs = [inverse_power_def]


for i in range(FIRST_SIZE, LAST_SIZE + 1, STEP_SIZE):
    print(f"Current size:{i}")
    for j in range(COUNT_TRIES):
        print(j + 1)
        matr = generate_non_singular(i)
        matr = matr @ matr.T
        # Пришлось ввести симметричную матрицу из-за комплексных корней
        ddf.iloc[((i - FIRST_SIZE) // STEP_SIZE) * COUNT_TRIES + j] = output(A=matr, funcs_names=ffuncs_names, funcs=ffuncs, is_numpy=True)
    print()

ddf["delta_min_def"] = abs(ddf["true_value_min"] - ddf["value_min_def"])

ddf


res_ddf = ddf[["size", "value_min_def_time", "delta_min_def"]]

mean_ddf = res_ddf.groupby("size").mean()
mean_ddf


# ## Графики


plt.figure(figsize=figsize)

fig, ax1 = plt.subplots(figsize=figsize)
ax1.set_xticks(ticks=b_k)
ax1.set_xlabel('Размер матрицы')

plt.plot(mean_ddf["delta_min_def"], label="Разница с numpy минимальным значением")
plt.legend()

plt.savefig('img/diff_min_numpy.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=figsize)

fig, ax1 = plt.subplots(figsize=figsize)

ax1.plot(
    mean_ddf["value_min_def_time"],
    marker="o",
    markersize=4,
    label="График времени выполнения для минимума",
)
ax1.set_xlabel("Размер матрицы")
ax1.set_xticks(b_k)

ax2 = ax1.twinx()

ax2.plot(
    x_lt,
    (x_lt**3) / 1e6,
    color="green",
    linestyle="--",
    label="График асимптотики алгоритма для минимума",
)


ax1.legend(loc="upper left")
ax2.legend(loc="lower left")

plt.savefig('img/time_min_numpy.png', dpi=300, bbox_inches='tight')
plt.show()


# Assuming b_k, mean_df, and x_lt are already defined.
plt.figure(figsize=figsize)

# Create the first subplot for time execution
ax1 = plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
ax1.plot(
    mean_ddf["value_min_def_time"],
    marker="o",
    markersize=4,
    label="График времени выполнения для минимума",
)
ax1.set_xlabel("Размер матрицы")
ax1.set_xticks(b_k)
ax1.set_title("Время выполнения")
ax1.legend(loc="upper left")

ax2_u = ax1.twinx()

ax2_u.plot(
    x_lt,
    (x_lt**3) / 1e6,
    color="green",
    linestyle="--",
    label="График асимптотики алгоритма для минимума",
)
ax2_u.legend(loc="lower left")


# Create the second subplot for the difference calculation
ax2 = plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
ax2.plot(
    mean_ddf["delta_min_def"],
    label="Разница с numpy минимальным значением"
)
ax2.set_xlabel("Размер матрицы")
ax2.set_xticks(b_k)
ax2.set_title("Разница с numpy")
ax2.legend(loc="upper left")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined figure with the two plots
plt.savefig('img/time_min_combined_numpy.png', dpi=300, bbox_inches='tight')
plt.show()
