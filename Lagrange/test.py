import pandas as pd
import sympy


def lagrange(points: pd.DataFrame, bp: float) -> float:
    count_points = len(points['x'])
    result = 0
    for k in range(count_points):
        point = points.iloc[[k]]
        x = point['x'][k]
        y = point['y'][k]
        point = [x, y]
        multiply = 1
        for j in range(k - 1):
            sub_point = points.iloc[[j]]
            x = sub_point['x'][j]
            multiply *= ((bp - x) / (point[0] - x))
        for i in range(k + 1, count_points):
            sub_point = points.iloc[[i]]
            x = sub_point['x'][i]
            multiply *= ((bp - x) / (point[0] - x))
        result += multiply * point[1]
    return result


def f1(x: float) -> float:
    return abs(x)


df = pd.read_csv("points.csv")

print(lagrange(df, -1, f1), f1(-1))
