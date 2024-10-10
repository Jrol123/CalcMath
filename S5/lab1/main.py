import numpy as np

SIZE = 25

mass_matrix = [np.random.rand(SIZE,SIZE) for i in range(5)]

print(*mass_matrix)