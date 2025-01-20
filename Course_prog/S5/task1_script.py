import numpy as np
import pandas as pd

SIZE = 100
ITERS = 100


def M(mat):
    return mat.shape[0] * np.max(np.abs(mat))


def test_M(m_A, m_B):
    M_A = M(m_A)
    M_B = M(m_B)
    M_AB = M(m_A @ m_B)
    
    return M_A, M_B, M_A * M_B, M_AB, M_AB <= M_A * M_B
    
    print(f"M(A): {M_A}, M(B): {M_B}, M(AB): {M_AB}")
    print(f"M(AB) <= M(A) * M(B): {M_AB <= M_A * M_B}")


df = pd.DataFrame(columns=["M(A)", "M(B)", "M(A) * M(B)", "M(AB)", "M(AB) <= M(A) * M(B)"], index=range(ITERS))

df


for index in range(ITERS):
    A = np.random.rand(SIZE, SIZE)
    B = np.random.rand(SIZE, SIZE)
    
    df.loc[index] = test_M(A, B)

df


df[df["M(AB) <= M(A) * M(B)"] == False].count()
# Везде неравенство выполняется


import matplotlib.pyplot as plt


# График с двумя осями
plt.figure(figsize=(18, 10))

fig, ax1 = plt.subplots(figsize=(18, 10))

ax1.plot(df["M(A) * M(B)"], label="M(A) * M(B)", color="orange", marker="o")
ax2 = ax1.twinx()
ax2.plot(df["M(AB)"], label="M(AB)", marker="o")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")


plt.show()


# График с одной осью
plt.figure(figsize=(18, 10))

plt.plot(df["M(A) * M(B)"], label="M(A) * M(B)", marker="o")
plt.plot(df["M(AB)"], label="M(AB)", marker="o")
plt.legend()

plt.savefig('img/task1.png', dpi=300, bbox_inches='tight')
