import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. Функції згідно з пунктом 2 завдання
# ==========================================

def read_data(matrix_file, vector_file):
    """Зчитування матриці А та вектора В """
    A = np.loadtxt(matrix_file)
    B = np.loadtxt(vector_file)
    return A, B


def lu_decomposition(A):
    """Знаходження LU-розкладу (формули 12) [cite: 5, 12]"""
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i][i] = 1.0  # Діагональні елементи U рівні 1 [cite: 5, 10]

    for k in range(n):
        # Обчислення елементів L [cite: 12, 13]
        for i in range(k, n):
            sum_l = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_l

        # Обчислення елементів U [cite: 12, 14]
        for i in range(k + 1, n):
            sum_u = sum(L[k][j] * U[j][i] for j in range(k))
            U[k][i] = (A[k][i] - sum_u) / L[k][k]

    return L, U


def solve_lu(L, U, B):
    """Розв'язок систем LZ=B та UX=Z [cite: 16, 17]"""
    n = len(L)
    # Прямий хід: LZ = B
    Z = np.zeros(n)
    for k in range(n):
        sum_z = sum(L[k][j] * Z[j] for j in range(k))
        Z[k] = (B[k] - sum_z) / L[k][k]

    # Зворотний хід: UX = Z
    X = np.zeros(n)
    for k in range(n - 1, -1, -1):
        sum_x = sum(U[k][j] * X[j] for j in range(k + 1, n))
        X[k] = Z[k] - sum_x
    return X


def mat_vec_mult(A, X):
    """Обчислення добутку матриці на вектор [cite: 41, 43]"""
    n = len(A)
    return np.array([sum(A[i][j] * X[j] for j in range(n)) for i in range(n)])


def vector_norm(V):
    """Обчислення норми вектора (max елемент) [cite: 43, 46]"""
    return np.max(np.abs(V))


# ==========================================
# 2. Основний алгоритм виконання [cite: 37]
# ==========================================

def main():
    n = 100
    eps_target = 1e-14  # Задана точність [cite: 48]

    # КРОК 1. Генерація даних [cite: 38, 39]
    # Створюємо випадкову матрицю та додаємо діагональне переважання для стабільності
    A_gen = np.random.rand(n, n)
    for i in range(n):
        A_gen[i][i] += n  # Важливо для збіжності!

    # Точний розв'язок (x_j = 2.5) [cite: 39]
    X_exact = np.full(n, 2.5)
    B_gen = mat_vec_mult(A_gen, X_exact)

    # Запис у файли [cite: 38, 42]
    np.savetxt("matrix_A.txt", A_gen)
    np.savetxt("vector_B.txt", B_gen)

    # КРОК 2-3. Зчитування та LU-розклад [cite: 43, 44]
    A, B = read_data("matrix_A.txt", "vector_B.txt")
    L, U = lu_decomposition(A)
    np.savetxt("matrix_LU.txt", np.vstack((L, U)))  # Збереження LU

    # Початковий розв'язок
    X_0 = solve_lu(L, U, B)

    # КРОК 4. Початкова похибка [cite: 45, 46]
    initial_error = vector_norm(mat_vec_mult(A, X_0) - B)
    print(f"Початкова похибка нев'язки: {initial_error:.4e}")

    # КРОК 5. Ітераційне уточнення [cite: 47, 48]
    X = np.copy(X_0)
    history = []

    for i in range(1, 21):  # Максимум 20 ітерацій
        # Вектор нев'язки R = B - AX [cite: 23]
        R = B - mat_vec_mult(A, X)

        # Розв'язок для похибки A * dX = R [cite: 29]
        dX = solve_lu(L, U, R)

        # Уточнення розв'язку [cite: 30]
        X = X + dX

        norm_dX = vector_norm(dX)
        history.append(norm_dX)

        print(f"Ітерація {i}: ||dX|| = {norm_dX:.4e}")

        if norm_dX < eps_target:  # Перевірка умови закінчення [cite: 31, 48]
            print(f"--- Точність {eps_target} досягнута за {i} ітерацій ---")
            break

    # Остаточна перевірка
    final_diff = vector_norm(X - X_exact)
    print(f"Кінцеве відхилення від 2.5: {final_diff:.4e}")

    # БОНУС: Графік збіжності
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(history) + 1), history, 'r-o', label='||ΔX||')
    plt.axhline(y=eps_target, color='blue', linestyle='--', label='Target EPS (1e-14)')
    plt.title('Графік збіжності ітераційного уточнення')
    plt.xlabel('Номер ітерації')
    plt.ylabel('Норма похибки (log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
