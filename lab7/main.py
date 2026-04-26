import random
import matplotlib.pyplot as plt


# =====================================================================
# КРОК 1: Генерація даних та запис у файли
# =====================================================================
def generate_data(n=100, x_val=2.5):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_sum = 0
        for j in range(n):
            if i != j:
                A[i][j] = random.uniform(-10, 10)
                row_sum += abs(A[i][j])
        A[i][i] = row_sum + random.uniform(1, 10)

    X_true = [x_val] * n
    B = [sum(A[i][j] * X_true[j] for j in range(n)) for i in range(n)]

    with open("matrix_A.txt", "w") as f_a:
        for row in A:
            f_a.write(" ".join(map(str, row)) + "\n")

    with open("vector_B.txt", "w") as f_b:
        f_b.write(" ".join(map(str, B)) + "\n")


# =====================================================================
# КРОК 2: Допоміжні функції
# =====================================================================
def read_matrix(filename):
    with open(filename, "r") as f:
        return [[float(x) for x in line.split()] for line in f]


def read_vector(filename):
    with open(filename, "r") as f:
        return [float(x) for x in f.read().split()]


def mat_vec_mult(A, x):
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        result[i] = sum(A[i][j] * x[j] for j in range(n))
    return result


def vec_norm(x):
    return max(abs(val) for val in x)


def mat_norm(A):
    return max(sum(abs(val) for val in row) for row in A)


def vec_diff(x1, x2):
    return [a - b for a, b in zip(x1, x2)]


# =====================================================================
# ІТЕРАЦІЙНІ МЕТОДИ (тепер повертають ще й масив похибок)
# =====================================================================

def simple_iteration(A, B, x0, eps):
    n = len(A)
    norm_A = mat_norm(A)
    tau = 1.0 / norm_A

    x_k = x0[:]
    iterations = 0
    errors = []  # Масив для збереження похибки на кожному кроці

    while True:
        iterations += 1
        x_k_next = [0.0] * n
        AX = mat_vec_mult(A, x_k)
        for i in range(n):
            x_k_next[i] = x_k[i] - tau * (AX[i] - B[i])

        current_error = vec_norm(vec_diff(x_k_next, x_k))
        errors.append(current_error)

        if current_error < eps:
            break
        x_k = x_k_next

        if iterations > 50000:
            print("Метод простої ітерації не зійшовся за 50000 ітерацій!")
            break

    return x_k_next, iterations, errors


def jacobi(A, B, x0, eps):
    n = len(A)
    x_k = x0[:]
    iterations = 0
    errors = []

    while True:
        iterations += 1
        x_k_next = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x_k[j] for j in range(n) if j != i)
            x_k_next[i] = (B[i] - s) / A[i][i]

        current_error = vec_norm(vec_diff(x_k_next, x_k))
        errors.append(current_error)

        if current_error < eps:
            break
        x_k = x_k_next

    return x_k_next, iterations, errors


def seidel(A, B, x0, eps):
    n = len(A)
    x_k = x0[:]
    iterations = 0
    errors = []

    while True:
        iterations += 1
        x_k_next = x_k[:]
        for i in range(n):
            s1 = sum(A[i][j] * x_k_next[j] for j in range(i))
            s2 = sum(A[i][j] * x_k[j] for j in range(i + 1, n))
            x_k_next[i] = (B[i] - s1 - s2) / A[i][i]

        current_error = vec_norm(vec_diff(x_k_next, x_k))
        errors.append(current_error)

        if current_error < eps:
            break
        x_k = x_k_next

    return x_k_next, iterations, errors


# =====================================================================
# ГОЛОВНИЙ БЛОК ПРОГРАМИ
# =====================================================================
if __name__ == "__main__":
    n_size = 100
    epsilon = 1e-14

    print("1. Генеруємо дані...")
    generate_data(n_size)
    A_file = read_matrix("matrix_A.txt")
    B_file = read_vector("vector_B.txt")
    x0_start = [1.0] * n_size

    print(f"2. Розв'язуємо СЛАР з точністю {epsilon}...\n")

    # Виклик методів та збереження історії похибок
    x_simp, it_simp, err_simp = simple_iteration(A_file, B_file, x0_start, epsilon)
    print(f"-> Проста ітерація: {it_simp} ітерацій")

    x_jac, it_jac, err_jac = jacobi(A_file, B_file, x0_start, epsilon)
    print(f"-> Метод Якобі: {it_jac} ітерацій")

    x_seid, it_seid, err_seid = seidel(A_file, B_file, x0_start, epsilon)
    print(f"-> Метод Зейделя: {it_seid} ітерацій")

    # =====================================================================
    # БУДУЄМО ГРАФІК
    # =====================================================================
    print("\n3. Будуємо графік збіжності...")

    plt.figure(figsize=(10, 6))

    # Малюємо лінії для кожного методу
    plt.plot(range(1, it_simp + 1), err_simp, label='Проста ітерація', color='blue')
    plt.plot(range(1, it_jac + 1), err_jac, label='Метод Якобі', color='green')
    plt.plot(range(1, it_seid + 1), err_seid, label='Метод Зейделя', color='red')

    # Налаштування графіка
    plt.yscale('log')  # Логарифмічна шкала для осі Y, бо похибка падає експоненційно
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.title('Графік збіжності ітераційних методів')
    plt.xlabel('Номер ітерації (k)')
    plt.ylabel('Похибка ||X^(k+1) - X^(k)|| (логарифмічна шкала)')
    plt.legend()

    # Зберігаємо графік у файл (зручно для вставки у звіт) та показуємо на екрані
    plt.savefig('convergence_plot.png', dpi=300)
    plt.show()
