import numpy as np
import matplotlib.pyplot as plt
import csv

# Налаштування темної естетики з неоновими кольорами для графіків
plt.style.use("dark_background")
NEON_CYAN = "#00FFFF"
NEON_PINK = "#FF00FF"
NEON_GREEN = "#39FF14"


def read_data(filename):
    """Зчитування середньомісячних температур з CSV."""
    x = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропускаємо заголовок Month, Temp
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(x), np.array(y)


def form_matrix(x, m):
    """Формування матриці системи нормальних рівнянь."""
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
    return A


def form_vector(x, y, m):
    """Формування вектора вільних членів."""
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))
    return b


def gauss_solve(A, b):
    """Розв'язок СЛАР методом Гауса з вибором головного елемента по стовпцях."""
    n = len(b)
    A = A.copy()
    b = b.copy()

    # Прямий хід
    for k in range(n - 1):
        # Вибір найбільшого по модулю головного елементу
        max_row = np.argmax(np.abs(A[k:n, k])) + k

        # Перестановка рядків
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            if A[k, k] == 0:
                continue
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотний хід
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if A[i, i] == 0:
            x_sol[i] = 0
        else:
            x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]

    return x_sol


def polynomial(x_vals, coef):
    """Обчислення значення алгебраїчного многочлена."""
    y_poly = np.zeros_like(x_vals, dtype=float)
    for i in range(len(coef)):
        y_poly += coef[i] * (x_vals ** i)
    return y_poly


def calc_variance(y_true, y_approx):
    """Обчислення дисперсії за формулою з методички."""
    n = len(y_true)
    return np.sqrt(np.sum((y_true - y_approx) ** 2) / (n + 1))


def main():
    # 1. Зчитування вхідних даних
    x, y = read_data('data.csv')
    n_points = len(x)

    # 2. Знаходження апроксимуючого многочлена та дисперсії для m = 1...10
    variances = []
    degrees = list(range(1, 11))

    for m in degrees:
        A = form_matrix(x, m)
        b = form_vector(x, y, m)
        coef = gauss_solve(A, b)
        y_approx = polynomial(x, coef)
        var = calc_variance(y, y_approx)
        variances.append(var)
        print(f"Степінь m={m}, Дисперсія = {var:.4f}")

    # 3. Вибір оптимального значення степені m (мінімум дисперсії)
    optimal_m = degrees[np.argmin(variances)]
    print(f"\nОптимальний степінь полінома: m = {optimal_m}")

    # 4. Побудова апроксимації для оптимального m
    A_opt = form_matrix(x, optimal_m)
    b_opt = form_vector(x, y, optimal_m)
    coef_opt = gauss_solve(A_opt, b_opt)

    # 5. Екстраполяція (прогноз) на наступні 3 місяці (Додаткове завдання)
    x_future = np.array([25, 26, 27])
    y_future = polynomial(x_future, coef_opt)
    print("\nПрогноз на наступні 3 місяці:")
    for month, temp in zip(x_future, y_future):
        print(f"Місяць {month}: {temp:.2f} градусів")

    # Генерація плавних точок для гарного відображення кривої
    x_smooth = np.linspace(min(x), max(x_future), 200)
    y_smooth = polynomial(x_smooth, coef_opt)

    # 6. Табулювання похибки
    # Оскільки фактичних даних для проміжних точок немає, похибку рахуємо лише для вузлів
    y_approx_nodes = polynomial(x, coef_opt)
    error_nodes = np.abs(y - y_approx_nodes)

    # === БЛОК ПОБУДОВИ ГРАФІКІВ ===

    # Графік 1: Залежність дисперсії від степені
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, variances, marker='o', color=NEON_PINK, linewidth=2, markersize=8)
    plt.title("Залежність дисперсії від степені многочлена", color="white", fontsize=14)
    plt.xlabel("Степінь (m)", color="white")
    plt.ylabel("Дисперсія", color="white")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Графік 2: Апроксимація та прогноз
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="white", label="Фактичні дані (таблиця)", zorder=5)
    plt.plot(x_smooth, y_smooth, color=NEON_CYAN, linewidth=2.5, label=f"Апроксимація (m={optimal_m})")
    plt.scatter(x_future, y_future, color=NEON_GREEN, marker='*', s=150, label="Прогноз (наступні 3 міс.)", zorder=5)
    plt.title("Апроксимація даних та прогноз температури", color="white", fontsize=14)
    plt.xlabel("Місяць", color="white")
    plt.ylabel("Температура", color="white")
    plt.legend(facecolor='black', edgecolor='white')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Графік 3: Похибка у вузлах
    plt.figure(figsize=(10, 6))
    plt.bar(x, error_nodes, color=NEON_PINK, alpha=0.7, width=0.5)
    plt.plot(x, error_nodes, color="white", marker='o', linestyle='--')
    plt.title("Абсолютна похибка апроксимації", color="white", fontsize=14)
    plt.xlabel("Місяць (вузли)", color="white")
    plt.ylabel("Похибка", color="white")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Відобразити всі створені графіки окремими вікнами
    plt.show()


if __name__ == "__main__":
    main()