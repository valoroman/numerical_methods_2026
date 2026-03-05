import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import math


# =====================================================================
# ЧАСТИНА 1: Варіант 3 (Прогноз часу тренування - Метод Ньютона)
# =====================================================================
def create_sample_csv(filename="data.csv"):
    data = [
        {"n": 10000, "t": 8},
        {"n": 20000, "t": 20},
        {"n": 40000, "t": 55},
        {"n": 80000, "t": 150},
        {"n": 160000, "t": 420}
    ]
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["n", "t"])
            writer.writeheader()
            writer.writerows(data)


def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)


def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef[0, :]


def newton_polynomial(x_data, y_data, x_val):
    coef = divided_differences(x_data, y_data)
    n = len(x_data)
    result = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x_val - x_data[j])
        result += term
    return result


# =====================================================================
# ЧАСТИНА 2: Дослідження ефекту Рунге (Загальне завдання)
# =====================================================================
def runge_function(x):
    return 1 / (1 + 25 * x ** 2)


def omega_function(x_nodes, x_val):
    result = 1.0
    for xi in x_nodes:
        result *= (x_val - xi)
    return result


def investigate_runge_with_errors():
    x_dense = np.linspace(-1, 1, 500)
    y_true = runge_function(x_dense)
    nodes_list = [5, 10, 20]

    for n in nodes_list:
        x_nodes = np.linspace(-1, 1, n)
        y_nodes = runge_function(x_nodes)

        y_interp = [newton_polynomial(x_nodes, y_nodes, xi) for xi in x_dense]
        error = [abs(y_true[i] - y_interp[i]) for i in range(len(x_dense))]
        omega = [omega_function(x_nodes, xi) for xi in x_dense]

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Дослідження інтерполяції (n={n})", fontsize=16)

        axs[0].plot(x_dense, y_true, 'k-', label="f(x)")
        axs[0].plot(x_dense, y_interp, 'b-', label="N_n(x)")
        axs[0].scatter(x_nodes, y_nodes, color='red', zorder=5)
        axs[0].set_title("Функція та Інтерполяція")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(x_dense, error, 'r-')
        axs[1].set_title("Похибка e(x) = |f(x) - N_n(x)|")
        axs[1].set_yscale('log')
        axs[1].grid(True)

        axs[2].plot(x_dense, omega, 'g-')
        axs[2].set_title("Функція вузлів w_n(x)")
        axs[2].grid(True)
        plt.tight_layout()


def investigate_fixed_step():
    h = 0.2
    a = 0
    nodes_list = [5, 10, 20]

    plt.figure(figsize=(10, 6))
    for n in nodes_list:
        b = a + h * n
        x_nodes = np.linspace(a, b, n)
        y_nodes = runge_function(x_nodes)

        x_dense = np.linspace(a, b, 200)
        y_true = runge_function(x_dense)
        y_interp = [newton_polynomial(x_nodes, y_nodes, xi) for xi in x_dense]

        error = [abs(y_true[i] - y_interp[i]) for i in range(len(x_dense))]
        plt.plot(x_dense, error, label=f"Похибка (n={n}, b={b:.1f})")

    plt.title("Похибка при фіксованому кроці h=0.2 та змінному інтервалі")
    plt.xlabel("x")
    plt.ylabel("e(x)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)


# =====================================================================
# ЧАСТИНА 3: Факторіальні многочлени (Переклад С-коду для sin(x))
# =====================================================================
def factorial_func(x):
    return math.sin(x)


def fact(k):
    if k == 0 or k == 1:
        return 1
    return k * fact(k - 1)


def Cnk(n, k):
    return fact(n) // (fact(k) * fact(n - k))


def step(n):
    return -1 if n % 2 != 0 else 1


def deltaf(f_values, n):
    r = 0.0
    for k in range(n + 1):
        r += f_values[k] * step(n - k) * Cnk(n, k)
    return r


def factmn(t, k):
    mn = 1.0
    if k != 0:
        for i in range(k):
            mn *= (t - i)
    return mn


def fappr(f_values, n, t):
    res = 0.0
    for k in range(n + 1):
        res += deltaf(f_values, k) * factmn(t, k) / fact(k)
    return res


def factorial_analysis():
    a, b = 0.0, 1.0
    n = 20
    h = (b - a) / n

    # Генерація in.txt
    with open("in.txt", "w") as file1:
        for i in range(n + 1):
            x_val = a + i * h
            y_val = factorial_func(x_val)
            file1.write(f"{x_val:e} \t {y_val:e}\n")

    # Зчитування in.txt
    x_nodes, f_nodes = [], []
    with open("in.txt", "r") as file1:
        for line in file1:
            parts = line.split()
            if len(parts) >= 2:
                x_nodes.append(float(parts[0]))
                f_nodes.append(float(parts[1]))

    # Обчислення
    t = 0.0
    ht = float(n) / (20.0 * n)
    t_plot, appr_plot, error_plot = [], [], []

    with open("fappr.txt", "w") as file2, open("R.txt", "w") as file3:
        for j in range(20 * n + 1):
            appr_val = fappr(f_nodes, n, t)
            exact_val = factorial_func(a + h * t)
            error_val = abs(appr_val - exact_val)

            file2.write(f"{t:e} \t {appr_val:e}\n")
            file3.write(f"{t:e} \t {error_val:e}\n")

            t_plot.append(t)
            appr_plot.append(appr_val)
            error_plot.append(error_val)
            t += ht

    # Побудова графіка
    x_dense = [a + h * t_val for t_val in t_plot]
    y_exact = [factorial_func(x_val) for x_val in x_dense]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Інтерполяція факторіальним многочленом (C-код)', fontsize=14)

    axs[0].plot(x_dense, y_exact, 'k-', linewidth=2, label="f(x) = sin(x)")
    axs[0].plot(x_dense, appr_plot, 'b--', label="Факторіальний поліном")
    axs[0].scatter(x_nodes, f_nodes, color='red', zorder=5, label="Вузли інтерполяції")
    axs[0].set_title("Функція та Інтерполяція")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(x_dense, error_plot, 'r-')
    axs[1].set_title("Абсолютна похибка Eps(x)")
    axs[1].set_yscale('log')
    axs[1].grid(True)
    plt.tight_layout()


# =====================================================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# =====================================================================
if __name__ == "__main__":
    print("Виконується розрахунок Варіанту 3...")
    create_sample_csv()
    x_data, y_data = read_data("data.csv")

    target_x = 120000
    pred_newton = newton_polynomial(x_data, y_data, target_x)

    # Побудова графіка для Варіанту 3
    x_plot = np.linspace(min(x_data), max(x_data), 100)
    y_plot_newton = [newton_polynomial(x_data, y_data, xi) for xi in x_plot]

    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label='Експериментальні дані')
    plt.plot(x_plot, y_plot_newton, 'b-', label='Інтерполяція Ньютона')
    plt.scatter([target_x], [pred_newton], color='green', s=70, zorder=5,
                label=f'Прогноз ({target_x}) = {pred_newton:.1f}')
    plt.title("Прогноз часу тренування моделі")
    plt.xlabel("Розмір датасету (n)")
    plt.ylabel("Час (сек)")
    plt.legend()
    plt.grid(True)

    print("Виконується дослідження ефекту Рунге...")
    investigate_runge_with_errors()
    investigate_fixed_step()

    print("Виконується розрахунок факторіальних многочленів...")
    factorial_analysis()

    print("Відкриваю всі графіки! (Закрий їх, щоб завершити програму)")
    plt.show()  # Ця команда відкриє всі 6 вікон одночасно