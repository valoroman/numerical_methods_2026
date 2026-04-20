import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi


# --- П. 1: Задана функція ---
def f(x):
    # Функція навантаження на сервер
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a, b = 0, 24

# --- П. 2: Точне значення інтегралу ---
# Використовуємо вбудовану функцію для знаходження еталонного (точного) значення
I_0, _ = spi.quad(f, a, b, epsabs=1e-14)
print(f"2. Точне значення інтегралу I_0: {I_0:.12f}")


# --- П. 3: Складова формула Сімпсона ---
def simpson_composite(f, a, b, N):
    if N % 2 != 0:
        N += 1  # Число розбиттів має бути парним для Сімпсона
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    # Формула Сімпсона
    I_N = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return I_N


# --- П. 4: Дослідження залежності точності та пошук N_opt ---
N_values = np.arange(10, 1002, 2)
errors = []
N_opt = None
eps_opt = None
target_eps = 1e-12

for N in N_values:
    I_N = simpson_composite(f, a, b, N)
    err = abs(I_N - I_0)
    errors.append(err)
    if err <= target_eps and N_opt is None:
        N_opt = N
        eps_opt = err

print(f"4. N_opt для заданої точності ({target_eps}): {N_opt}")
print(f"   Похибка eps_opt: {eps_opt:.2e}")

# Побудова графіка залежності похибки від N
plt.figure(figsize=(10, 6))
plt.plot(N_values, errors, label=r'$\epsilon(N) = |I(N) - I_0|$', color='red')
plt.axhline(y=target_eps, color='blue', linestyle='--', label='Задана точність 1e-12')
plt.yscale('log')
plt.title('Залежність точності обчислення інтегралу від числа розбиттів N')
plt.xlabel('Число розбиттів (N)')
plt.ylabel('Похибка (логарифмічний масштаб)')
plt.grid(True)
plt.legend()
plt.show()

# --- П. 5: Обчислення похибки при N_0 ---
# Знаходимо N0 = N_opt / 10, кратне 8
N0_raw = max(8, N_opt // 10)
N_0 = N0_raw + (8 - N0_raw % 8) if N0_raw % 8 != 0 else N0_raw

I_N0 = simpson_composite(f, a, b, N_0)
eps0 = abs(I_N0 - I_0)
print(f"5. Обране N_0: {N_0}")
print(f"   Похибка eps0 при N_0: {eps0:.2e}")

# --- П. 6: Метод Рунге-Ромберга ---
I_N0_half = simpson_composite(f, a, b, N_0 // 2)
# Уточнення за формулою Рунге-Ромберга
I_R = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(I_R - I_0)
print(f"6. Значення за Рунге-Ромбергом I_R: {I_R:.12f}")
print(f"   Похибка epsR: {epsR:.2e}")

# --- П. 7: Метод Ейткена ---
I_N0_quarter = simpson_composite(f, a, b, N_0 // 4)

# Уточнення за формулою Ейткена
numerator = (I_N0_half) ** 2 - I_N0 * I_N0_quarter
denominator = 2 * I_N0_half - (I_N0 + I_N0_quarter)
I_E = numerator / denominator

# Порядок методу
p = (1 / np.log(2)) * np.log(abs((I_N0_quarter - I_N0_half) / (I_N0_half - I_N0)))
epsE = abs(I_E - I_0)

print(f"7. Значення за методом Ейткена I_E: {I_E:.12f}")
print(f"   Порядок методу p: {p:.4f}")
print(f"   Похибка epsE: {epsE:.2e}")

# --- П. 8: Аналіз ---
print(
    "8. Аналіз: Зменшення похибки при використанні методів Рунге-Ромберга та Ейткена демонструє їх ефективність для уточнення результатів базової формули Сімпсона без значного збільшення кількості вузлів.")


# --- П. 9: Адаптивний алгоритм ---
def adaptive_simpson(f, a, b, delta, func_calls=0):
    """Адаптивний алгоритм на основі формули Сімпсона."""
    m = (a + b) / 2
    h = b - a

    # Інтеграл на всьому відрізку (2 підвідрізки)
    I1 = (h / 6) * (f(a) + 4 * f(m) + f(b))
    func_calls += 3

    # Інтеграл на половинках відрізку
    m1 = (a + m) / 2
    m2 = (m + b) / 2
    I2_left = (h / 12) * (f(a) + 4 * f(m1) + f(m))
    I2_right = (h / 12) * (f(m) + 4 * f(m2) + f(b))
    I2 = I2_left + I2_right
    func_calls += 2  # f(a), f(m), f(b) вже були обчислені, додаємо лише f(m1), f(m2)

    # Умова збіжності
    if abs(I1 - I2) <= delta:
        return I2, func_calls
    else:
        # Рекурсивний поділ
        left_I, left_calls = adaptive_simpson(f, a, m, delta, 0)
        right_I, right_calls = adaptive_simpson(f, m, b, delta, 0)
        return left_I + right_I, func_calls + left_calls + right_calls


deltas = [1e-3, 1e-6, 1e-9]
print("9. Адаптивний алгоритм:")
for d in deltas:
    I_adapt, calls = adaptive_simpson(f, a, b, d)
    err_adapt = abs(I_adapt - I_0)
    print(f"   Delta: {d:.0e} | Значення: {I_adapt:.10f} | Похибка: {err_adapt:.2e} | Викликів функції: {calls}")
