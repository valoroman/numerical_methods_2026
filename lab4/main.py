import numpy as np
import matplotlib.pyplot as plt

# 1. Задана функція та її точна похідна
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def exact_derivative(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# Точка, в якій шукаємо похідну (згідно з прикладом з методички t0 = 1)
x0 = 1.0
exact_val = exact_derivative(x0)
print(f"1. Точне значення похідної в точці x0 = {x0}: {exact_val:.6f}")

# 2. Дослідження залежності похибки від кроку h
def approx_derivative(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

# Генеруємо значення h від 10^-20 до 10^3
h_values = np.logspace(-20, 3, 1000)
errors = [abs(approx_derivative(M, x0, h) - exact_val) for h in h_values]

# Знаходимо оптимальний крок
min_error_idx = np.argmin(errors)
h0 = h_values[min_error_idx]
R0 = errors[min_error_idx]

print(f"\n2. Оптимальний крок h0: {h0:e}")
print(f"   Найкраща досягнута точність R0: {R0:e}")

# Побудова графіка для п.2 (опціонально, але корисно для звіту)
plt.figure(figsize=(8, 5))
plt.loglog(h_values, errors)
plt.xlabel('Крок h')
plt.ylabel('Похибка R')
plt.title('Залежність похибки чисельного диференціювання від кроку h')
plt.grid(True, which="both", ls="--")
plt.axvline(h0, color='red', linestyle='--', label=f'Оптимальне h0 ≈ {h0:.1e}')
plt.legend()
plt.show()

# 3. Приймаємо крок h = 10^-3
h = 1e-3
print(f"\n3. Прийнято значення кроку h = {h}")

# 4. Обчислення значення похідної з кроками h та 2h
y_prime_h = approx_derivative(M, x0, h)
y_prime_2h = approx_derivative(M, x0, 2 * h)

print(f"\n4. Значення похідної з кроком h:  {y_prime_h:.8f}")
print(f"   Значення похідної з кроком 2h: {y_prime_2h:.8f}")

# 5. Обчислення похибки при кроці h
R1 = abs(y_prime_h - exact_val)
print(f"\n5. Похибка R1 при кроці h: {R1:e}")

# 6. Метод Рунге-Ромберга
y_prime_RR = y_prime_h + (y_prime_h - y_prime_2h) / 3
R2 = abs(y_prime_RR - exact_val)

print(f"\n6. Метод Рунге-Ромберга:")
print(f"   Уточнене значення: {y_prime_RR:.8f}")
print(f"   Похибка R2: {R2:e}")
print(f"   Характер зміни: похибка зменшилась у {R1/R2:.1f} разів")

# 7. Метод Ейткена
y_prime_4h = approx_derivative(M, x0, 4 * h)

# Уточнене значення за Ейткеном
numerator = y_prime_2h**2 - y_prime_4h * y_prime_h
denominator = 2 * y_prime_2h - (y_prime_4h + y_prime_h)
y_prime_Eitken = numerator / denominator

# Порядок точності p
p = (1 / np.log(2)) * np.log(abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h)))

R3 = abs(y_prime_Eitken - exact_val)

print(f"\n7. Метод Ейткена:")
print(f"   Значення похідної з кроком 4h: {y_prime_4h:.8f}")
print(f"   Уточнене значення: {y_prime_Eitken:.8f}")
print(f"   Оцінка порядку точності p: {p:.4f}")
print(f"   Похибка R3: {R3:e}")
print(f"   Характер зміни: похибка зменшилась у {R1/R3:.1f} разів порівняно з початковою")
