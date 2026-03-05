import requests
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1-3. Запит до Open-Elevation API та табуляція
# ==========================================
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

print("Виконується запит до API...")
try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    print("Помилка доступу до API. Використовуються резервні дані.")
    # Резервні дані на випадок, якщо API не працює
    results = [{'latitude': 48.164214, 'longitude': 24.536044, 'elevation': 1200},
               {'latitude': 48.164983, 'longitude': 24.534836, 'elevation': 1250},
               {'latitude': 48.160250, 'longitude': 24.500106, 'elevation': 2061}]

n_points = len(results)
print("Кількість вузлів:", n_points)
print("\nТабуляція вузлів:")
print(" i |  Latitude  |  Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")


# ==========================================
# 4. Обчислення кумулятивної відстані
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]

for i in range(1, n_points):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

print("\nТабуляція (відстань, висота):")
print(" i | Distance (m) | Elevation (m)")
for i in range(n_points):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")


# ==========================================
# 6-9. Побудова кубічного сплайна та метод прогонки
# ==========================================
def tridiagonal_matrix_algorithm(alpha, beta, gamma, delta):
    """Метод прогонки (Thomas algorithm)"""
    n = len(delta)
    A = np.zeros(n - 1)
    B = np.zeros(n)

    # Пряма прогонка
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]

    for i in range(1, n - 1):
        denominator = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denominator
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denominator

    B[-1] = (delta[-1] - alpha[-1] * B[-2]) / (alpha[-1] * A[-2] + beta[-1])

    # Зворотна прогонка
    x = np.zeros(n)
    x[-1] = B[-1]
    for i in range(n - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]

    return x


def compute_spline_coefficients(x, y):
    n = len(x)
    h = np.diff(x)

    # Коефіцієнти a
    a = y[:-1]

    # Формування системи для c
    alpha = np.zeros(n)
    beta = np.ones(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)

    for i in range(1, n - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Граничні умови (вільний сплайн c_0 = 0, c_n = 0)
    beta[0] = 1
    gamma[0] = 0
    delta[0] = 0

    alpha[-1] = 0
    beta[-1] = 1
    delta[-1] = 0

    # Метод прогонки
    c = tridiagonal_matrix_algorithm(alpha, beta, gamma, delta)

    # Коефіцієнти b та d
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b, c[:-1], d


x_data = np.array(distances)
y_data = np.array(elevations)
a, b, c, d = compute_spline_coefficients(x_data, y_data)

print("\nКоефіцієнти сплайнів (a, b, c, d) для всіх точок:")
for i in range(len(a)):
    print(f"Сплайн {i + 1}: a={a[i]:.2f}, b={b[i]:.4f}, c={c[i]:.6f}, d={d[i]:.8f}")


# Функція для обчислення значення сплайна
def evaluate_spline(x_eval, x, a, b, c, d):
    y_eval = np.zeros_like(x_eval)
    for k, xv in enumerate(x_eval):
        i = np.searchsorted(x, xv) - 1
        i = np.clip(i, 0, len(a) - 1)
        dx = xv - x[i]
        y_eval[k] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return y_eval


# ==========================================
# 10-12. Побудова графіків (основний, 10/15/20 вузлів та похибка)
# ==========================================
x_smooth = np.linspace(x_data.min(), x_data.max(), 500)
y_smooth = evaluate_spline(x_smooth, x_data, a, b, c, d)

# 1. Основний графік (пункт 5)
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ko', label='Вихідні GPS дані')
plt.plot(x_smooth, y_smooth, 'b-', label='Кубічний сплайн (всі точки)')
plt.title('Профіль висоти: Заросляк - Говерла')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()
plt.show()

# Підготовка даних д0ля 10, 15 та 20 вузлів
idx_10 = np.linspace(0, n_points - 1, 10).astype(int)
idx_15 = np.linspace(0, n_points - 1, 15).astype(int)
idx_20 = np.linspace(0, n_points - 1, 20).astype(int)

def get_spline_for_subset(indices):
    x_sub = x_data[indices]
    y_sub = y_data[indices]
    a_s, b_s, c_s, d_s = compute_spline_coefficients(x_sub, y_sub)
    return x_sub, y_sub, evaluate_spline(x_smooth, x_sub, a_s, b_s, c_s, d_s)

x10, y10, smooth10 = get_spline_for_subset(idx_10)
x15, y15, smooth15 = get_spline_for_subset(idx_15)
x20, y20, smooth20 = get_spline_for_subset(idx_20)

# 2. Графіки з різною кількістю вузлів (пункт 10)
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

axs[0].plot(x_data, y_data, 'o', color='lightgray', label='Всі GPS дані')
axs[0].plot(x10, y10, 'ko', label='10 вузлів')
axs[0].plot(x_smooth, smooth10, 'r-', label='Сплайн (10 вузлів)')
axs[0].set_title('Інтерполяція: 10 вузлів')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(x_data, y_data, 'o', color='lightgray', label='Всі GPS дані')
axs[1].plot(x15, y15, 'ko', label='15 вузлів')
axs[1].plot(x_smooth, smooth15, 'g-', label='Сплайн (15 вузлів)')
axs[1].set_title('Інтерполяція: 15 вузлів')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(x_data, y_data, 'o', color='lightgray', label='Всі GPS дані')
axs[2].plot(x20, y20, 'ko', label='20 вузлів')
axs[2].plot(x_smooth, smooth20, 'b-', label='Сплайн (20 вузлів)')
axs[2].set_title('Інтерполяція: 20 вузлів')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

# 3. Графік похибки (пункт 12)
plt.figure(figsize=(10, 4))
error_10 = np.abs(smooth20 - smooth10)
error_15 = np.abs(smooth20 - smooth15)
plt.plot(x_smooth, error_10, 'r-', label='Похибка (10 вузлів)')
plt.plot(x_smooth, error_15, 'g-', label='Похибка (15 вузлів)')
plt.title('Графік похибки (відносно 20 вузлів)')
plt.xlabel('Відстань (м)')
plt.ylabel('Похибка (м)')
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# ДОДАТКОВІ ЗАВДАННЯ
# ==========================================
print("\n--- Додаткові характеристики маршруту ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n_points))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")

total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n_points))
print(f"Сумарний спуск (м): {total_descent:.2f}")

# Градієнт
grad_full = np.gradient(y_smooth, x_smooth) * 100
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

# Механічна енергія
mass = 80
g = 9.81
energy = mass * g * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")