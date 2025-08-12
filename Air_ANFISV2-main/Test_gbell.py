import numpy as np
import matplotlib.pyplot as plt

def gbellmf(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

def gbellmf_slope(x, a, b, c):
    term1 = (x - c) / a
    term2 = np.abs(term1) ** (2 * b - 1)
    return -2 * b / a * term2 * np.sign(x - c) / ((1 + np.abs(term1) ** (2 * b)) ** 2)

# ตัวอย่างพารามิเตอร์
a, b, c = 0.18951922054595577, 2.0150305045426804, 0.15748846328563104

# สร้างช่วงข้อมูล x
x_values = np.linspace(-1000, 1000, 400)
y_values = gbellmf(x_values, a, b, c)
slope_values = gbellmf_slope(x_values, a, b, c)

# สร้างกราฟ
plt.figure(figsize=(12, 6))

# พล็อตฟังก์ชัน Membership Function
plt.subplot(1, 2, 1)
plt.plot(x_values, y_values, 'b-', label='Gaussian Bell MF')
plt.title('Gaussian Bell Membership Function')
plt.xlabel('x')
plt.ylabel('Degree of Membership')
plt.grid(True)
plt.legend()

# พล็อตความชัน
plt.subplot(1, 2, 2)
plt.plot(x_values, slope_values, 'r-', label='Slope of Gaussian Bell MF')
plt.title('Slope of Gaussian Bell Membership Function')
plt.xlabel('x')
plt.ylabel('Slope')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()