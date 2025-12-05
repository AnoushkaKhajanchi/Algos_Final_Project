import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def f_2(t):
    return np.sqrt(2 / np.pi) * np.exp(-t**2 / 2)

def p_collision(c, r):
    def integrand(t):
        return (1 / c) * f_2(t / c) * (1 - t / r)
    result, _ = quad(integrand, 0, r)
    return result

def dp_dc_numerical(c, r, h=1e-6):
    return (p_collision(c + h, r) - p_collision(c - h, r)) / (2 * h)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

r_values = [2.0, 4.0, 6.0, 8.0]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot 1: |dp/dc| vs c/r (should collapse)
ax1 = axes[0]
for r, color in zip(r_values, colors):
    c_values = np.linspace(r / 2, 3 * r, 100)
    # Multiply by r to normalize
    derivatives = [abs(dp_dc_numerical(c, r)) * r for c in c_values]
    ax1.plot(c_values / r, derivatives, label=f'r = {r}', color=color, linewidth=2)

ax1.axhline(y=0.69, color='black', linestyle='--', linewidth=2, label='Upper bound $L \\approx 0.69$')
ax1.set_xlabel('$c / r$', fontsize=12)
ax1.set_ylabel('$|dp/dc| \\times r$', fontsize=12)
ax1.set_title('Scaled Derivative (All $r$ Values Collapse)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 0.8)

# Plot 2: Verification that max * r = constant
ax2 = axes[1]
r_test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
L_values = []
for r in r_test:
    c_values = np.linspace(r / 2, 3 * r, 100)
    derivatives = [abs(dp_dc_numerical(c, r)) for c in c_values]
    L_values.append(max(derivatives) * r)

ax2.bar([str(r) for r in r_test], L_values, color='steelblue', edgecolor='black')
ax2.axhline(y=0.69, color='red', linestyle='--', linewidth=2, label='$L \\approx 0.69$')
ax2.set_xlabel('Bucket width $r$', fontsize=12)
ax2.set_ylabel('$\\max|dp/dc| \\times r$', fontsize=12)
ax2.set_title('Lipschitz Constant $L$ (Maximum over $c$)', fontsize=12)
ax2.set_ylim(0, 0.8)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lipschitz_verification_corrected.pdf', dpi=300, bbox_inches='tight')
plt.savefig('lipschitz_verification_corrected.png', dpi=300, bbox_inches='tight')
plt.show()