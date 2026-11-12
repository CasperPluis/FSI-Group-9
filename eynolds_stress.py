
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: GROUP PARAMETERS
# ==========================================

h = 25.0       # Water depth [m]
U = 1.5        # Current velocity [m/s]
rho = 1025.0   # Seawater density [kg/m3]

# ==========================================
# STEP 2: READ MEASUREMENT FILES
# ==========================================

folder = Path(__file__).resolve().parent
files = sorted(folder.glob("Z_*.dat"))

if not files:
    raise FileNotFoundError("No measurement files found.")

z_values = []
tau_values = []

for file in files:

    # Dimensionless height z/h
    z_h = float(file.stem.split("_")[1]) / 100

    # Read horizontal and vertical velocities
    data = np.loadtxt(file, usecols=(1, 2))

    u = data[:, 0]
    w = data[:, 1]

    # ======================================
    # STEP 3: VELOCITY FLUCTUATIONS
    # ======================================

    u_prime = u - np.mean(u)
    w_prime = w - np.mean(w)

    # ======================================
    # STEP 4: REYNOLDS SHEAR STRESS
    # ======================================

    tau_R = -rho * U**2 * np.mean(u_prime * w_prime)

    z_values.append(z_h)
    tau_values.append(tau_R)

z_values = np.array(z_values)
tau_values = np.array(tau_values)

# Sort by height
order = np.argsort(z_values)
z_values = z_values[order]
tau_values = tau_values[order]

# ==========================================
# STEP 5: SELECT POINTS FOR LINEAR FIT
# ==========================================

# Exclude ONLY the lowest measurement
# All 13 remaining points are included

mask = z_values > np.min(z_values)

z_fit = z_values[mask]
tau_fit = tau_values[mask]

print("Points used for Reynolds stress fit:")
print(z_fit)
print(f"Number of points: {len(z_fit)}")

# ==========================================
# STEP 6: LINEAR REGRESSION
# ==========================================

# tau_R = slope * (z/h) + intercept

slope, intercept = np.polyfit(z_fit, tau_fit, 1)

# Bed shear stress at z = 0
tau_b = intercept

# Fitted line
z_line = np.linspace(0, 1, 300)
tau_line = slope * z_line + intercept

# ==========================================
# STEP 7: R-SQUARED
# ==========================================

tau_predicted = slope * z_fit + intercept

ss_res = np.sum((tau_fit - tau_predicted)**2)
ss_tot = np.sum((tau_fit - np.mean(tau_fit))**2)

r_squared = 1 - ss_res / ss_tot

# ==========================================
# STEP 8: PRINT RESULTS
# ==========================================

print("\nMETHOD 2 - REYNOLDS SHEAR STRESS")
print("--------------------------------")

print("\nMeasured Reynolds shear stresses:")

for z, tau in zip(z_values, tau_values):
    print(f"z/h = {z:.3f}, tau_R = {tau:.4f} Pa")

print("\nLinear regression results:")
print(f"Slope = {slope:.6f} Pa")
print(f"Intercept = {intercept:.6f} Pa")
print(f"R-squared = {r_squared:.4f}")

print("\nBED SHEAR STRESS - METHOD 2")
print(f"tau_b = {tau_b:.4f} Pa")

# ==========================================
# STEP 9: PLOT
# ==========================================

plt.figure(figsize=(8, 7))

# 13 measurements used for fit
plt.scatter(
    tau_values[mask],
    z_values[mask] * h,
    color="blue",
    label="Measurements used for fit"
)

# Lowest measurement excluded
plt.scatter(
    tau_values[~mask],
    z_values[~mask] * h,
    color="orange",
    marker="x",
    s=100,
    label="Excluded lowest measurement"
)

# Linear regression
plt.plot(
    tau_line,
    z_line * h,
    color="red",
    linewidth=2,
    label="Linear regression"
)

# Bed shear stress at z = 0
plt.scatter(
    tau_b,
    0,
    color="black",
    marker="x",
    s=120,
    label=f"Bed shear stress = {tau_b:.2f} Pa",
    zorder=5
)

plt.xlabel("Reynolds shear stress [Pa]")
plt.ylabel("Height above seabed [m]")
plt.title("Method 2 - Reynolds Shear Stress")

plt.legend()
plt.grid(True)
plt.ylim(0, h)

plt.savefig(folder / "method2_reynolds_stress.png", dpi=300)
plt.show()
