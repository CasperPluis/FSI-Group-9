
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: GROUP PARAMETERS
# ==========================================

h = 25.0       # Water depth [m]
U = 1.5        # Current velocity [m/s]
rho = 1025.0   # Seawater density [kg/m3]
kappa = 0.41   # Von Karman constant

# ==========================================
# STEP 2: READ MEASUREMENT FILES
# ==========================================

folder = Path(__file__).resolve().parent
files = sorted(folder.glob("Z_*.dat"))

if not files:
    raise FileNotFoundError("No measurement files found.")

z_values = []
u_mean_values = []

for file in files:

    # Dimensionless height z/h
    z_h = float(file.stem.split("_")[1]) / 100

    # Dimensionless horizontal velocity
    u = np.loadtxt(file, usecols=1)

    # Mean velocity
    u_mean = np.mean(u)

    z_values.append(z_h)
    u_mean_values.append(u_mean)

z_values = np.array(z_values)
u_mean_values = np.array(u_mean_values)

# Sort measurements by height
order = np.argsort(z_values)
z_values = z_values[order]
u_mean_values = u_mean_values[order]

# ==========================================
# STEP 3: SELECT POINTS FOR LOG FIT
# ==========================================

# Use ALL points with z/h < 0.20
# The lowest point IS included

mask = z_values < 0.20

z_fit = z_values[mask]
u_fit = u_mean_values[mask]

print("Points used for logarithmic fit:")
print(z_fit)
print(f"Number of points: {len(z_fit)}")

# ==========================================
# STEP 4: LOGARITHMIC REGRESSION
# ==========================================

# u/U = a * ln(z/h) + b

a, b = np.polyfit(np.log(z_fit), u_fit, 1)

# Generate fitted curve within selected region
z_line = np.linspace(min(z_fit), 0.20, 300)
u_line = a * np.log(z_line) + b

# ==========================================
# STEP 5: PHYSICAL PARAMETERS
# ==========================================

# Friction velocity [m/s]
u_star = a * kappa * U

# Roughness length [m]
z0 = h * np.exp(-b / a)

# Bed shear stress [Pa]
tau_b = rho * u_star**2

# R-squared
u_predicted = a * np.log(z_fit) + b

ss_res = np.sum((u_fit - u_predicted)**2)
ss_tot = np.sum((u_fit - np.mean(u_fit))**2)

r_squared = 1 - ss_res / ss_tot

# ==========================================
# STEP 6: PRINT RESULTS
# ==========================================

print("\nMETHOD 1 - LOGARITHMIC VELOCITY PROFILE")
print("---------------------------------------")

print(f"Coefficient a = {a:.6f}")
print(f"Coefficient b = {b:.6f}")
print(f"R-squared = {r_squared:.4f}")

print(f"Friction velocity u* = {u_star:.6f} m/s")
print(f"Roughness length z0 = {z0:.6f} m")
print(f"Bed shear stress = {tau_b:.4f} Pa")

# ==========================================
# STEP 7: PLOT
# ==========================================

plt.figure(figsize=(8, 7))

# All measurements not used in fit
plt.scatter(
    u_mean_values[~mask] * U,
    z_values[~mask] * h,
    color="gray",
    label="Measurements not used"
)

# Measurements used in fit
plt.scatter(
    u_fit * U,
    z_fit * h,
    color="blue",
    label="Measurements used for fit",
    zorder=3
)

# Logarithmic fit
plt.plot(
    u_line * U,
    z_line * h,
    color="red",
    linewidth=2,
    label="Logarithmic fit"
)

plt.xlabel("Mean horizontal velocity [m/s]")
plt.ylabel("Height above seabed [m]")
plt.title("Method 1 - Logarithmic Velocity Profile")

plt.legend()
plt.grid(True)
plt.ylim(0, h)

plt.savefig(folder / "method1_log_fit.png", dpi=300)
plt.show()
