"""
Bed shear stress tau_b from the velocity measurements, determined in two ways.

Reynolds decomposition:   u = u_mean + u'     w = w_mean + w'
Reynolds shear stress:    tau = -rho * <u'w'>  (u'w' is negative so -rho * <u'w'> is positive)

Method 1 - log fit of the mean velocity profile (log law, rough bed):
    u(z) = (u*/kappa) * ln(z / z0)
    This is a straight line  u = a * ln(z) + b  ->  u* = kappa * a,   z0 = exp(-b / a)
    tau_b = rho * u*^2,   roughness height ks = 30 * z0

Method 2 - extrapolation of the Reynolds stress to the bed:
    tau(z) = tau_b * (1 - z/h)       (linear, zero at the water surface)
    least-squares fit of tau_b, tau_b = value of the fitted line at z = 0
"""

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- input
h = 25.0       # water depth [m]
U = 1.5        # depth-averaged velocity [m/s]
rho = 1025.0   # seawater density [kg/m3]
kappa = 0.41   # von Karman constant

# z/h range of the points used in each fit
LOG_ZMIN, LOG_ZMAX = 0.0, 0.2   # log fit: only the log layer near the bed. Above z/h ~ 0.2
                                # the shear stress is no longer ~constant and the eddy size
                                # is limited by the depth, so the log law does not hold there.
TAU_ZMIN = 0.05                 # Reynolds-stress fit: points above this z/h.
                                # The point at z/h = 0.027 is left out: that close to the
                                # bed -rho<u'w'> drops off (near-wall damping of w'), so it
                                # does not follow the linear profile.

# ---- read the data and do the Reynolds decomposition
z_list = []
u_mean_list = []
w_mean_list = []
uw_list = []

for filename in sorted(glob.glob("Z_*.dat")):
    # height from the file name: 'Z_23.3.dat' -> z = 0.233 * h
    z_i = float(filename[2:-4]) / 100 * h

    # read the data (dimensionless) and convert to m/s
    data = pd.read_csv(filename, sep=r"\s+", header=None, names=["t", "u", "w"])
    u_i = data["u"].values * U
    w_i = data["w"].values * U

    # Reynolds decomposition: mean + fluctuation
    u_fluct = u_i - u_i.mean()
    w_fluct = w_i - w_i.mean()

    z_list.append(z_i)
    u_mean_list.append(u_i.mean())
    w_mean_list.append(w_i.mean())
    uw_list.append(np.mean(u_fluct * w_fluct))   # <u'w'>

    print(f"done: {filename}")

z = np.array(z_list)
u = np.array(u_mean_list)
w = np.array(w_mean_list)
uw = np.array(uw_list)

# ---- Reynolds shear stress
tau = -rho * uw   # [Pa]

# ---- method 1: log fit of the velocity profile
# Log law for a rough bed:  u(z) = (u*/kappa) * ln(z / z0)
# Written out:              u(z) = (u*/kappa) * ln(z) - (u*/kappa) * ln(z0)
# This is a straight line   u = a * ln(z) + b   with   a = u*/kappa,   b = -(u*/kappa) * ln(z0)
# So by plotting u against ln(z) and fitting a straight line, we get u* and z0.

# step 1: select the points in the log layer (z/h between LOG_ZMIN and LOG_ZMAX)
in_log = (z / h >= LOG_ZMIN) & (z / h <= LOG_ZMAX)

# step 2: least-squares straight line through (ln z, u); polyfit returns slope a and intercept b
a, b = np.polyfit(np.log(z[in_log]), u[in_log], 1)   # u = a*ln(z) + b

# step 3: friction velocity from the slope, a = u*/kappa  ->  u* = kappa * a
u_star_log = kappa * a

# step 4: roughness length z0 from the intercept, b = -a * ln(z0)  ->  z0 = exp(-b/a)
#         (z0 is the height where the extrapolated log profile gives u = 0)
z0 = np.exp(-b / a)

# step 5: equivalent sand roughness (Nikuradse, hydraulically rough bed): ks = 30 * z0
ks = 30 * z0

# step 6: bed shear stress from the friction velocity, by definition u* = sqrt(tau_b / rho)
tau_b_log = rho * u_star_log**2

# step 7: fitted log profile at all measurement heights (for the table and the plot)
u_fit = u_star_log / kappa * np.log(z / z0)

# step 8: quality of the fit, R^2 = 1 - (sum of squared residuals) / (total sum of squares),
#         computed only over the points used in the fit
r2_log = 1 - np.sum((u[in_log] - u_fit[in_log])**2) / np.sum((u[in_log] - u[in_log].mean())**2)

# sensitivity check: the same log fit through all points, to see how much
# tau_b depends on the choice of the fit range
a_all, b_all = np.polyfit(np.log(z), u, 1)
tau_b_log_all = rho * (kappa * a_all)**2

# ---- method 2: Reynolds stress extrapolated to the bed, tau = tau_b * (1 - z/h)
in_tau = z / h >= TAU_ZMIN
x = 1 - z / h
tau_b_rey = np.sum(x[in_tau] * tau[in_tau]) / np.sum(x[in_tau]**2)   # least squares
u_star_rey = np.sqrt(tau_b_rey / rho)
tau_fit = tau_b_rey * x
r2_rey = 1 - np.sum((tau[in_tau] - tau_fit[in_tau])**2) / np.sum((tau[in_tau] - tau[in_tau].mean())**2)

# ---- results
results = pd.DataFrame({
    "z [m]": z,
    "z/h [-]": z / h,
    "u_mean [m/s]": u,
    "w_mean [m/s]": w,
    "<u'w'> [m2/s2]": uw,
    "tau [Pa]": tau,
    "u_logfit [m/s]": u_fit,
    "tau_fit [Pa]": tau_fit,
    "in log fit": in_log,
    "in tau fit": in_tau,
})
results.to_csv("bed_shear_stress_results.csv", index=False)

difference = (tau_b_log - tau_b_rey) / tau_b_rey * 100

print()
print(results.to_string(index=False))
print()
print(f"Method 1, log fit ({in_log.sum()} of {len(z)} points, {LOG_ZMIN} <= z/h <= {LOG_ZMAX}):")
print(f"   u*    = {u_star_log:.4f} m/s")
print(f"   z0    = {z0:.4f} m   (ks = 30 z0 = {ks:.3f} m)")
print(f"   tau_b = {tau_b_log:.2f} Pa,   R^2 = {r2_log:.3f}")
print(f"   (check: log fit through all {len(z)} points gives tau_b = {tau_b_log_all:.2f} Pa)")
print(f"Method 2, Reynolds stress ({in_tau.sum()} of {len(z)} points, z/h >= {TAU_ZMIN}):")
print(f"   u*    = {u_star_rey:.4f} m/s")
print(f"   tau_b = {tau_b_rey:.2f} Pa,   R^2 = {r2_rey:.3f}")
print(f"Difference (log fit vs Reynolds stress): {difference:.1f} %")

# ---- plots
# used in a fit: filled coloured circles; not used: large grey crosses
EXCL = dict(marker="x", color="grey", markersize=10, markeredgewidth=2, linestyle="none")

z_line = np.linspace(z0, h, 300)
u_line = u_star_log / kappa * np.log(z_line / z0)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5))

# 1: velocity profile, regular axis
ax1.plot(u[in_log], z[in_log], "o", color="tab:red", label="used in log fit")
if (~in_log).any():
    ax1.plot(u[~in_log], z[~in_log], **EXCL, label="not used in fit")
ax1.plot(u_line, z_line, "-", color="tab:red", label=f"log fit, u* = {u_star_log:.3f} m/s")
ax1.set_xlim(left=0)
ax1.set_ylim(0, h)
ax1.set_xlabel("u [m/s]")
ax1.set_ylabel("z [m]")
ax1.set_title("Velocity profile")
ax1.grid(True)
ax1.legend(loc="upper left")

# 2: velocity profile, log z-axis (log law = straight line)
ax2.semilogy(u[in_log], z[in_log], "o", color="tab:red", label="used in log fit")
if (~in_log).any():
    ax2.semilogy(u[~in_log], z[~in_log], **EXCL, label="not used in fit")
ax2.semilogy(u_line, z_line, "-", color="tab:red",
             label=f"log fit, z0 = {z0:.3f} m (ks = {ks:.2f} m)")
ax2.set_xlim(left=0)
ax2.set_ylim(z0, h)
ax2.set_xlabel("u [m/s]")
ax2.set_ylabel("z [m] (log scale)")
ax2.set_title("Velocity profile, log axis")
ax2.grid(True, which="both")
ax2.legend(loc="upper left")

# 3: shear stress profile with both estimates of tau_b
z_tau = np.linspace(0, h, 100)
ax3.plot(tau[in_tau], z[in_tau], "o", color="tab:blue", label="used in Reynolds fit")
if (~in_tau).any():
    ax3.plot(tau[~in_tau], z[~in_tau], **EXCL, label="not used in fit")
ax3.plot(tau_b_rey * (1 - z_tau / h), z_tau, "-", color="tab:blue",
         label=rf"Reynolds fit, $\tau_b$ = {tau_b_rey:.1f} Pa")
ax3.plot(tau_b_log * (1 - z_tau / h), z_tau, "--", color="tab:red",
         label=rf"from log fit, $\tau_b$ = {tau_b_log:.1f} Pa")
ax3.set_xlabel(r"$\tau = -\rho\langle u'w' \rangle$ [Pa]")
ax3.set_ylabel("z [m]")
ax3.set_title("Shear stress profile")
ax3.set_xlim(left=0)
ax3.set_ylim(0, h)
ax3.grid(True)
ax3.legend(loc="upper right")

fig.tight_layout()
fig.savefig("bed_shear_stress.png", dpi=150)
plt.show()
