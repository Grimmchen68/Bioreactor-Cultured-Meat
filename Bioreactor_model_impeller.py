import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Bioreactor Parameters
# -----------------------------
V = 5.0  # Reactor volume [m^3]
N = 2.0  # Impeller speed [rev/s]
D_impeller = 0.5  # Impeller diameter [m]
rho = 1000  # Liquid density [kg/m^3]
mu = 0.001  # Dynamic viscosity [Pa·s]

# Kinetic parameters
mu_max = 0.4       # Maximum specific growth rate [1/h]
Ks = 0.1           # Half-saturation constant [g/L]
Yxs = 0.5          # Biomass yield coefficient [gX/gS]
Ypx = 0.2          # Product yield coefficient [gP/gX]

# Initial conditions [g/L]
X0 = 0.1  # Biomass
S0 = 20.0 # Substrate
P0 = 0.0  # Product

# Simulation parameters
t_end = 20  # hours
dt = 0.1    # hours

# -----------------------------
# Impeller Power Calculation
# -----------------------------
def reynolds_number(N, D, rho, mu):
    """Calculate Reynolds number for impeller."""
    return rho * N * D**2 / mu

def power_number(Re):
    """Estimate power number (Np) based on flow regime."""
    if Re < 10:
        return 10.0  # Laminar
    elif Re > 1e4:
        return 5.0   # Turbulent
    else:
        # Transitional regime interpolation
        return 10.0 - (5.0 * (Re - 10) / (1e4 - 10))

def impeller_power(N, D, rho, mu):
    """Calculate impeller power in Watts."""
    Re = reynolds_number(N, D, rho, mu)
    Np = power_number(Re)
    return Np * rho * (N**3) * (D**5)

# -----------------------------
# Monod Growth Model
# -----------------------------
def derivatives(X, S, P):
    """Return dX/dt, dS/dt, dP/dt."""
    mu_val = mu_max * S / (Ks + S)
    dXdt = mu_val * X
    dSdt = -(1 / Yxs) * dXdt
    dPdt = Ypx * dXdt
    return dXdt, dSdt, dPdt

# -----------------------------
# Simulation Loop
# -----------------------------
time = np.arange(0, t_end + dt, dt)
X, S, P = [X0], [S0], [P0]

for _ in time[1:]:
    dX, dS, dP = derivatives(X[-1], S[-1], P[-1])
    X.append(X[-1] + dX * dt)
    S.append(max(S[-1] + dS * dt, 0))  # Prevent negative substrate
    P.append(P[-1] + dP * dt)

# -----------------------------
# Results
# -----------------------------
P_impeller = impeller_power(N, D_impeller, rho, mu)

print(f"Impeller Power: {P_impeller:.2f} W")
print(f"Final Biomass: {X[-1]:.2f} g/L")
print(f"Final Substrate: {S[-1]:.2f} g/L")
print(f"Final Product: {P[-1]:.2f} g/L")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(time, X, label="Biomass (X)")
plt.plot(time, S, label="Substrate (S)")
plt.plot(time, P, label="Product (P)")
plt.xlabel("Time [h]")
plt.ylabel("Concentration [g/L]")
plt.title("Impeller-Driven Bioreactor Simulation")
plt.legend()
plt.grid(True)
plt.show()
