import os
import numpy as np
import matplotlib.pyplot as plt

np.seterr(over='ignore', invalid='ignore')

# -----------------------------------------------------------------------------
# Weakly compressible bioreactor model
# -----------------------------------------------------------------------------
# This file is a variant of Bioreactor_OTR_model.py with a simple weakly
# compressible liquid model. The density field `rho` is allowed to vary mildly
# with pressure using a bulk modulus, while the flow solver still retains a
# projection-style pressure correction.
#
# Notes:
# - rho is initialized as a constant field and updated from pressure each step.
# - This is not a full compressible gas/liquid solver; it is a demo of weak
#   compressibility for a liquid with a very high bulk modulus.
# - Pressure correction uses mass divergence: div(rho * u) instead of div(u).
# -----------------------------------------------------------------------------

# =========================
# Physical dimensions
# =========================
volume_target = 20.0  # m^3
height = 3.0  # m
radius = np.sqrt(volume_target / (np.pi * height))
diameter = 2.0 * radius

# Reduced grid resolution for faster simulation
nx, ny, nz = 24, 24, 32
Lx = Ly = diameter
Lz = height

n_impellers = 2  # Number of Rushton impellers

# =========================
# Bioreactor-specific parameters
# =========================
T = 37.0  # Temperature [°C]
P_atm = 101325.0  # Atmospheric pressure [Pa]
P_gas = P_atm + 50000.0  # Gas pressure above liquid [Pa]

# Impeller rotation speed
N_rpm = 200.0  # Speed [1/min]
N = N_rpm / 60.0  # Speed [1/s]

# Aeration
Q_air = 0.05 * volume_target  # Gas flow rate [m³/s] (0.05 vvm)

# Oxygen solubility data (25°C, 1 atm)
O2_sat_ref = 8.6e-3  # mol/L at 25°C, 1 atm
Henry_const = 769.0  # Henry's constant [atm/mol_fraction]

# Monod kinetics parameters (substrate-based growth)
X_biomass_init = 5.0  # Initial biomass concentration [g/L]
mu_max = 0.35  # Maximum growth rate [1/h]
K_m_sub = 0.5  # Substrate half-saturation constant [g/L]
K_m_O2 = 6.0  # Oxygen half-saturation constant [mol/m³]
Y_XS = 0.5  # Yield coefficient [g X / g substrate]
Y_OS = 1.0  # Oxygen yield on substrate [g O2 / g substrate]

# Fed-Batch parameters
fed_batch_start_time = 10.0  # When to start feeding [s]
fed_batch_strategy = "linear"  # Options: "linear", "exponential", "do_stat"
feed_rate = 0.05  # Feed rate [L/s] - will be calculated based on strategy
substrate_conc_feed = 400.0  # Substrate concentration in feed [g/L]
initial_volume = 10.0  # Initial liquid volume [L]

# Thermal parameters
T_ref = 37.0  # Reference temperature [°C]
T_wall = 35.0  # Wall temperature [°C] (cooling)
alpha_heat = 1e-7  # Thermal diffusivity [m²/s] - reduced for stability
rho_cp = 4.18e6  # Volumetric heat capacity [J/(m³·K)]
h_wall = 50.0  # Wall heat transfer coefficient [W/(m²·K)]
A_wall = 2 * np.pi * radius * height  # Wall area [m²]
V_liquid = volume_target * 0.8  # Liquid volume [m³]

# Biological heat production
delta_H_bio = -250e3  # Enthalpy of biological O2 consumption [J/mol O2]
demo_heat_boost = 1.0  # 1.0 = realistic, >1 for faster demo

# Weak compressibility parameters
rho0 = 1000.0  # Reference density [kg/m³]
K_bulk = 2.0e8  # Bulk modulus [Pa]

# =========================
# Reactor geometry
# =========================
def create_stirred_tank(nx, ny, nz, radius, height):
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    x = (np.arange(nx, dtype=np.float32) + 0.5) * dx - radius
    y = (np.arange(ny, dtype=np.float32) + 0.5) * dy - radius
    z = (np.arange(nz, dtype=np.float32) + 0.5) * dz

    X, Y = np.meshgrid(x, y, indexing='ij')
    radial = (X**2 + Y**2) < radius**2
    liquid_level = 0.8 * height

    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[..., z <= liquid_level] = radial[:, :, None]

    return mask

mask = create_stirred_tank(nx, ny, nz, radius, height)


dx = dy = Lx / nx
dz = Lz / nz
dt = 0.001  # initial time step (s)
dt_max = 0.02  # maximum time-step size (s)
nt = 10000  # safety cap
t_final = 300.0  # desired simulation duration in seconds

mu = 0.005  # Molecular (dynamic) viscosity [Pa·s]

# =========================
# K-Epsilon turbulence model parameters (compressible version)
# =========================
C_1eps = 1.44  # Production constant
C_2eps = 1.92  # Dissipation constant
C_mu = 0.09    # Turbulent viscosity constant
sigma_k = 1.0  # Turbulent Prandtl number for k
sigma_eps = 1.3  # Turbulent Prandtl number for epsilon

# CFL control parameter
max_velocity_allowed = 1.5
CFL = max_velocity_allowed * dt / dx

# =========================
# Fields
# =========================
dtype = np.float32
u = np.zeros((nx, ny, nz), dtype=dtype)
v = np.zeros((nx, ny, nz), dtype=dtype)
w = np.zeros((nx, ny, nz), dtype=dtype)
p = np.zeros((nx, ny, nz), dtype=dtype)
rho = np.ones((nx, ny, nz), dtype=dtype) * rho0

# Initialize oxygen concentration field (mol/m³)
C_O2 = np.ones((nx, ny, nz), dtype=dtype) * 0.25
C_O2[~mask] = 0.0

# Initialize substrate concentration field (g/L)
C_sub = np.ones((nx, ny, nz), dtype=dtype) * 50.0  # Initial substrate concentration
C_sub[~mask] = 0.0

# Biomass field
X_bio = np.ones((nx, ny, nz), dtype=dtype) * X_biomass_init
X_bio[~mask] = 0.0

# Temperature field
T = np.ones((nx, ny, nz), dtype=dtype) * T_ref
T[~mask] = T_wall

# Turbulent kinetic energy (k) and dissipation rate (epsilon) fields
k_turb = np.ones((nx, ny, nz), dtype=dtype) * 1e-6  # Initial k [m²/s²]
k_turb[~mask] = 0.0
epsilon_turb = np.ones((nx, ny, nz), dtype=dtype) * 1e-8  # Initial epsilon [m²/s³]
epsilon_turb[~mask] = 0.0

# =========================
# Bubble tracking (simplified - only kLa field)
# =========================
def update_kLa_field(impeller_speed_rpm, bubble_count):
    N_rps = impeller_speed_rpm / 60.0
    kLa_base = 0.002 * (N_rps ** 0.8) * (Q_air / volume_target) ** 0.5
    kLa_field = np.ones((nx, ny, nz), dtype=np.float32) * kLa_base * 0.5

    z_imp = np.array([0.30, 0.65]) * nz
    for z_c in z_imp:
        iz = int(z_c)
        for k in range(max(0, iz-3), min(nz, iz+4)):
            dist_factor = np.exp(-(k - z_c)**2 / 4.0)
            kLa_field[:, :, k] = np.maximum(
                kLa_field[:, :, k],
                kLa_base * 1.5 * dist_factor
            )

    return kLa_field

# =========================
# Spatial derivatives
# =========================
def ddx(f):
    df = np.zeros_like(f)
    df[1:-1,:,:] = (f[2:,:,:] - f[:-2,:,:]) / (2*dx)
    return df


def ddy(f):
    df = np.zeros_like(f)
    df[:,1:-1,:] = (f[:,2:,:] - f[:,:-2,:]) / (2*dy)
    return df


def ddz(f):
    df = np.zeros_like(f)
    df[:,:,1:-1] = (f[:,:,2:] - f[:,:,:-2]) / (2*dz)
    return df


def laplace(f):
    lap = np.zeros_like(f)
    lap[1:-1,1:-1,1:-1] = (
        (f[2:,1:-1,1:-1] - 2*f[1:-1,1:-1,1:-1] + f[:-2,1:-1,1:-1]) / dx**2 +
        (f[1:-1,2:,1:-1] - 2*f[1:-1,1:-1,1:-1] + f[1:-1,:-2,1:-1]) / dy**2 +
        (f[1:-1,1:-1,2:] - 2*f[1:-1,1:-1,1:-1] + f[1:-1,1:-1,:-2]) / dz**2
    )
    return lap


def upwind_x(u, f):
    df = np.zeros_like(f)
    df[1:-1,:,:] = np.where(
        u[1:-1,:,:] > 0,
        (f[1:-1,:,:] - f[:-2,:,:]) / dx,
        (f[2:,:,:] - f[1:-1,:,:]) / dx
    )
    return df


def upwind_y(v, f):
    df = np.zeros_like(f)
    df[:,1:-1,:] = np.where(
        v[:,1:-1,:] > 0,
        (f[:,1:-1,:] - f[:,:-2,:]) / dy,
        (f[:,2:,:] - f[:,1:-1,:]) / dy
    )
    return df


def upwind_z(w, f):
    df = np.zeros_like(f)
    df[:,:,1:-1] = np.where(
        w[:,:,1:-1] > 0,
        (f[:,:,1:-1] - f[:,:,:-2]) / dz,
        (f[:,:,2:] - f[:,:,1:-1]) / dz
    )
    return df


def divergence(u,v,w):
    return ddx(u) + ddy(v) + ddz(w)


def divergence_mass(rho, u, v, w):
    return ddx(rho * u) + ddy(rho * v) + ddz(rho * w)

# =========================
# Bioreactor functions
# =========================
def henry_saturation(T_local, P_gas_pa):
    H_T = Henry_const * np.exp(0.016 * (T_local - 25.0))
    P_gas_atm = P_gas_pa / 101325.0
    X_O2 = 0.21
    P_O2 = P_gas_atm * X_O2
    C_sat = (P_O2 / H_T) * 1000.0
    return C_sat


def oxygen_transfer_rate(C_O2_local, C_sat, kLa_local):
    return np.maximum(kLa_local * (C_sat - C_O2_local), 0.0)


def monod_respiration(C_O2_local, X_local, T_local):
    q_O2_base = 10e-3 / 3600
    k_T = np.exp(0.1 * (T_local - T_ref))
    q_O2_T = q_O2_base * k_T
    respiration = q_O2_T * X_local * (C_O2_local / (K_m_O2 + C_O2_local))
    return respiration


def biological_heat_generation(respiration_rate):
    return -delta_H_bio * respiration_rate


# ===========================
# Fed-Batch functions
# ===========================
def get_feed_rate(time, strategy="linear"):
    """
    Calculate feeding rate based on strategy.
    Returns feed rate in L/s
    """
    if time < fed_batch_start_time:
        return 0.0
    
    elapsed = time - fed_batch_start_time
    
    if strategy == "linear":
        # Constant feeding
        return 0.05  # L/s
    
    elif strategy == "exponential":
        # Exponential feeding for constant specific growth rate
        mu_set = 0.25  # Target growth rate [1/h]
        return 0.05 * np.exp((mu_set / 3600.0) * elapsed)
    
    elif strategy == "do_stat":
        # Dissolved oxygen stat: feed when DO drops below threshold
        return 0.03  # Simplified - use external DO control
    
    return 0.0


def fed_batch_source_terms(time, volume_L, C_O2, X_bio, C_sub):
    """
    Calculate source terms due to fed-batch feeding.
    Adds substrate, but dilutes existing concentrations.
    Returns dilution factor and substrate addition.
    """
    feed_rate = get_feed_rate(time, fed_batch_strategy)
    
    if feed_rate > 0:
        # Volume change rate [L/s]
        V_dot = feed_rate
        # Dilution rate [1/s]
        dilution = V_dot / volume_L if volume_L > 0 else 0.0
        
        # Substrate being added
        substrate_in = feed_rate * substrate_conc_feed  # [g/s]
        
        return dilution, substrate_in
    else:
        return 0.0, 0.0


def growth_kinetics(C_O2_local, C_sub_local, X_local, T_local):
    """
    Calculate biomass growth rate based on substrate and oxygen availability.
    Uses Monod kinetics for both substrate and oxygen.
    Returns: specific growth rate [1/s], substrate consumption [g/L/s], oxygen consumption [mol/m³/s]
    """
    # Temperature correction
    k_T = np.exp(0.1 * (T_local - T_ref))
    
    # Monod kinetics for substrate and oxygen
    f_sub = C_sub_local / (K_m_sub + C_sub_local)
    f_O2 = C_O2_local / (K_m_O2 + C_O2_local)
    
    # Combined limitation (multiplicative)
    mu_specific = (mu_max * k_T * f_sub * f_O2) / 3600.0  # Convert from 1/h to 1/s
    
    # Growth rate
    dX_dt = mu_specific * X_local
    
    # Substrate consumption for growth
    dS_dt_growth = dX_dt / Y_XS
    
    # Oxygen consumption for growth
    dO2_dt_growth = dX_dt * Y_OS / 32.0  # Convert g to mol (O2 MW = 32)
    
    return mu_specific, dS_dt_growth, dO2_dt_growth

# =========================
# Pressure solver
# =========================
def pressure_poisson(p, u_star, v_star, w_star, rho_field):
    b = divergence_mass(rho_field, u_star, v_star, w_star) / dt
    b[~mask] = 0

    omega = 1.0
    tol = 1e-5
    coeff = dx**2

    for _ in range(120):
        p_old = p[1:-1,1:-1,1:-1].copy()

        neighbor_sum = (
            p[2:,1:-1,1:-1] + p[:-2,1:-1,1:-1] +
            p[1:-1,2:,1:-1] + p[1:-1,:-2,1:-1] +
            p[1:-1,1:-1,2:] + p[1:-1,1:-1,:-2]
        )

        p_new = (neighbor_sum - coeff * b[1:-1,1:-1,1:-1]) / 6.0
        p_relaxed = p_old + omega * (p_new - p_old)

        p[1:-1,1:-1,1:-1][mask_int] = p_relaxed[mask_int]
        p[~mask] = 0

        p[0,:,:]  = p[1,:,:]
        p[-1,:,:] = p[-2,:,:]
        p[:,0,:]  = p[:,1,:]
        p[:,-1,:] = p[:,-2,:]
        p[:,:,0]  = p[:,:,1]
        p[:,:,-1] = p[:,:,-2]

        if np.max(np.abs(p[1:-1,1:-1,1:-1] - p_old)) < tol:
            break

    return p

# =========================
# Impeller force
# =========================
def impeller_force(num_impellers=n_impellers):
    x = (np.arange(nx, dtype=np.float32) + 0.5) * dx - radius
    y = (np.arange(ny, dtype=np.float32) + 0.5) * dy - radius
    z = (np.arange(nz, dtype=np.float32) + 0.5) * dz

    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2) + 1e-6

    if num_impellers == 2:
        impeller_z = np.array([0.30, 0.65], dtype=np.float32) * Lz
    else:
        impeller_z = np.linspace(0.25 * Lz, 0.75 * Lz, num_impellers, dtype=np.float32)

    impeller_radius = 0.25 * radius
    impeller_thickness = 0.05 * Lz
    radial_blade_radius = 0.22 * radius
    radial_blade_width = 0.08 * radius

    strength_radial = np.float32(260.0)
    strength_axial = np.float32(80.0)

    Fx = np.zeros((nx, ny, nz), dtype=np.float32)
    Fy = np.zeros((nx, ny, nz), dtype=np.float32)
    Fz = np.zeros((nx, ny, nz), dtype=np.float32)

    radial_profile = np.exp(-((R - radial_blade_radius) / radial_blade_width)**2).astype(np.float32)
    tangential_x = -Y / R * strength_radial * radial_profile
    tangential_y = X / R * strength_radial * radial_profile
    radial_profile_3d = radial_profile[:, :, None]

    z_grid = z[None, None, :]
    axial_sign = np.where(z_grid >= impeller_z[:, None, None, None], 1.0, -1.0)
    z_profile = np.exp(-((z_grid - impeller_z[:, None, None, None]) / (impeller_thickness * 0.5))**2).astype(np.float32)

    for idx in range(num_impellers):
        weight = z_profile[idx]
        Fx += tangential_x[:, :, None] * weight
        Fy += tangential_y[:, :, None] * weight
        Fz += strength_axial * axial_sign[idx] * weight * radial_profile_3d

    return Fx, Fy, Fz

# =========================
# Baffles and shaft
# =========================
def add_baffles(mask):
    cx, cy = nx//2, ny//2
    width = 1
    length = nx//4

    mask[cx+length:cx+length+width, cy-1:cy+1, :] = 0
    mask[cx-length:cx-length+width, cy-1:cy+1, :] = 0
    mask[cx-1:cx+1, cy+length:cy+length+width, :] = 0
    mask[cx-1:cx+1, cy-length:cy-length+width, :] = 0

    return mask

mask = add_baffles(mask)


def add_shaft(mask):
    cx, cy = nx//2, ny//2
    r_shaft = int(max(2, 0.04 * nx))

    x = np.arange(nx) - cx
    y = np.arange(ny) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    shaft = (X**2 + Y**2) < r_shaft**2
    mask[shaft, :] = 0

    return mask

# =========================
# K-Epsilon turbulence model functions
# =========================
def compute_turbulent_viscosity(k_field, epsilon_field):
    """
    Compute turbulent dynamic viscosity from k and epsilon.
    mu_t = rho * C_mu * k^2 / epsilon
    Uses clipping to avoid division by zero and unphysical values.
    """
    epsilon_safe = np.maximum(epsilon_field, 1e-12)
    k_safe = np.maximum(k_field, 1e-12)
    mu_t = rho * C_mu * k_safe**2 / epsilon_safe
    # Clip turbulent viscosity to reasonable values
    mu_t = np.clip(mu_t, 0.0, 100.0 * mu)  # Cap at 100x molecular viscosity
    return mu_t


def compute_strain_rate_tensor(u_field, v_field, w_field):
    """
    Compute tensor of mean strain rate magnitudes S = sqrt(2*S_ij*S_ij)
    where S_ij = 1/2 * (du_i/dx_j + du_j/dx_i)
    Used for k-epsilon production term.
    """
    # Diagonal terms: du/dx, dv/dy, dw/dz
    dux = np.abs(ddx(u_field)) + 1e-12
    dvy = np.abs(ddy(v_field)) + 1e-12
    dwz = np.abs(ddz(w_field)) + 1e-12
    
    # Shear terms: du/dy + dv/dx, etc.
    duy = np.abs(ddy(u_field)) + 1e-12
    dvx = np.abs(ddx(v_field)) + 1e-12
    
    dvz = np.abs(ddz(v_field)) + 1e-12
    dwx = np.abs(ddx(w_field)) + 1e-12
    
    duz = np.abs(ddz(u_field)) + 1e-12
    dwz_y = np.abs(ddy(w_field)) + 1e-12
    
    # Magnitude of strain rate: S = sqrt(2*(S_11^2 + S_22^2 + ... + 2*S_12^2 + ...))
    S = np.sqrt(
        2.0 * (dux**2 + dvy**2 + dwz**2) +
        (duy + dvx)**2 + (dvz + dwx)**2 + (duz + dwz_y)**2
    )
    
    return S


def update_k_epsilon_equations(k_field, epsilon_field, u_field, v_field, w_field, mu_field, rho_field, mask_reg):
    """
    Solve k-epsilon transport equations for compressible flow.
    Returns updated k and epsilon fields.
    
    Standard k-epsilon model equations:
    dk/dt + div(rho*u*k) = div(mu_t/sigma_k * grad(k)) + P_k - rho*epsilon
    d(epsilon)/dt + div(rho*u*epsilon) = div(mu_t/sigma_eps * grad(epsilon)) + C_1eps*epsilon/k*P_k - C_2eps*rho*epsilon^2/k
    """
    # Compute strain rate magnitude
    S = compute_strain_rate_tensor(u_field, v_field, w_field)
    
    # Production of kinetic energy: P_k = mu_t * S^2
    # Avoid production in very low velocity regions
    S_safe = np.maximum(S, 1e-12)
    P_k = mu_field * S_safe**2
    P_k[~mask_reg] = 0.0
    
    # Dissipation rate
    epsilon_safe = np.maximum(epsilon_field, 1e-12)
    
    # k-equation: convection + diffusion + production - dissipation
    conv_k = u_field*upwind_x(u_field, k_field) + v_field*upwind_y(v_field, k_field) + w_field*upwind_z(w_field, k_field)
    mu_t_over_sigma_k = mu_field / sigma_k
    diff_k = laplace(mu_t_over_sigma_k * k_field)
    
    dk_dt = -conv_k + diff_k + P_k / rho_field - epsilon_field
    
    # epsilon-equation: convection + diffusion + production - dissipation
    conv_eps = u_field*upwind_x(u_field, epsilon_field) + v_field*upwind_y(v_field, epsilon_field) + w_field*upwind_z(w_field, epsilon_field)
    mu_t_over_sigma_eps = mu_field / sigma_eps
    diff_eps = laplace(mu_t_over_sigma_eps * epsilon_field)
    
    # Production and dissipation terms
    Pk_eps = C_1eps * epsilon_field / k_field * P_k  # Production scaled by epsilon/k
    eps_dissipation = C_2eps * rho_field * epsilon_field**2 / k_field  # Dissipation
    
    # Avoid division by zero
    Pk_eps[k_field < 1e-12] = 0.0
    eps_dissipation[k_field < 1e-12] = 0.0
    
    deps_dt = -conv_eps + diff_eps + Pk_eps / rho_field - eps_dissipation / rho_field
    
    # Update with clipping to ensure physical values
    k_new = k_field + dt * dk_dt
    epsilon_new = epsilon_field + dt * deps_dt
    
    k_new = np.clip(k_new, 0.0, 1000.0)  # Upper limit to prevent numerical issues
    epsilon_new = np.clip(epsilon_new, 1e-15, 1e-3)  # Ensure epsilon stays positive
    
    k_new[~mask_reg] = 0.0
    epsilon_new[~mask_reg] = 0.0
    
    return k_new, epsilon_new

mask = add_shaft(mask)
mask_int = mask[1:-1,1:-1,1:-1]
Fx, Fy, Fz = impeller_force()

# =========================
# Boundary conditions
# =========================
def apply_bc(u,v,w):
    u[mask == 0] = 0
    v[mask == 0] = 0
    w[mask == 0] = 0
    
    u[0,:,:]=u[-1,:,:]=0
    u[:,0,:]=u[:,-1,:]=0
    u[:,:,0]=u[:,:,-1]=0

    v[0,:,:]=v[-1,:,:]=0
    v[:,0,:]=v[:,-1,:]=0
    v[:,:,0]=v[:,:,-1]=0

    w[0,:,:]=w[-1,:,:]=0
    w[:,0,:]=w[:,-1,:]=0
    w[:,:,0]=w[:,:,-1]=0

    return u,v,w

# =========================
# Monitoring
# =========================
history = {
    'time': [],
    'OTR': [],
    'C_O2_avg': [],
    'C_O2_max': [],
    'C_sub_avg': [],
    'C_sub_max': [],
    'X_bio_avg': [],
    'X_bio_max': [],
    'T_avg': [],
    'T_max': [],
    'rho_avg': [],
    'max_velocity': [],
    'volume': [],
    'k_turb_avg': [],
    'k_turb_max': [],
    'epsilon_turb_avg': [],
    'epsilon_turb_max': [],
    'mu_t_avg': []
}

# =========================
# Simulation loop
# =========================
t_step = 0
time = 0.0
volume_L = initial_volume  # Liquid volume in liters
est_steps = int(max(1, t_final / max(dt, 1e-9)))
while time < t_final and t_step < max(nt, est_steps*10):
    speed = np.sqrt(u**2 + v**2 + w**2)
    max_speed = np.max(speed) + 1e-6

    CFL_target = 0.3
    dt = CFL_target * dx / max_speed
    dt = min(dt, dt_max)

    # Fed-Batch dynamics
    dilution, substrate_in = fed_batch_source_terms(time, volume_L, C_O2, X_bio, C_sub)
    
    # Update volume
    feed_rate = get_feed_rate(time, fed_batch_strategy)
    volume_L += feed_rate * dt
    
    # K-Epsilon turbulence model: compute turbulent viscosity
    mu_t = compute_turbulent_viscosity(k_turb, epsilon_turb)
    # Effective viscosity (molecular + turbulent)
    mu_eff = mu + mu_t
    nu_eff = mu_eff / rho
    
    conv_u = u*upwind_x(u,u) + v*upwind_y(v,u) + w*upwind_z(w,u)
    conv_v = u*upwind_x(u,v) + v*upwind_y(v,v) + w*upwind_z(w,v)
    conv_w = u*upwind_x(u,w) + v*upwind_y(v,w) + w*upwind_z(w,w)

    visc_u = nu_eff * laplace(u)
    visc_v = nu_eff * laplace(v)
    visc_w = nu_eff * laplace(w)

    u_star = u + dt * (-conv_u + visc_u + Fx/rho)
    v_star = v + dt * (-conv_v + visc_v + Fy/rho)
    w_star = w + dt * (-conv_w + visc_w + Fz/rho)
    
    D_O2 = 3e-9
    D_sub = 1e-9  # Substrate diffusivity [m²/s]

    C_sat = henry_saturation(T, P_gas)
    kLa_field = update_kLa_field(N_rpm, len(history['time']))
    OTR_local = oxygen_transfer_rate(C_O2, C_sat, kLa_field)
    
    # Growth kinetics
    mu_specific, dS_dt_growth, dO2_dt_growth = growth_kinetics(C_O2, C_sub, X_bio, T)
    
    # Biological heat generation from growth
    Q_bio = -delta_H_bio * dO2_dt_growth * demo_heat_boost

    # Oxygen equation with transfer, growth, and dilution
    conv_O2 = (
        u * upwind_x(u, C_O2) +
        v * upwind_y(v, C_O2) +
        w * upwind_z(w, C_O2)
    )
    diff_O2 = D_O2 * laplace(C_O2)
    C_O2 = C_O2 + dt * (
        -conv_O2 + diff_O2 + OTR_local 
        - dO2_dt_growth  # Growth consumption
        - dilution * C_O2  # Dilution by feeding
    )

    # Substrate equation with consumption and feeding
    conv_sub = (
        u * upwind_x(u, C_sub) +
        v * upwind_y(v, C_sub) +
        w * upwind_z(w, C_sub)
    )
    diff_sub = D_sub * laplace(C_sub)
    # Substrate source: feed contribution, sink: growth consumption
    substrate_source = (substrate_in / (volume_L * 1000.0)) * dt if volume_L > 0 else 0  # Convert to g/L concentration
    C_sub = C_sub + dt * (
        -conv_sub + diff_sub 
        - dS_dt_growth  # Growth consumption
        - dilution * C_sub  # Dilution by feeding
    ) + substrate_source
    
    # Biomass equation with growth and dilution
    conv_X = (
        u * upwind_x(u, X_bio) +
        v * upwind_y(v, X_bio) +
        w * upwind_z(w, X_bio)
    )
    X_bio = X_bio + dt * (
        -conv_X 
        + mu_specific * X_bio  # Growth
        - dilution * X_bio  # Dilution by feeding
    )

    # Temperature equation
    conv_T = (
        u * upwind_x(u, T) +
        v * upwind_y(v, T) +
        w * upwind_z(w, T)
    )
    diff_T = alpha_heat * laplace(T)

    Q_wall = np.zeros_like(T)
    wall_adj = np.zeros_like(mask, dtype=bool)
    wall_adj[1:,:,:]  |= mask[1:,:,:]  & ~mask[:-1,:,:]
    wall_adj[:-1,:,:] |= mask[:-1,:,:] & ~mask[1:,:,:]
    wall_adj[:,1:,:]  |= mask[:,1:,:] & ~mask[:,:-1,:]
    wall_adj[:,:-1,:] |= mask[:,:-1,:] & ~mask[:,1:,:]
    wall_adj[:,:,1:]  |= mask[:,:,1:]  & ~mask[:,:,:-1]
    wall_adj[:,:,:-1] |= mask[:,:,:-1] & ~mask[:,:,1:]
    Q_wall[wall_adj] = h_wall * (T_wall - T[wall_adj]) / dz

    T = T + dt * (-conv_T + diff_T + Q_bio / rho_cp + Q_wall / rho_cp)

    p = pressure_poisson(p, u_star, v_star, w_star, rho)

    u = u_star - dt * ddx(p) / rho
    v = v_star - dt * ddy(p) / rho
    w = w_star - dt * ddz(p) / rho

    # Update density from local pressure using a weakly compressible equation
    # of state. For liquids the bulk modulus is very large, so density changes
    # remain small even for noticeable pressure variations.
    rho = rho0 * (1.0 + (p - P_atm) / K_bulk)
    rho[~mask] = rho0
    rho = np.clip(rho, rho0 * 0.97, rho0 * 1.03)

    u,v,w = apply_bc(u,v,w)

    u *= 0.995
    v *= 0.995
    w *= 0.995
    
    # Update k-epsilon turbulence model
    k_turb, epsilon_turb = update_k_epsilon_equations(k_turb, epsilon_turb, u, v, w, mu_eff, rho, mask_int)

    u[mask == 0] = 0
    v[mask == 0] = 0
    w[mask == 0] = 0
    C_O2[mask == 0] = 0
    C_sub[mask == 0] = 0
    X_bio[mask == 0] = 0
    T[~mask] = T_wall
    k_turb[mask == 0] = 0.0
    epsilon_turb[mask == 0] = 0.0

    u = np.clip(u, -max_velocity_allowed, max_velocity_allowed)
    v = np.clip(v, -max_velocity_allowed, max_velocity_allowed)
    w = np.clip(w, -max_velocity_allowed, max_velocity_allowed)
    C_O2 = np.clip(C_O2, 0.0, C_sat.max())
    C_sub = np.clip(C_sub, 0.0, substrate_conc_feed)
    X_bio = np.clip(X_bio, 0.0, 200.0)  # Upper limit on biomass
    T = np.clip(T, 20.0, 50.0)

    div = np.max(np.abs(divergence_mass(rho, u, v, w)))
    speed = np.sqrt(u**2 + v**2 + w**2)
    max_speed = np.max(speed) + 1e-6

    dt = min(dt, dt_max)

    history['time'].append(time)
    history['OTR'].append(np.sum(OTR_local[mask]) * dx * dy * dz)
    history['C_O2_avg'].append(np.mean(C_O2[mask]))
    history['C_O2_max'].append(np.max(C_O2[mask]))
    history['C_sub_avg'].append(np.mean(C_sub[mask]))
    history['C_sub_max'].append(np.max(C_sub[mask]))
    history['X_bio_avg'].append(np.mean(X_bio[mask]))
    history['X_bio_max'].append(np.max(X_bio[mask]))
    history['T_avg'].append(np.mean(T[mask]))
    history['T_max'].append(np.max(T[mask]))
    history['rho_avg'].append(np.mean(rho[mask]))
    history['max_velocity'].append(max_speed)
    history['volume'].append(volume_L)
    history['k_turb_avg'].append(np.mean(k_turb[mask]))
    history['k_turb_max'].append(np.max(k_turb[mask]))
    history['epsilon_turb_avg'].append(np.mean(epsilon_turb[mask]))
    history['epsilon_turb_max'].append(np.max(epsilon_turb[mask]))
    history['mu_t_avg'].append(np.mean(mu_t[mask]))

    if np.isnan(div) or np.isinf(div) or div > 5.0:
        print(f"DIVERGENT at step {t_step}: div={div:.2e}")
        break

    time += dt
    t_step += 1

print(f"\n=== Fed-Batch Bioreactor Simulation Complete ===")
print(f"Simulation finished after {t_step} steps, final time {time:.2f} s")
print(f"Final reactor volume: {volume_L:.2f} L")
print(f"Final biomass concentration: {np.mean(X_bio[mask]):.2f} g/L")
print(f"Final substrate concentration: {np.mean(C_sub[mask]):.2f} g/L")
print(f"Average density: {np.mean(rho[mask]):.2f} kg/m³")
print(f"Final temperature: {np.mean(T[mask]):.2f} °C")

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes[0, 0].plot(history['time'], history['volume'], 'b-', marker='o', markersize=3, linewidth=2)
axes[0, 0].set_ylabel('Volume [L]')
axes[0, 0].set_title('Fed-Batch Volume Profile')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history['time'], history['X_bio_avg'], 'g-', label='Avg', marker='o', markersize=3, linewidth=2)
axes[0, 1].plot(history['time'], history['X_bio_max'], 'g--', label='Max', marker='o', markersize=3, linewidth=1)
axes[0, 1].set_ylabel('Biomass [g/L]')
axes[0, 1].set_title('Biomass Concentration')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(history['time'], history['C_sub_avg'], 'purple', label='Avg', marker='o', markersize=3, linewidth=2)
axes[0, 2].plot(history['time'], history['C_sub_max'], 'purple', linestyle='--', label='Max', marker='o', markersize=3, linewidth=1)
axes[0, 2].set_ylabel('Substrate [g/L]')
axes[0, 2].set_title('Substrate Concentration')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(history['time'], history['C_O2_avg'], 'orange', label='Avg', marker='o', markersize=3, linewidth=2)
axes[1, 0].plot(history['time'], history['C_O2_max'], 'orange', linestyle='--', label='Max', marker='o', markersize=3, linewidth=1)
axes[1, 0].set_ylabel('Oxygen [mol/m³]')
axes[1, 0].set_xlabel('Time [s]')
axes[1, 0].set_title('Dissolved Oxygen Concentration')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history['time'], history['T_avg'], 'r-', label='Avg', marker='o', markersize=3, linewidth=2)
axes[1, 1].plot(history['time'], history['T_max'], 'r--', label='Max', marker='o', markersize=3, linewidth=1)
axes[1, 1].set_ylabel('Temperature [°C]')
axes[1, 1].set_xlabel('Time [s]')
axes[1, 1].set_title('Temperature Profile')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(history['time'], history['OTR'], 'b-', marker='o', markersize=3, linewidth=2)
axes[1, 2].set_ylabel('OTR [mol/s]')
axes[1, 2].set_xlabel('Time [s]')
axes[1, 2].set_title('Oxygen Transfer Rate')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()