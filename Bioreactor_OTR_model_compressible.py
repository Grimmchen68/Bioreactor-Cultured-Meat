import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import cg

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
# - k-epsilon: realizable constraints (variable Cmu, C1, production limiter)
#   plus conservative div(rho*u*phi) transport and dilatation terms.
# - Low-Mach preconditioner: blends compressible terms toward the
#   incompressible limit when local Mach number is small (typical for liquids).
# -----------------------------------------------------------------------------

# =========================
# Physical dimensions
# =========================
diameter = 2.516  # m
height = 1.6 * diameter  # m
radius = diameter / 2.0
volume_target = np.pi * radius**2 * height  # m^3
liquid_fill_fraction = 0.8  # fraction of tank height occupied by liquid

# Reduced grid resolution for faster simulation
nx, ny, nz = 24, 24, 32
Lx = Ly = diameter
Lz = height

n_impellers = 3  # Number of Rushton impellers

# =========================
# Bioreactor-specific parameters
# =========================
T = 37.0  # Temperature [°C]
P_atm = 101325.0  # Atmospheric pressure [Pa]
P_gas = P_atm + 50000.0  # Gas pressure above liquid [Pa]

# Impeller rotation speed
N_rpm = 200.0  # Speed [1/min]

# Aeration (Q_air is set from the grid liquid volume after geometry is built)
aeration_vvm = 0.05  # m³ gas per m³ liquid per second in this simplified model

# Oxygen solubility data (25°C, 1 atm)
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
substrate_conc_feed = 400.0  # Substrate concentration in feed [g/L]

# Thermal parameters
T_ref = 37.0  # Reference temperature [°C]
T_wall = 35.0  # Wall temperature [°C] (cooling)
alpha_heat = 1e-7  # Thermal diffusivity [m²/s] - reduced for stability
rho_cp = 4.18e6  # Volumetric heat capacity [J/(m³·K)]
h_wall = 50.0  # Wall heat transfer coefficient [W/(m²·K)]

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
    liquid_level = liquid_fill_fraction * height

    mask = np.zeros((nx, ny, nz), dtype=bool)
    mask[..., z <= liquid_level] = radial[:, :, None]

    return mask

mask = create_stirred_tank(nx, ny, nz, radius, height)


dx = dy = Lx / nx
dz = Lz / nz
x_coords = (np.arange(nx, dtype=np.float64) + 0.5) * dx - radius
y_coords = (np.arange(ny, dtype=np.float64) + 0.5) * dy - radius
z_coords = (np.arange(nz, dtype=np.float64) + 0.5) * dz
R_coords = np.sqrt(x_coords[:, None, None] ** 2 + y_coords[None, :, None] ** 2)
R_coords = np.broadcast_to(R_coords, (nx, ny, nz))

OUTPUT_DIR = os.path.join("results", "compressible_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
dt = 0.0005  # initial time step (s)
dt_max = 0.005  # maximum time-step size (s) - reduced for stability
dt_min = 1e-6  # minimum time-step size for robustness
dt_reduction_factor = 0.5  # reduce dt when step retries are needed
max_step_retries = 3  # limit number of retry attempts
nt = 1000000  # safety cap (increased for long simulations)
t_final = 3600  

mu = 0.005  # Molecular (dynamic) viscosity [Pa·s]

# =========================
# Realizable k-epsilon (Shih et al.) with compressible transport
# =========================
A0_real = 4.04
As_real = np.sqrt(6.0) * np.cos(1.0 / 3.0 * np.arccos(np.sqrt(6.0) / 3.0))
C_2eps = 1.9  # Dissipation constant
sigma_k = 1.0
sigma_eps = 1.2
P_limiter = 10.0  # Realizability cap: P_k <= P_limiter * rho * epsilon
C_comp_k = 2.0 / 3.0  # Dilatation correction in k equation
C_comp_eps = 2.0 / 3.0  # Dilatation correction factor in epsilon equation

max_velocity_allowed = 1.5

# Low-Mach preconditioner (liquid bioreactor: M ~ 0.001-0.01)
c_sound_ref = np.sqrt(K_bulk / rho0)
M_ref_precond = max(max_velocity_allowed / c_sound_ref, 1e-4)
precond_use_turbulent_mach = True
precond_blend_pressure_rhs = True
precond_blend_turbulence_transport = True
precond_blend_eos = True
eos_precond_floor = 0.2  # minimum acoustic density coupling at very low M

# =========================
# Fields
# =========================
dtype = np.float32
u = np.zeros((nx, ny, nz), dtype=dtype)
v = np.zeros((nx, ny, nz), dtype=dtype)
w = np.zeros((nx, ny, nz), dtype=dtype)
p = np.ones((nx, ny, nz), dtype=dtype) * P_atm
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
def update_kLa_field(impeller_speed_rpm):
    N_rps = impeller_speed_rpm / 60.0
    kLa_base = 0.002 * (N_rps ** 0.8) * (Q_air / volume_liquid_m3) ** 0.5
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


def div_upwind_flux_x(u, flux):
    df = np.zeros_like(flux)
    df[1:-1, :, :] = np.where(
        u[1:-1, :, :] > 0,
        (flux[1:-1, :, :] - flux[:-2, :, :]) / dx,
        (flux[2:, :, :] - flux[1:-1, :, :]) / dx,
    )
    return df


def div_upwind_flux_y(v, flux):
    df = np.zeros_like(flux)
    df[:, 1:-1, :] = np.where(
        v[:, 1:-1, :] > 0,
        (flux[:, 1:-1, :] - flux[:, :-2, :]) / dy,
        (flux[:, 2:, :] - flux[:, 1:-1, :]) / dy,
    )
    return df


def div_upwind_flux_z(w, flux):
    df = np.zeros_like(flux)
    df[:, :, 1:-1] = np.where(
        w[:, :, 1:-1] > 0,
        (flux[:, :, 1:-1] - flux[:, :, :-2]) / dz,
        (flux[:, :, 2:] - flux[:, :, 1:-1]) / dz,
    )
    return df


def convection_rho_scalar(rho_field, u_field, v_field, w_field, phi):
    """Conservative advection term div(rho * u * phi)."""
    fx = rho_field * u_field * phi
    fy = rho_field * v_field * phi
    fz = rho_field * w_field * phi
    return (
        div_upwind_flux_x(u_field, fx)
        + div_upwind_flux_y(v_field, fy)
        + div_upwind_flux_z(w_field, fz)
    )


def compute_speed_of_sound(rho_field):
    """Isothermal speed of sound for a weakly compressible liquid: c = sqrt(K/rho)."""
    return np.sqrt(K_bulk / np.maximum(rho_field.astype(np.float64), 1e-8))


def compute_low_mach_preconditioner(u_field, v_field, w_field, k_field, rho_field):
    """
    Local low-Mach preconditioner phi in [0, 1].

    phi -> 0 when M << M_ref (nearly incompressible limit)
    phi -> 1 when M ~ M_ref or larger

    M_local = max(|u|/c, sqrt(2k/3)/c) when turbulent Mach is enabled.
    """
    c_safe = np.maximum(compute_speed_of_sound(rho_field), 1e-8)
    speed = np.sqrt(
        u_field.astype(np.float64) ** 2
        + v_field.astype(np.float64) ** 2
        + w_field.astype(np.float64) ** 2
    )
    M_flow = speed / c_safe

    if precond_use_turbulent_mach:
        k_pos = np.maximum(k_field.astype(np.float64), 0.0)
        M_turb = np.sqrt(2.0 * k_pos / 3.0) / c_safe
        M_local = np.maximum(M_flow, M_turb)
    else:
        M_local = M_flow

    M2 = M_local ** 2
    Mr2 = M_ref_precond ** 2
    phi = M2 / (M2 + Mr2)
    return phi.astype(np.float32), M_local.astype(np.float32)


def precondition_mass_divergence(rho_field, u_field, v_field, w_field, k_field=None):
    """
    Low-Mach preconditioned mass divergence for the pressure Poisson RHS.

    Blends div(rho u) (compressible) with rho div(u) (low-Mach stable form):
      div_pre = phi * div(rho u) + (1 - phi) * rho div(u)
    """
    if k_field is None:
        k_field = np.zeros_like(u_field, dtype=np.float32)

    div_rho_u = divergence_mass(rho_field, u_field, v_field, w_field)
    if not precond_blend_pressure_rhs:
        return div_rho_u

    phi, _ = compute_low_mach_preconditioner(
        u_field, v_field, w_field, k_field, rho_field
    )
    div_u = divergence(u_field, v_field, w_field)
    div_low_mach = rho_field * div_u
    return phi * div_rho_u + (1.0 - phi) * div_low_mach


def apply_low_mach_density_eos(p_field, u_field, v_field, w_field, k_field, rho_field):
    """
    Weakly compressible EOS with low-Mach preconditioning.

    rho = rho0 * (1 + eos_scale * (p - P_atm) / K_bulk)
    eos_scale = max(phi, eos_precond_floor) when preconditioning is enabled.
    """
    pressure_term = (p_field - P_atm) / K_bulk
    if precond_blend_eos:
        phi, _ = compute_low_mach_preconditioner(
            u_field, v_field, w_field, k_field, rho_field
        )
        eos_scale = np.maximum(phi, eos_precond_floor)
        rho_new = rho0 * (1.0 + eos_scale * pressure_term)
    else:
        rho_new = rho0 * (1.0 + pressure_term)

    rho_new = rho_new.astype(np.float32)
    rho_new[~mask] = rho0
    return np.clip(rho_new, rho0 * 0.97, rho0 * 1.03)

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
_poisson_system = None


def apply_pressure_mirror_bc(p64):
    """Neumann (zero-gradient) pressure at domain boundaries."""
    p64[0, :, :] = p64[1, :, :]
    p64[-1, :, :] = p64[-2, :, :]
    p64[:, 0, :] = p64[:, 1, :]
    p64[:, -1, :] = p64[:, -2, :]
    p64[:, :, 0] = p64[:, :, 1]
    p64[:, :, -1] = p64[:, :, -2]


def _build_poisson_system():
    """Build sparse Laplacian for interior fluid cells (built once after mask is final)."""
    mi = mask[1:-1, 1:-1, 1:-1]
    shape = mi.shape
    inv_dx2 = 1.0 / dx**2
    inv_dy2 = 1.0 / dy**2
    inv_dz2 = 1.0 / dz**2

    fluid = np.argwhere(mi)
    n = len(fluid)
    index_map = -np.ones(shape, dtype=np.int32)
    for idx, (i, j, k) in enumerate(fluid):
        index_map[i, j, k] = idx

    A = lil_matrix((n, n), dtype=np.float64)
    offsets = [
        (1, 0, 0, inv_dx2), (-1, 0, 0, inv_dx2),
        (0, 1, 0, inv_dy2), (0, -1, 0, inv_dy2),
        (0, 0, 1, inv_dz2), (0, 0, -1, inv_dz2),
    ]

    for idx, (i, j, k) in enumerate(fluid):
        diag = 0.0
        for di, dj, dk, coeff in offsets:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
                if mi[ni, nj, nk]:
                    A[idx, index_map[ni, nj, nk]] = coeff
                    diag -= coeff
                else:
                    # Solid neighbor: p' = 0 (homogeneous Dirichlet perturbation)
                    diag -= coeff
            # Domain boundary: Neumann mirror BC, no extra stencil term.
        A[idx, idx] = diag

    return csr_matrix(A), index_map, mi, fluid


def _poisson_equation_residual(p64, b64, mi):
    """Max |∇²p' - b| over fluid interior cells (p64 is perturbation from P_atm)."""
    lap = laplace(p64.astype(np.float32)).astype(np.float64)
    res_vals = np.abs(lap[1:-1, 1:-1, 1:-1][mi] - b64[1:-1, 1:-1, 1:-1][mi])
    return float(np.max(res_vals)) if res_vals.size else 0.0


def _pressure_poisson_sor(p64, b64, mi, max_iter=200, omega=1.0, tol=1e-4):
    """Red-black Gauss-Seidel fallback aligned with the incompressible model."""
    coeff_x = dx**2
    pc = p64[1:-1, 1:-1, 1:-1]

    for _ in range(max_iter):
        p_before = pc.copy()
        for color in (0, 1):
            I, J, K = np.mgrid[0:pc.shape[0], 0:pc.shape[1], 0:pc.shape[2]]
            color_mask = ((I + J + K) % 2 == color) & mi

            nsum = (
                p64[2:, 1:-1, 1:-1] + p64[:-2, 1:-1, 1:-1]
                + p64[1:-1, 2:, 1:-1] + p64[1:-1, :-2, 1:-1]
                + p64[1:-1, 1:-1, 2:] + p64[1:-1, 1:-1, :-2]
            )
            b_inner = b64[1:-1, 1:-1, 1:-1]
            p_new = (nsum - coeff_x * b_inner) / 6.0

            pc[color_mask] = (
                pc[color_mask]
                + omega * (p_new[color_mask] - pc[color_mask])
            )
            p64[1:-1, 1:-1, 1:-1] = pc
            p64[~mask] = 0.0
            apply_pressure_mirror_bc(p64)

        res = np.max(np.abs(pc[mi] - p_before[mi])) if np.any(mi) else 0.0
        eq_res = _poisson_equation_residual(p64, b64, mi)
        if res < tol or eq_res < tol:
            return eq_res, True

    return _poisson_equation_residual(p64, b64, mi), False


def pressure_poisson(p, u_star, v_star, w_star, rho_field, k_field=None):
    """
    Solve ∇²p' = div_pre(rho u*) / dt for pressure perturbation p' (p = P_atm + p').
    div_pre uses low-Mach preconditioning when enabled.
    """
    global _poisson_system
    if _poisson_system is None:
        _poisson_system = _build_poisson_system()

    A, index_map, mi, fluid = _poisson_system

    b = precondition_mass_divergence(rho_field, u_star, v_star, w_star, k_field) / dt
    b[~mask] = 0.0
    b64 = b.astype(np.float64)

    rhs = np.array([b64[i + 1, j + 1, k + 1] for i, j, k in fluid], dtype=np.float64)

    p64 = p.astype(np.float64) - P_atm
    x0 = np.array([p64[i + 1, j + 1, k + 1] for i, j, k in fluid], dtype=np.float64)

    recovery_used = False
    p_vec, info = cg(A, rhs, x0=x0, rtol=1e-8, atol=1e-10, maxiter=1000)
    converged = info == 0

    if not converged or np.isnan(p_vec).any() or np.isinf(p_vec).any():
        recovery_used = True
        p64.fill(0.0)
        eq_res, converged = _pressure_poisson_sor(p64, b64, mi)
        if not converged:
            print(f"pressure_poisson warning: SOR fallback did not fully converge, eq_res={eq_res:.3e}")
    else:
        for idx, (i, j, k) in enumerate(fluid):
            p64[i + 1, j + 1, k + 1] = p_vec[idx]
        p64[~mask] = 0.0
        apply_pressure_mirror_bc(p64)
        eq_res = _poisson_equation_residual(p64, b64, mi)
        converged = eq_res < 1e-3

    p64 = p64 + P_atm
    p_max = P_atm + 0.001 * K_bulk
    p_min = P_atm - 0.001 * K_bulk
    p64 = np.nan_to_num(p64, nan=P_atm, posinf=p_max, neginf=p_min)
    p64 = np.clip(p64, p_min, p_max)
    p[:] = p64.astype(p.dtype)
    return p, converged, recovery_used

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

    impeller_thickness = 0.05 * Lz
    radial_blade_radius = 0.22 * radius
    radial_blade_width = 0.08 * radius

    # Scale factor to reduce impeller forcing for numerical stability
    strength_scale = 0.01
    strength_radial = np.float32(260.0 * strength_scale)
    strength_axial = np.float32(80.0 * strength_scale)

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

    Fx = smooth_force_field(Fx)
    Fy = smooth_force_field(Fy)
    Fz = smooth_force_field(Fz)

    Fx[~mask] = 0.0
    Fy[~mask] = 0.0
    Fz[~mask] = 0.0

    return Fx, Fy, Fz


def smooth_force_field(field, iterations=2):
    """Apply simple 3D smoothing to reduce sharp force gradients."""
    f = field.astype(np.float32)
    for _ in range(iterations):
        f_new = np.zeros_like(f)
        f_new[1:-1,1:-1,1:-1] = (
            f[1:-1,1:-1,1:-1] * 4.0 +
            f[2:,1:-1,1:-1] + f[:-2,1:-1,1:-1] +
            f[1:-1,2:,1:-1] + f[1:-1,:-2,1:-1] +
            f[1:-1,1:-1,2:] + f[1:-1,1:-1,:-2]
        ) / 10.0
        f = f_new
    return f


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
def compute_strain_rate_magnitude(u_field, v_field, w_field):
    """Return S = sqrt(2 S_ij S_ij)."""
    dudx = ddx(u_field)
    dudy = ddy(u_field)
    dudz = ddz(u_field)
    dvdx = ddx(v_field)
    dvdy = ddy(v_field)
    dvdz = ddz(v_field)
    dwdx = ddx(w_field)
    dwdy = ddy(w_field)
    dwdz = ddz(w_field)

    s11 = dudx
    s22 = dvdy
    s33 = dwdz
    s12 = 0.5 * (dudy + dvdx)
    s13 = 0.5 * (dudz + dwdx)
    s23 = 0.5 * (dvdz + dwdy)
    strain2 = 2.0 * (
        s11**2 + s22**2 + s33**2
        + 2.0 * s12**2 + 2.0 * s13**2 + 2.0 * s23**2
    )
    return np.sqrt(np.maximum(strain2, 1e-24))


def compute_realizable_coefficients(strain_mag, k_field, epsilon_field):
    """
    Realizable k-epsilon coefficients (Shih et al.):
      eta = S k / epsilon
      Cmu = 1 / (A0 + As eta / (eta + 5))
      C1  = max(0.43, eta / (eta + 5))
    """
    k_safe = np.maximum(k_field, 1e-12)
    eps_safe = np.maximum(epsilon_field, 1e-12)
    eta = strain_mag * k_safe / eps_safe
    eta_term = eta / (eta + 5.0)
    c_mu = 1.0 / (A0_real + As_real * eta_term)
    c1 = np.maximum(0.43, eta_term)
    return c_mu, c1, eta


def compute_turbulent_viscosity(k_field, epsilon_field, rho_field, u_field, v_field, w_field):
    """
    Realizable turbulent viscosity:
      mu_t = rho * Cmu(eta) * k^2 / epsilon
    """
    strain_mag = compute_strain_rate_magnitude(u_field, v_field, w_field)
    c_mu, _, _ = compute_realizable_coefficients(strain_mag, k_field, epsilon_field)
    epsilon_safe = np.maximum(epsilon_field, 1e-12)
    k_safe = np.maximum(k_field, 1e-12)
    mu_t = rho_field * c_mu * k_safe**2 / epsilon_safe
    mu_t = np.clip(mu_t, 0.0, 100.0 * mu)
    return mu_t


def compute_turbulent_production(
    mu_t_field,
    strain_mag,
    rho_field,
    epsilon_field,
):
    """
    Turbulent production with realizability limiter:
      P_k = min(mu_t * S^2, P_limiter * rho * epsilon)
    """
    P_k = mu_t_field * strain_mag**2
    P_lim = P_limiter * rho_field * np.maximum(epsilon_field, 1e-12)
    return np.minimum(P_k, P_lim)


def _resolve_turbulence_mask(mask_reg, field_shape):
    if mask_reg.shape != field_shape:
        if len(field_shape) == 3 and mask_reg.shape == tuple(s - 2 for s in field_shape):
            mask_used = np.zeros(field_shape, dtype=bool)
            mask_used[1:-1, 1:-1, 1:-1] = mask_reg
            return mask_used
        try:
            return np.broadcast_to(mask_reg, field_shape)
        except Exception:
            return np.ones(field_shape, dtype=bool)
    return mask_reg


def update_k_epsilon_equations(
    k_field,
    epsilon_field,
    u_field,
    v_field,
    w_field,
    rho_field,
    mask_reg,
):
    """
    Realizable compressible k-epsilon transport for variable-density liquid flow.

    Realizability:
      - Cmu(eta), C1(eta) with eta = S k / epsilon
      - P_k capped at P_limiter * rho * epsilon

    Low-Mach preconditioner (phi):
      - Turbulence advection blends conservative and incompressible forms
      - Dilatation terms scaled by phi
      - phi -> 0 when M << M_ref_precond

    Compressible transport:
      dk/dt = [-div(rho u k) + k div(rho u)] / rho + diffusion + P_k/rho - eps
              - C_comp_k * k * div(u)
      de/dt = [-div(rho u e) + e div(rho u)] / rho + diffusion + C1 e/k P_k/rho
              - C2 rho e^2/k / rho - C_comp_eps * C1 e/k * div(u)
    """
    rho_safe = np.maximum(rho_field.astype(np.float64), 1e-8)
    k_safe = np.maximum(k_field, 1e-12)

    phi, _ = compute_low_mach_preconditioner(
        u_field, v_field, w_field, k_field, rho_field
    )

    strain_mag = compute_strain_rate_magnitude(u_field, v_field, w_field)
    c_mu, c1, _ = compute_realizable_coefficients(strain_mag, k_field, epsilon_field)
    mu_t_field = rho_field * c_mu * k_safe**2 / epsilon_safe
    mu_t_field = np.clip(mu_t_field, 0.0, 100.0 * mu)

    P_k = compute_turbulent_production(mu_t_field, strain_mag, rho_field, epsilon_field)
    mask_used = _resolve_turbulence_mask(mask_reg, P_k.shape)
    P_k = np.where(mask_used, P_k, 0.0)
    mu_t_field = np.where(mask_used, mu_t_field, 0.0)

    div_rho_u = divergence_mass(rho_field, u_field, v_field, w_field)
    div_u = div_rho_u / rho_safe

    conv_k = convection_rho_scalar(rho_field, u_field, v_field, w_field, k_field)
    conv_eps = convection_rho_scalar(rho_field, u_field, v_field, w_field, epsilon_field)

    mu_k = mu + mu_t_field / sigma_k
    mu_eps = mu + mu_t_field / sigma_eps
    diff_k = laplace(mu_k * k_field)
    diff_eps = laplace(mu_eps * epsilon_field)

    advect_k_comp = (-conv_k + k_field * div_rho_u) / rho_safe
    advect_eps_comp = (-conv_eps + epsilon_field * div_rho_u) / rho_safe
    if precond_blend_turbulence_transport:
        conv_k_incomp = (
            u_field * upwind_x(u_field, k_field)
            + v_field * upwind_y(v_field, k_field)
            + w_field * upwind_z(w_field, k_field)
        )
        conv_eps_incomp = (
            u_field * upwind_x(u_field, epsilon_field)
            + v_field * upwind_y(v_field, epsilon_field)
            + w_field * upwind_z(w_field, epsilon_field)
        )
        advect_k = phi * advect_k_comp + (1.0 - phi) * (-conv_k_incomp)
        advect_eps = phi * advect_eps_comp + (1.0 - phi) * (-conv_eps_incomp)
    else:
        advect_k = advect_k_comp
        advect_eps = advect_eps_comp

    compress_k = C_comp_k * k_field * div_u * phi
    compress_eps = C_comp_eps * c1 * (epsilon_field / k_safe) * div_u * phi

    dk_dt = (
        advect_k
        + diff_k
        + P_k / rho_safe
        - epsilon_field
        - compress_k
    )

    Pk_eps = c1 * (epsilon_field / k_safe) * P_k
    eps_dissipation = C_2eps * rho_safe * epsilon_field**2 / k_safe

    deps_dt = (
        advect_eps
        + diff_eps
        + Pk_eps / rho_safe
        - eps_dissipation / rho_safe
        - compress_eps
    )

    k_new = k_field + dt * dk_dt
    epsilon_new = epsilon_field + dt * deps_dt

    k_new = np.clip(k_new, 0.0, 1000.0)
    epsilon_new = np.clip(epsilon_new, 1e-15, 1e-3)
    k_new[~mask_used] = 0.0
    epsilon_new[~mask_used] = 0.0

    return k_new, epsilon_new

mask = add_shaft(mask)
mask_int = mask[1:-1,1:-1,1:-1]

cell_volume = dx * dy * dz
volume_liquid_m3 = float(np.sum(mask) * cell_volume)
volume_liquid_L = volume_liquid_m3 * 1000.0
Q_air = aeration_vvm * volume_liquid_m3  # Gas flow rate [m³/s]

print(
    f"Liquid volume from CFD grid: {volume_liquid_L:.1f} L "
    f"({volume_liquid_m3:.3f} m³), fill={liquid_fill_fraction:.0%}, "
    f"Q_air={Q_air:.4f} m³/s"
)
print(
    f"Low-Mach preconditioner: c_sound={c_sound_ref:.1f} m/s, "
    f"M_ref={M_ref_precond:.4e}, eos_floor={eos_precond_floor:.2f}"
)

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
    'mu_t_avg': [],
    'pressure_recovery_used': [],
    'pressure_dt_retries': []
}

# =========================
# Simulation loop
# =========================
t_step = 0
time = 0.0
volume_L = volume_liquid_L  # Fed-batch working volume [L], matched to CFD liquid cells
est_steps = int(max(1, t_final / max(dt, 1e-9)))
while time < t_final and t_step < max(nt, est_steps*10):
    speed = np.sqrt(u**2 + v**2 + w**2)
    max_speed = np.max(speed) + 1e-6

    CFL_target = 0.1  # lower CFL target for smaller adaptive dt
    dt = CFL_target * dx / max_speed
    dt = max(dt_min, min(dt, dt_max))

    # Save rollback state in case a reduced dt retry is needed
    u_prev = u.copy()
    v_prev = v.copy()
    w_prev = w.copy()
    p_prev = p.copy()
    rho_prev = rho.copy()
    C_O2_prev = C_O2.copy()
    C_sub_prev = C_sub.copy()
    X_bio_prev = X_bio.copy()
    T_prev = T.copy()
    k_turb_prev = k_turb.copy()
    epsilon_turb_prev = epsilon_turb.copy()
    volume_prev = volume_L

    retry_count = 0
    step_success = False
    while True:
        # Fed-Batch dynamics
        dilution, substrate_in = fed_batch_source_terms(time, volume_L, C_O2, X_bio, C_sub)
        
        # Update volume
        feed_rate = get_feed_rate(time, fed_batch_strategy)
        volume_L += feed_rate * dt
        
        # K-Epsilon turbulence model: compute turbulent viscosity
        mu_t = compute_turbulent_viscosity(k_turb, epsilon_turb, rho, u, v, w)
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

        # Maintain zero provisional velocity in solid cells before pressure solve
        u_star[~mask] = 0.0
        v_star[~mask] = 0.0
        w_star[~mask] = 0.0
        
        D_O2 = 3e-9
        D_sub = 1e-9  # Substrate diffusivity [m²/s]

        C_sat = henry_saturation(T, P_gas)
        kLa_field = update_kLa_field(N_rpm)
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
        substrate_source = (substrate_in / volume_L) * dt if volume_L > 0 else 0  # [g/L] increment from feed
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

        # Diagnostic: check for NaNs/Infs before pressure solve
        if t_step < 5:
            for name, arr in [('u_star', u_star), ('v_star', v_star), ('w_star', w_star), ('rho', rho), ('mu_eff', mu_eff), ('k_turb', k_turb), ('epsilon_turb', epsilon_turb)]:
                try:
                    print(f"{name}: hasNaN={np.isnan(arr).any()}, hasInf={np.isinf(arr).any()}, min={np.nanmin(arr):.3e}, max={np.nanmax(arr):.3e}")
                except Exception as e:
                    print(f"{name}: diagnostic failed: {e}")
            div_star = divergence_mass(rho, u_star, v_star, w_star)
            print(f"div_star: min={np.min(div_star):.3e}, max={np.max(div_star):.3e}, mean={np.mean(div_star):.3e}")
            b_star = div_star / dt
            print(f"b_star: min={np.min(b_star):.3e}, max={np.max(b_star):.3e}, mean={np.mean(b_star):.3e}")
        p, pressure_converged, pressure_recovery_used = pressure_poisson(
            p, u_star, v_star, w_star, rho, k_turb
        )

        if not pressure_converged:
            if retry_count < max_step_retries and dt > dt_min:
                print(f"pressure solve failed at step {t_step}, reducing dt from {dt:.3e} to {max(dt*dt_reduction_factor, dt_min):.3e} and retrying")
                u = u_prev.copy()
                v = v_prev.copy()
                w = w_prev.copy()
                p = p_prev.copy()
                rho = rho_prev.copy()
                C_O2 = C_O2_prev.copy()
                C_sub = C_sub_prev.copy()
                X_bio = X_bio_prev.copy()
                T = T_prev.copy()
                k_turb = k_turb_prev.copy()
                epsilon_turb = epsilon_turb_prev.copy()
                volume_L = volume_prev
                dt = max(dt_min, dt * dt_reduction_factor)
                retry_count += 1
                continue
            else:
                print(f"ERROR: pressure solver failed after {retry_count+1} attempts at step {t_step}. Aborting simulation.")
                break

        # Cell-centered pressure projection using pressure gradients
        rho_safe = np.maximum(rho.astype(np.float64), 1e-8)
        grad_p_x = ddx(p.astype(np.float64))
        grad_p_y = ddy(p.astype(np.float64))
        grad_p_z = ddz(p.astype(np.float64))

        u_new = u_star.astype(np.float64) - dt * grad_p_x / rho_safe
        v_new = v_star.astype(np.float64) - dt * grad_p_y / rho_safe
        w_new = w_star.astype(np.float64) - dt * grad_p_z / rho_safe

        # Validate projection result before using it
        if (np.isnan(u_new).any() or np.isinf(u_new).any() or
            np.isnan(v_new).any() or np.isinf(v_new).any() or
            np.isnan(w_new).any() or np.isinf(w_new).any()):
            print(f"pressure projection produced invalid velocities at step {t_step}")
            if retry_count < max_step_retries and dt > dt_min:
                print(f"Retrying step {t_step} with smaller dt after invalid projection.")
                u = u_prev.copy()
                v = v_prev.copy()
                w = w_prev.copy()
                p = p_prev.copy()
                rho = rho_prev.copy()
                C_O2 = C_O2_prev.copy()
                C_sub = C_sub_prev.copy()
                X_bio = X_bio_prev.copy()
                T = T_prev.copy()
                k_turb = k_turb_prev.copy()
                epsilon_turb = epsilon_turb_prev.copy()
                volume_L = volume_prev
                dt = max(dt_min, dt * dt_reduction_factor)
                retry_count += 1
                continue
            else:
                print(f"ERROR: invalid projected velocities after {retry_count+1} attempts at step {t_step}. Aborting simulation.")
                step_success = False
                break

        # Apply mask and cast back
        u_new[~mask] = 0.0
        v_new[~mask] = 0.0
        w_new[~mask] = 0.0

        if t_step < 5:
            print(f"u_new stats: min={np.min(u_new):.3e}, max={np.max(u_new):.3e}, mean={np.mean(u_new):.3e}")
            print(f"v_new stats: min={np.min(v_new):.3e}, max={np.max(v_new):.3e}, mean={np.mean(v_new):.3e}")
            print(f"w_new stats: min={np.min(w_new):.3e}, max={np.max(w_new):.3e}, mean={np.mean(w_new):.3e}")

        u = u_new.astype(u.dtype)
        v = v_new.astype(v.dtype)
        w = w_new.astype(w.dtype)

        projected_div = divergence_mass(rho, u, v, w)
        projected_div_fluid = np.max(np.abs(projected_div[mask])) if np.any(mask) else 0.0
        if np.isnan(projected_div_fluid) or np.isinf(projected_div_fluid) or projected_div_fluid > 0.2:
            print(f"pressure projection divergence too large at step {t_step}: {projected_div_fluid:.3e}")
            if retry_count < max_step_retries and dt > dt_min:
                print(f"Retrying step {t_step} with smaller dt after large projected divergence.")
                u = u_prev.copy()
                v = v_prev.copy()
                w = w_prev.copy()
                p = p_prev.copy()
                rho = rho_prev.copy()
                C_O2 = C_O2_prev.copy()
                C_sub = C_sub_prev.copy()
                X_bio = X_bio_prev.copy()
                T = T_prev.copy()
                k_turb = k_turb_prev.copy()
                epsilon_turb = epsilon_turb_prev.copy()
                volume_L = volume_prev
                dt = max(dt_min, dt * dt_reduction_factor)
                retry_count += 1
                continue
            else:
                print(f"ERROR: projected divergence remains too large after {retry_count+1} attempts at step {t_step}. Aborting simulation.")
                step_success = False
                break

        if t_step < 5:
            projected_div = divergence_mass(rho, u, v, w)
            du = u - u_star
            dv = v - v_star
            dw = w - w_star
            print(f"pressure correction du: min={np.min(du):.3e}, max={np.max(du):.3e}")
            print(f"pressure correction dv: min={np.min(dv):.3e}, max={np.max(dv):.3e}")
            print(f"pressure correction dw: min={np.min(dw):.3e}, max={np.max(dw):.3e}")
            print(f"u_star: min={np.min(u_star):.3e}, max={np.max(u_star):.3e}")
            print(f"v_star: min={np.min(v_star):.3e}, max={np.max(v_star):.3e}")
            print(f"w_star: min={np.min(w_star):.3e}, max={np.max(w_star):.3e}")
            print(f"p: min={np.min(p):.3e}, max={np.max(p):.3e}")
            print(f"post-projection div: min={np.min(projected_div):.3e}, max={np.max(projected_div):.3e}, mean={np.mean(projected_div):.3e}")

        # Weakly compressible EOS with low-Mach preconditioning
        rho = apply_low_mach_density_eos(p, u, v, w, k_turb, rho)

        u,v,w = apply_bc(u,v,w)

        u *= 0.995
        v *= 0.995
        w *= 0.995
        
        # Update k-epsilon turbulence model (use density after pressure/EOS update)
        mu_t = compute_turbulent_viscosity(k_turb, epsilon_turb, rho, u, v, w)
        k_turb, epsilon_turb = update_k_epsilon_equations(
            k_turb, epsilon_turb, u, v, w, rho, mask_int
        )

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

        # Diagnostic: check for NaNs/Infs after pressure update
        if t_step < 5:
            for name, arr in [('u', u), ('v', v), ('w', w), ('rho', rho), ('p', p)]:
                try:
                    print(f"post-p solve {name}: hasNaN={np.isnan(arr).any()}, hasInf={np.isinf(arr).any()}, min={np.nanmin(arr):.3e}, max={np.nanmax(arr):.3e}")
                except Exception as e:
                    print(f"post-p solve {name}: diagnostic failed: {e}")
        div_field = divergence_mass(rho, u, v, w)
        div_fluid = np.max(np.abs(div_field[mask])) if np.any(mask) else 0.0
        div_global = np.max(np.abs(div_field))
        # Diagnostic prints: show shapes and stats for first steps or large fluid divergence
        if t_step < 5 or div_fluid > 0.1:
            try:
                # Check divergence of forcing terms (should be small)
                # Check force per unit mass for NaNs/Infs before differencing
                force_over_rho = Fx / rho
                try:
                    hasNa = np.isnan(force_over_rho)
                    print(f"Fx/rho: hasNaN={hasNa.any()}, hasInf={np.isinf(force_over_rho).any()}, min={np.nanmin(force_over_rho):.3e}, max={np.nanmax(force_over_rho):.3e}")
                    if hasNa.any():
                        idxs = np.argwhere(hasNa)
                        for ii in idxs[:5]:
                            i,j,k = ii
                            print(f"NaN at idx {i,j,k}: Fx={Fx[i,j,k]:.3e}, rho={rho[i,j,k]:.3e}")
                except Exception as e:
                    print("Fx/rho diagnostic failed:", e)
                div_force = ddx(force_over_rho) + ddy(Fy / rho) + ddz(Fz / rho)
                div_Fx = ddx(Fx) + ddy(Fy) + ddz(Fz)
                print(f"Div(force/rho) stats: min={np.nanmin(div_force):.3e}, max={np.nanmax(div_force):.3e}, mean={np.nanmean(div_force):.3e}")
                print(f"Div(Fx) stats: min={np.min(div_Fx):.3e}, max={np.max(div_Fx):.3e}, mean={np.mean(div_Fx):.3e}")
                print(f"STEP {t_step}: dt={dt:.3e}, dt_max={dt_max:.3e}, max_speed={max_speed:.3e}, div_fluid={div_fluid:.3e}, div_global={div_global:.3e}")
                print(f"Shapes: rho={rho.shape}, u={u.shape}, mask={mask.shape}, mask_int={mask_int.shape}, mu_eff={mu_eff.shape}")
                print(f"Div stats fluid: min={np.min(div_field[mask]):.3e}, max={np.max(div_field[mask]):.3e}, mean={np.mean(div_field[mask]):.3e}")
                print(f"Div stats global: min={np.min(div_field):.3e}, max={np.max(div_field):.3e}, mean={np.mean(div_field):.3e}")
                print(f"k_turb stats: min={np.min(k_turb):.3e}, max={np.max(k_turb):.3e}")
            except Exception as e:
                print("Diagnostic print failed:", e)
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
        history['pressure_recovery_used'].append(int(pressure_recovery_used))
        history['pressure_dt_retries'].append(retry_count)

        div_velocity = np.max(np.abs(divergence(u, v, w)[mask])) if np.any(mask) else 0.0
        div_mass_norm = np.max(np.abs(div_field[mask] / rho[mask])) if np.any(mask) else 0.0

        if np.isnan(div_velocity) or np.isinf(div_velocity) or div_velocity > 2.0:
            print(f"ERROR: divergence still too large after projection at step {t_step}: div_velocity={div_velocity:.2e}, div_mass_norm={div_mass_norm:.2e}")
            break

        time += dt
        t_step += 1
        step_success = True
        break

    if not step_success:
        print(f"ERROR: timestep {t_step} failed after {retry_count} retries. Aborting simulation.")
        break

print(f"\n=== Fed-Batch Bioreactor Simulation Complete ===")
print(f"Simulation finished after {t_step} steps, final time {time:.2f} s")
print(f"Final reactor volume: {volume_L:.2f} L")
print(f"Final biomass concentration: {np.mean(X_bio[mask]):.2f} g/L")
print(f"Final substrate concentration: {np.mean(C_sub[mask]):.2f} g/L")
print(f"Average density: {np.mean(rho[mask]):.2f} kg/m³")
print(f"Final temperature: {np.mean(T[mask]):.2f} °C")

# =========================
# Post-processing & spatial visualization
# =========================
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
TIME_LABEL = "Simulation time t [s]"


def masked_field(field):
    """Return field with solid/non-liquid cells set to NaN for plotting."""
    out = np.asarray(field, dtype=np.float64).copy()
    out[~mask] = np.nan
    return out


def save_figure(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_scalar_map(ax, field, extent, title, cbar_label, cmap):
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    return im


# Final spatial fields at end of simulation
C_sat_final = henry_saturation(T, P_gas)
kLa_final = update_kLa_field(N_rpm, len(history["time"]))
OTR_field = oxygen_transfer_rate(C_O2, C_sat_final, kLa_final)
speed_field = np.sqrt(u ** 2 + v ** 2 + w ** 2)

np.savez(
    os.path.join(OUTPUT_DIR, "final_fields.npz"),
    x=x_coords,
    y=y_coords,
    z=z_coords,
    mask=mask,
    X_bio=X_bio,
    C_sub=C_sub,
    C_O2=C_O2,
    T=T,
    OTR_local=OTR_field,
    kLa=kLa_final,
    speed=speed_field,
    u=u,
    v=v,
    w=w,
    p=p,
    rho=rho,
    history_time=np.array(history["time"]),
    history_OTR=np.array(history["OTR"]),
)

i_center, j_center = nx // 2, ny // 2
k_mid = nz // 2
j_mid = ny // 2
extent_xz = [x_coords[0], x_coords[-1], z_coords[0], z_coords[-1]]
extent_xy = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

with plt.rc_context(PLOT_STYLE):
    # --- Figure 1: vertical mid-plane maps (x-z through reactor center) ---
    fig_maps, axes_maps = plt.subplots(2, 3, figsize=(15, 8.5))
    fig_maps.suptitle(
        f"Spatial state fields at t = {time:.2f} s  |  "
        f"D = {diameter:.2f} m, H = {height:.2f} m, {n_impellers} impellers @ {N_rpm:.0f} rpm",
        fontsize=12,
        y=1.02,
    )
    spatial_maps = [
        (masked_field(X_bio)[:, j_mid, :], "Biomass concentration X [g/L]", "YlGn"),
        (masked_field(C_sub)[:, j_mid, :], "Substrate concentration S [g/L]", "Purples"),
        (masked_field(C_O2)[:, j_mid, :], "Dissolved O₂ concentration C_O₂ [mol/m³]", "plasma"),
        (masked_field(T)[:, j_mid, :], "Liquid temperature T [°C]", "inferno"),
        (masked_field(OTR_field)[:, j_mid, :], "Local OTR [mol/(m³·s)]", "Blues"),
        (masked_field(speed_field)[:, j_mid, :], "Flow speed |u| [m/s]", "turbo"),
    ]
    for ax, (data, cbar_label, cmap) in zip(axes_maps.ravel(), spatial_maps):
        plot_scalar_map(
            ax,
            data,
            extent_xz,
            title=cbar_label.split("[")[0].strip(),
            cbar_label=cbar_label,
            cmap=cmap,
        )
        ax.set_xlabel("Radial position x [m]")
        ax.set_ylabel("Axial position z [m]")
    save_figure(fig_maps, "spatial_fields_xz_midplane.png")

    # --- Figure 2: horizontal mid-height maps (x-y at impeller mid-plane) ---
    fig_xy, axes_xy = plt.subplots(2, 3, figsize=(15, 8.5))
    fig_xy.suptitle(
        f"Horizontal slice at z = {z_coords[k_mid]:.2f} m (mid liquid height)",
        fontsize=12,
        y=1.02,
    )
    horizontal_maps = [
        (masked_field(X_bio)[:, :, k_mid], "Biomass concentration X [g/L]", "YlGn"),
        (masked_field(C_sub)[:, :, k_mid], "Substrate concentration S [g/L]", "Purples"),
        (masked_field(C_O2)[:, :, k_mid], "Dissolved O₂ concentration C_O₂ [mol/m³]", "plasma"),
        (masked_field(T)[:, :, k_mid], "Liquid temperature T [°C]", "inferno"),
        (masked_field(OTR_field)[:, :, k_mid], "Local OTR [mol/(m³·s)]", "Blues"),
        (masked_field(speed_field)[:, :, k_mid], "Flow speed |u| [m/s]", "turbo"),
    ]
    for ax, (data, cbar_label, cmap) in zip(axes_xy.ravel(), horizontal_maps):
        plot_scalar_map(
            ax,
            data,
            extent_xy,
            title=cbar_label.split("[")[0].strip(),
            cbar_label=cbar_label,
            cmap=cmap,
        )
        ax.set_xlabel("Radial position x [m]")
        ax.set_ylabel("Radial position y [m]")
    save_figure(fig_xy, "spatial_fields_xy_midheight.png")

    # --- Figure 3: axial profiles along reactor centerline ---
    fig_axial, axes_axial = plt.subplots(2, 3, figsize=(15, 8.5))
    fig_axial.suptitle("Axial profiles along centerline (x = 0, y = 0)", fontsize=12, y=1.02)
    center_profiles = [
        (X_bio[i_center, j_center, :], "Biomass concentration X [g/L]", "tab:green"),
        (C_sub[i_center, j_center, :], "Substrate concentration S [g/L]", "tab:purple"),
        (C_O2[i_center, j_center, :], "Dissolved O₂ concentration C_O₂ [mol/m³]", "tab:orange"),
        (T[i_center, j_center, :], "Liquid temperature T [°C]", "tab:red"),
        (OTR_field[i_center, j_center, :], "Local OTR [mol/(m³·s)]", "tab:blue"),
        (speed_field[i_center, j_center, :], "Flow speed |u| [m/s]", "tab:gray"),
    ]
    for ax, (values, ylabel, color) in zip(axes_axial.ravel(), center_profiles):
        valid = mask[i_center, j_center, :]
        ax.plot(z_coords[valid], values[valid], color=color, linewidth=2)
        ax.set_xlabel("Axial position z [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("[")[0].strip())
    save_figure(fig_axial, "axial_profiles_centerline.png")

    # --- Figure 4: radial profiles at impeller-relevant heights ---
    impeller_z_indices = [int(0.30 * nz), int(0.50 * nz), int(0.65 * nz)]
    fig_radial, axes_radial = plt.subplots(2, 3, figsize=(15, 8.5))
    fig_radial.suptitle(
        "Radial profiles at mid impeller height "
        f"(z = {z_coords[impeller_z_indices[1]]:.2f} m)",
        fontsize=12,
        y=1.02,
    )
    k_imp = impeller_z_indices[1]
    r_line = R_coords[:, j_center, k_imp]
    radial_profiles = [
        (X_bio[:, j_center, k_imp], "Biomass concentration X [g/L]", "tab:green"),
        (C_sub[:, j_center, k_imp], "Substrate concentration S [g/L]", "tab:purple"),
        (C_O2[:, j_center, k_imp], "Dissolved O₂ concentration C_O₂ [mol/m³]", "tab:orange"),
        (T[:, j_center, k_imp], "Liquid temperature T [°C]", "tab:red"),
        (OTR_field[:, j_center, k_imp], "Local OTR [mol/(m³·s)]", "tab:blue"),
        (speed_field[:, j_center, k_imp], "Flow speed |u| [m/s]", "tab:gray"),
    ]
    for ax, (values, ylabel, color) in zip(axes_radial.ravel(), radial_profiles):
        valid = mask[:, j_center, k_imp]
        ax.plot(r_line[valid], values[valid], color=color, linewidth=2)
        ax.set_xlabel("Radial distance r [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel.split("[")[0].strip())
    save_figure(fig_radial, "radial_profiles_impeller_height.png")

    # --- Figure 5: integrated time-series dashboard ---
    fig_ts, axes_ts = plt.subplots(2, 3, figsize=(15, 8.5))
    fig_ts.suptitle(
        f"Transient reactor response  |  fed-batch strategy: {fed_batch_strategy}",
        fontsize=12,
        y=1.02,
    )

    axes_ts[0, 0].plot(history["time"], history["volume"], color="tab:blue", linewidth=2)
    axes_ts[0, 0].set_xlabel(TIME_LABEL)
    axes_ts[0, 0].set_ylabel("Liquid volume V [L]")
    axes_ts[0, 0].set_title("Fed-batch liquid volume")

    axes_ts[0, 1].plot(history["time"], history["X_bio_avg"], color="tab:green", linewidth=2, label="Volume average")
    axes_ts[0, 1].plot(history["time"], history["X_bio_max"], color="tab:green", linestyle="--", linewidth=1.5, label="Spatial maximum")
    axes_ts[0, 1].set_xlabel(TIME_LABEL)
    axes_ts[0, 1].set_ylabel("Biomass concentration X [g/L]")
    axes_ts[0, 1].set_title("Biomass concentration")
    axes_ts[0, 1].legend(frameon=False)

    axes_ts[0, 2].plot(history["time"], history["C_sub_avg"], color="tab:purple", linewidth=2, label="Volume average")
    axes_ts[0, 2].plot(history["time"], history["C_sub_max"], color="tab:purple", linestyle="--", linewidth=1.5, label="Spatial maximum")
    axes_ts[0, 2].set_xlabel(TIME_LABEL)
    axes_ts[0, 2].set_ylabel("Substrate concentration S [g/L]")
    axes_ts[0, 2].set_title("Substrate concentration")
    axes_ts[0, 2].legend(frameon=False)

    axes_ts[1, 0].plot(history["time"], history["C_O2_avg"], color="tab:orange", linewidth=2, label="Volume average")
    axes_ts[1, 0].plot(history["time"], history["C_O2_max"], color="tab:orange", linestyle="--", linewidth=1.5, label="Spatial maximum")
    axes_ts[1, 0].set_xlabel(TIME_LABEL)
    axes_ts[1, 0].set_ylabel("Dissolved O₂ concentration C_O₂ [mol/m³]")
    axes_ts[1, 0].set_title("Dissolved oxygen")
    axes_ts[1, 0].legend(frameon=False)

    axes_ts[1, 1].plot(history["time"], history["T_avg"], color="tab:red", linewidth=2, label="Volume average")
    axes_ts[1, 1].plot(history["time"], history["T_max"], color="tab:red", linestyle="--", linewidth=1.5, label="Spatial maximum")
    axes_ts[1, 1].set_xlabel(TIME_LABEL)
    axes_ts[1, 1].set_ylabel("Liquid temperature T [°C]")
    axes_ts[1, 1].set_title("Liquid temperature")
    axes_ts[1, 1].legend(frameon=False)

    axes_ts[1, 2].plot(history["time"], history["OTR"], color="tab:blue", linewidth=2)
    axes_ts[1, 2].set_xlabel(TIME_LABEL)
    axes_ts[1, 2].set_ylabel("Total OTR [mol/s]")
    axes_ts[1, 2].set_title("Oxygen transfer rate (volume integral)")

    save_figure(fig_ts, "transient_timeseries.png")

summary_path = os.path.join(OUTPUT_DIR, "simulation_summary.txt")
with open(summary_path, "w", encoding="utf-8") as summary_file:
    summary_file.write("Weakly Compressible Bioreactor Simulation Summary\n")
    summary_file.write("=" * 52 + "\n")
    summary_file.write(f"Simulation time: {time:.2f} s ({t_step} steps)\n")
    summary_file.write(f"Geometry: D={diameter:.3f} m, H={height:.3f} m, V={volume_target:.3f} m^3\n")
    summary_file.write(f"Grid: {nx} x {ny} x {nz} cells\n")
    summary_file.write(f"Impellers: {n_impellers} @ {N_rpm:.0f} rpm\n")
    summary_file.write(f"Fed-batch: {fed_batch_strategy}, start={fed_batch_start_time:.1f} s\n")
    summary_file.write(f"Final liquid volume: {volume_L:.2f} L\n")
    summary_file.write(f"Volume-averaged biomass X: {np.mean(X_bio[mask]):.3f} g/L\n")
    summary_file.write(f"Volume-averaged substrate S: {np.mean(C_sub[mask]):.3f} g/L\n")
    summary_file.write(f"Volume-averaged O2 C_O2: {np.mean(C_O2[mask]):.4f} mol/m^3\n")
    summary_file.write(f"Volume-averaged temperature T: {np.mean(T[mask]):.3f} C\n")
    summary_file.write(f"Total OTR (final step): {history['OTR'][-1]:.4e} mol/s\n")
    summary_file.write(f"Peak local OTR: {np.max(OTR_field[mask]):.4e} mol/(m^3 s)\n")
    summary_file.write(f"Peak flow speed: {np.max(speed_field[mask]):.4f} m/s\n")

print(f"\nSpatial outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"  - spatial_fields_xz_midplane.png")
print(f"  - spatial_fields_xy_midheight.png")
print(f"  - axial_profiles_centerline.png")
print(f"  - radial_profiles_impeller_height.png")
print(f"  - transient_timeseries.png")
print(f"  - final_fields.npz")
print(f"  - simulation_summary.txt")

plt.show()


