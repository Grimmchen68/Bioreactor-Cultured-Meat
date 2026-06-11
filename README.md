# Bioreactor-Cultured-Meat
Bioreactor and CFD of Cultured Meat 
A Simulation of a Bioreactor with the Parameter of Cultured Meat. This Code in 100% Python simulates a 20 kubik meter bioreactor with three Rushton impeller.


This project contains several Python files for simplified CFD and bioreactor simulations.

## Model variants

- `Bioreactor_OTR_model.py`
  - Original model
  - Liquid is treated as incompressible
  - Constant density `rho = 1000.0`
  - Pressure correction is based on `div(u,v,w) ≈ 0`

- `Bioreactor_OTR_model_compressible.py`
  - New variant with weakly compressible liquid
  - Density is stored as a field `rho(x,y,z)`
  - Pressure-to-density relation: `rho = rho0 * (1 + (p - P_atm) / K_bulk)`
  - Uses mass-conserving divergence `div(rho * u)` instead of `div(u)`

## Notes

- `Bioreactor_OTR_model_compressible.py` is not a fully compressible gas/liquid solver.
- It is a demo for a weakly compressible liquid with a high bulk modulus.
- Real high-fidelity compressible simulations require much more complex models and numerical methods.

# Fed-Batch Bioreactor Simulation - Implementation Guide

## Overview
The `Bioreactor_OTR_model_compressible.py` has been extended from a **Batch Process** to a **Fed-Batch Process**. This guide explains the new features and how to customize them.

## Key Modifications

### 1. New Parameters (Lines ~60-70)
```python
# Fed-Batch parameters
fed_batch_start_time = 10.0           # When to start feeding [s]
fed_batch_strategy = "linear"         # Feed strategy: "linear", "exponential", "do_stat"
feed_rate = 0.05                      # Feed rate [L/s]
substrate_conc_feed = 400.0           # Substrate concentration in feed [g/L]
initial_volume = 10.0                 # Initial liquid volume [L]
```

**Customize these for your desired process:**
- `fed_batch_start_time`: Delay before feeding starts (batch phase)
- `fed_batch_strategy`: Choose feeding strategy:
  - **"linear"**: Constant feeding rate
  - **"exponential"**: Exponential feeding for constant growth rate
  - **"do_stat"**: Dissolved oxygen-stat control (simplified)
- `substrate_conc_feed`: Glucose/carbon source concentration in feed
- `initial_volume`: Initial liquid volume before feeding

### 2. New Fields Added to Simulation

#### Substrate Field (C_sub)
- Tracks glucose/carbon source concentration [g/L]
- Initialized to ~50 g/L
- Consumed during cell growth
- Added during fed-batch feeding

#### Updated Biomass Model
- Changed from passive oxygen consumption model to **active growth model**
- Biomass now grows based on substrate and oxygen availability
- Uses **Monod kinetics** for both substrate and oxygen

### 3. New Functions

#### `get_feed_rate(time, strategy)`
Calculates feeding rate based on selected strategy.
- Returns 0 before `fed_batch_start_time`
- Options for linear, exponential, or DO-stat feeding

#### `fed_batch_source_terms(time, volume_L, C_O2, X_bio, C_sub)`
Calculates dilution and substrate addition due to feeding.
- **Returns:** dilution factor [1/s] and substrate input [g/s]
- Accounts for volume change and nutrient addition

#### `growth_kinetics(C_O2_local, C_sub_local, X_local, T_local)`
Calculates biomass growth and nutrient consumption based on Monod kinetics.
- **Returns:**
  - `mu_specific`: Specific growth rate [1/s]
  - `dS_dt_growth`: Substrate consumption rate [g/L/s]
  - `dO2_dt_growth`: Oxygen consumption for growth [mol/m³/s]

### 4. Modified Equations in Main Loop

#### Substrate Transport & Consumption
```python
# Substrate equation with convection, diffusion, consumption, and dilution
C_sub = C_sub + dt * (
    -conv_sub + diff_sub 
    - dS_dt_growth        # Consumed during growth
    - dilution * C_sub    # Diluted by feeding
) + substrate_source
```

#### Biomass Growth
```python
# Biomass growth based on Monod kinetics + dilution
X_bio = X_bio + dt * (
    -conv_X 
    + mu_specific * X_bio   # Growth term
    - dilution * X_bio      # Dilution by feeding
)
```

#### Oxygen Balance
```python
# Now includes both transfer AND consumption for growth
C_O2 = C_O2 + dt * (
    -conv_O2 + diff_O2 + OTR_local 
    - dO2_dt_growth       # Growth consumption
    - dilution * C_O2     # Dilution by feeding
)
```

## Kinetic Parameters (Lines ~62-67)

```python
mu_max = 0.35           # Maximum growth rate [1/h]
K_m_sub = 0.5           # Substrate half-saturation [g/L]
K_m_O2 = 6.0            # Oxygen half-saturation [mol/m³]
Y_XS = 0.5              # Yield: grams biomass per g substrate
Y_OS = 1.0              # Oxygen yield on substrate [g O2 / g substrate]
X_biomass_init = 5.0    # Initial biomass concentration [g/L]
```

**Tips for customization:**
- Higher `mu_max` = faster growth
- Lower `K_m_sub` = higher substrate affinity (faster at low S)
- Adjust `Y_XS` and `Y_OS` based on experimental data

## Typical Fed-Batch Scenarios

### Scenario 1: Linear Feeding (Constant Feed Rate)
```python
fed_batch_start_time = 10.0
fed_batch_strategy = "linear"
feed_rate = 0.05           # L/s = 50 mL/s = 3 L/min
```
**Use for:** Simple process, gradual volume increase, controlled substrate addition

### Scenario 2: Exponential Feeding (Constant Growth Rate)
```python
fed_batch_start_time = 10.0
fed_batch_strategy = "exponential"
feed_rate = 0.05           # Will grow exponentially
```
**Use for:** Maximize biomass production at maximum growth rate

### Scenario 3: DO-Stat (Oxygen-Limited Growth)
```python
fed_batch_start_time = 10.0
fed_batch_strategy = "do_stat"
```
**Use for:** Maintain dissolved oxygen at target level (simplified implementation)

## Monitoring Output

The simulation now tracks and plots:
- **Volume**: Increasing volume profile during fed-batch
- **Biomass (X)**: Growth curve showing exponential growth followed by stationary/decline
- **Substrate (S)**: Substrate depletion and replenishment
- **Oxygen (DO)**: Oxygen dynamics with transfer and consumption
- **Temperature**: Biological heat generation during growth
- **OTR**: Oxygen transfer rate (increases with oxygen demand)

## Example Output Interpretation

A typical fed-batch process shows:
1. **Batch Phase (0-10s)**: Initial growth on substrate, biomass increases
2. **Feeding Phase (10s+)**: Substrate added continuously, volume increases
3. **Growth Phase**: Biomass exponentially increases if O₂ and S are available
4. **Stationary/Decline**: If O₂ becomes limiting or substrate depletes faster

## How to Modify Parameters for Your Study

### To simulate faster-growing organism:
- Increase `mu_max` (e.g., 0.5-0.7 /h)
- Decrease `K_m_sub` (e.g., 0.1-0.3)

### To simulate substrate-limited growth:
- Reduce `feed_rate` or increase `fed_batch_start_time`
- Increase `K_m_sub`

### To simulate oxygen-limited growth:
- Check `K_m_O2` and aeration `Q_air`
- Reduce impeller speed `N_rpm`

### To simulate longer fed-batch:
- Increase `t_final` (simulation duration)
- Adjust `substrate_conc_feed` (higher concentration allows longer operation)

## Notes

- **Dilution factor** is calculated as: `dilution = feed_rate / volume_L [1/s]`
- All concentrations are automatically diluted during feeding
- Temperature effects are included (`k_T = exp(0.1 * (T - T_ref))`)
- The simulation maintains mass balance during feeding
- Oxygen transfer (`kLa`) is impeller-speed dependent

## Troubleshooting

**Issue**: Biomass not growing
- Check: Is substrate being added? (`fed_batch_start_time`, `feed_rate`)
- Check: Is oxygen available? (See `C_O2_avg`, should be > K_m_O2 ≈ 6 mol/m³)

**Issue**: Volume not increasing
- Check: `fed_batch_start_time` and `feed_rate` parameters
- Check: Current simulation `time` > `fed_batch_start_time`

**Issue**: Substrate depletion too fast
- Increase `Y_XS` (lower consumption) or reduce `feed_rate`

**Issue**: Oxygen becomes limiting
- Increase `N_rpm` (impeller speed) for better aeration
- Increase `Q_air` (aeration rate)
