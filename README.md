# Matsuno Shallow Water Model

Implementation of Matsuno's (1966) linearized shallow water equations on an equatorial beta plane for studying Kelvin and Rossby wave dynamics.

## Overview

This model solves the linearized shallow water equations:
- ∂u/∂t - βy·v = -g·∂η/∂x
- ∂v/∂t + βy·u = -g·∂η/∂y  
- ∂η/∂t = -H·(∂u/∂x + ∂v/∂y)

Where:
- u, v: zonal and meridional velocities
- η: surface height perturbation
- β: beta parameter (2Ω/R at equator)
- H: equivalent depth
- g: gravitational acceleration

## Features

- **Equatorial Beta Plane**: Proper f = βy approximation
- **Kelvin Waves**: Eastward-propagating, equatorially-trapped
- **Rossby Waves**: Westward-propagating, off-equatorial maxima
- **Leapfrog Integration**: With Robert-Asselin filter for stability
- **Multiple Initializations**: Kelvin, Rossby, and Gaussian perturbations
- **Atmospheric/Ocean Ready**: Configurable for different equivalent depths

## Quick Start

```python
from matsuno_shallow_water import create_matsuno_model

# Create atmospheric model (H = 25m)
model = create_matsuno_model(
    equivalent_depth=25.0,  # m
    dt_minutes=5.0,         # time step
    nt=288                  # 1 day simulation
)

# Initialize Kelvin wave
initial_state = model.initialize_kelvin_wave(amplitude=10.0)

# Run simulation
states = model.integrate(initial_state)

# Convert to xarray for analysis
dataset = model.to_xarray(states)
```

## Files

- `matsuno_shallow_water.py`: Main model implementation
- `atmospheric_waves_demo.ipynb`: **Atmospheric applications** (MJO, CCEWs, tropical meteorology)  
- `oceanic_waves_demo.ipynb`: **Oceanic applications** (ENSO, tropical Pacific dynamics)
- `matsuno_examples.ipynb`: General examples and technical analysis
- `test_matsuno.py`: Basic functionality tests
- `requirements.txt`: Python dependencies

## Parameters

### Atmospheric Configuration (Default)
- Equivalent depth: 25 m (first baroclinic mode)
- Wave speed: ~15.7 m/s
- Recommended time step: ≤ 10 minutes (CFL < 0.2)

### Ocean Configuration
- Equivalent depth: 1-5 m (baroclinic modes)  
- Wave speed: ~3-7 m/s
- Recommended time step: ≤ 15 minutes

## Wave Characteristics

### Kelvin Waves
- **Speed**: c = √(gH)
- **Structure**: Equatorially trapped, e^(-y²/2L²)
- **Scale**: L = √(c/β) ≈ 800-1500 km
- **Properties**: u,η in phase; v = 0

### Rossby Waves  
- **Speed**: Much slower than Kelvin waves
- **Structure**: Off-equatorial maxima
- **Dispersion**: ω = -βk/(k² + (2n+1)β/c)
- **Properties**: Westward propagating

## Stability

The model uses leapfrog time integration with Robert-Asselin filtering. Stability requires:
- CFL = c·dt/dx < 1.0 (preferably < 0.5)
- Small Robert-Asselin coefficient (0.01-0.05)

## Usage Examples

### 1. Basic Kelvin Wave
```python
model = create_matsuno_model(dt_minutes=5.0, nt=576)  # 2 days
kelvin_state = model.initialize_kelvin_wave(amplitude=10.0, wavelength=2000e3)
results = model.integrate(kelvin_state)
```

### 2. Rossby Wave
```python
rossby_state = model.initialize_rossby_wave(amplitude=5.0, mode=1)
results = model.integrate(rossby_state)
```

### 3. Wave Decomposition
```python
gaussian_state = model.initialize_gaussian_perturbation(amplitude=8.0)
results = model.integrate(gaussian_state)  # Shows both Kelvin and Rossby waves
```

### 4. Ocean Configuration
```python
ocean_model = create_matsuno_model(
    equivalent_depth=2.0,   # Ocean scale
    dt_minutes=10.0,
    nt=432  # 3 days
)
```

## Validation

The model correctly reproduces:
- Kelvin wave eastward propagation at speed c
- Equatorial trapping with scale L_eq
- Rossby wave westward propagation
- Energy conservation (within numerical precision)
- Wave dispersion relationships

## References

- Matsuno, T. (1966). Quasi-geostrophic motions in the equatorial area. Journal of the Meteorological Society of Japan, 44(1), 25-43.
- Gill, A. E. (1982). Atmosphere-Ocean Dynamics. Academic Press.
- Wheeler, M., & Kiladis, G. N. (1999). Convectively coupled equatorial waves. Reviews of Geophysics, 37(3), 275-298.

## Testing

Run the test suite:
```bash
python test_matsuno.py
```

## Demonstration Notebooks

### Atmospheric Applications
```bash
jupyter notebook atmospheric_waves_demo.ipynb
```
- **Focus**: MJO, convectively coupled equatorial waves (CCEWs), tropical meteorology
- **Parameters**: H = 25m, fast wave speeds (15+ m/s), hourly-daily time scales
- **Examples**: Kelvin wave MJO propagation, Rossby wave tropical cyclogenesis, convective coupling

### Oceanic Applications  
```bash
jupyter notebook oceanic_waves_demo.ipynb
```
- **Focus**: ENSO dynamics, tropical Pacific variability, ocean-climate coupling
- **Parameters**: H = 2.5m, slower wave speeds (~5 m/s), weekly-monthly time scales  
- **Examples**: El Niño Kelvin waves, La Niña Rossby adjustment, trans-Pacific propagation

### General Technical Examples
```bash
jupyter notebook matsuno_examples.ipynb
```
- **Focus**: Model validation, stability analysis, parameter sensitivity
- **Content**: Wave theory verification, numerical methods, comparative analysis