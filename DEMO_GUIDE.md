# Matsuno Model Demonstration Guide

---

## Generic Examples (`matsuno_examples.ipynb`)

- Gaussian Perturbation (Kelvin and Rossby Waves)

---

## Atmospheric Waves (`atmospheric_waves_demo.ipynb`)

- **Wave speeds**: 15-20 m/s (fast atmospheric dynamics)
- **Time scales**: Hours to days
- **Equivalent depth**: 25m (first baroclinic mode)

### Recommended Usage:
```python
# Atmospheric configuration
atm_model = create_matsuno_model(
    equivalent_depth=25.0,  # 25 m - typical for atmospheric convection
    dt_minutes=4.0,         # 4 minute time step for stability
    nt=720                  # 2 days simulation (48 hours)
)
```

---

## Oceanic Waves (`oceanic_waves_demo.ipynb`)

- **Wave speeds**: 2-5 m/s (slow oceanic dynamics)
- **Time scales**: Weeks to months  
- **Equivalent depth**: 2.5m (first baroclinic mode)

### Recommended Usage:
```python
# oceanic model - 
ocean_model = create_matsuno_model(
    equivalent_depth=2.5,   # 2.5 m - first baroclinic mode
    dt_minutes=60.0,        # 1 hour time step 
    nt=24*15                # 15 days simulation
)
```

## Quick Start Guide

### 1. Atmosphere
```bash
jupyter notebook atmospheric_waves_demo.ipynb
```

### 2. Ocean  
```bash
jupyter notebook oceanic_waves_demo.ipynb
```

### 3. General:
```bash
jupyter notebook matsuno_examples.ipynb
```

## Configuration Comparison

| Application | H (m) | c (m/s) | L_eq (km) | Time Step | Time Scales |
|-------------|-------|---------|-----------|-----------|-------------|
| **Atmosphere** | 25 | 15.7 | 827 | 4 min | Hours-days |
| **Ocean** | 2.5 | 4.9 | 465 | 15 min | Weeks-months |

## Need Help?

1. **Basic functionality**: Run `python test_matsuno.py`
2. **Installation issues**: Check `requirements.txt`
3. **Theory questions**: See `README.md` references
4. **Applications**: Start with the appropriate demo notebook above