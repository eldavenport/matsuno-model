# Matsuno Model Demonstration Guide

## 🌪️ Atmospheric Waves (`atmospheric_waves_demo.ipynb`)

- **Wave speeds**: 15-20 m/s (fast atmospheric dynamics)
- **Time scales**: Hours to days
- **Equivalent depth**: 25m (first baroclinic mode)

### Examples Include:
1. **MJO-like Kelvin waves** - Eastward convective propagation
2. **Tropical cyclogenesis** - Rossby wave energy accumulation  
3. **Convective coupling** - Wave-convection interaction simulation
4. **Parameter sensitivity** - Different atmospheric modes (H = 12-100m)
5. **Stability analysis** - Time step requirements for fast dynamics

### Recommended Usage:
```python
# Atmospheric configuration
model = create_matsuno_model(
    equivalent_depth=25.0,  # 1st baroclinic mode
    dt_minutes=4.0,         # Fast time step
    nt=720                  # 2 days
)
```

---

## 🌊 Oceanic Waves (`oceanic_waves_demo.ipynb`)

- **Wave speeds**: 2-5 m/s (slow oceanic dynamics)
- **Time scales**: Weeks to months  
- **Equivalent depth**: 2.5m (first baroclinic mode)

### Examples Include:
1. **El Niño Kelvin waves** - Trans-Pacific SSH propagation
2. **La Niña Rossby adjustment** - Basin-wide ocean response
3. **ENSO wind stress response** - Central Pacific forcing simulation
4. **Baroclinic mode comparison** - Different ocean stratifications
5. **ENSO time scale matching** - Realistic 2-7 year variability

### Recommended Usage:
```python
# Oceanic configuration  
model = create_matsuno_model(
    equivalent_depth=2.5,   # 1st baroclinic mode
    dt_minutes=15.0,        # Stable oceanic time step
    nt=1440                 # 15 days
)
```

---

## 🔬 Technical Analysis (`matsuno_examples.ipynb`)

- Mathematical verification and numerical analysis
- Wave theory, dispersion relations, stability criteria

### Examples Include:
1. **Wave theory verification** - Speed and structure validation
2. **Stability analysis** - CFL conditions and time step limits
3. **Energy conservation** - Numerical accuracy assessment
4. **Multi-mode comparison** - Different equivalent depths
5. **Computational performance** - Optimization strategies

---

## Quick Start Guide

### 1. Atmosphere
```bash
jupyter notebook atmospheric_waves_demo.ipynb
```

### 2. Ocean  
```bash
jupyter notebook oceanic_waves_demo.ipynb
```

### 3. Extra:
```bash
jupyter notebook matsuno_examples.ipynb
```

## Configuration Comparison

| Application | H (m) | c (m/s) | L_eq (km) | Time Step | Time Scales |
|-------------|-------|---------|-----------|-----------|-------------|
| **Atmosphere** | 25 | 15.7 | 827 | 4 min | Hours-days |
| **Ocean** | 2.5 | 4.9 | 465 | 15 min | Weeks-months |
| **General** | Variable | Variable | Variable | Adaptive | Research-dependent |

## Need Help?

1. **Basic functionality**: Run `python test_matsuno.py`
2. **Installation issues**: Check `requirements.txt`
3. **Theory questions**: See `README.md` references
4. **Applications**: Start with the appropriate demo notebook above

---

**Happy modeling!** 🌊🌪️

Choose your research focus and dive into the corresponding demonstration notebook.