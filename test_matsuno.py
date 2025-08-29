#!/usr/bin/env python3
"""
Quick test script for the Matsuno shallow water model.
"""

from matsuno_shallow_water import create_matsuno_model
import numpy as np
import matplotlib.pyplot as plt

def test_basic_functionality():
    """Test basic model functionality."""
    print("=== Testing Matsuno Model ===")
    
    # Create model
    model = create_matsuno_model(
        equivalent_depth=25.0,  # Atmospheric scale
        dt_minutes=5.0,
        nt=144  # 12 hours
    )
    
    print(f"✓ Model created: {model.config.nlat}×{model.config.nlon} grid")
    print(f"  Wave speed: {model.c:.1f} m/s")
    print(f"  Deformation radius: {model.L_eq/1000:.0f} km") 
    print(f"  CFL numbers: {model.cfl_x:.3f}, {model.cfl_y:.3f}")
    
    # Test Kelvin wave
    kelvin_initial = model.initialize_kelvin_wave(amplitude=5.0)
    kelvin_states = model.integrate(kelvin_initial)
    
    print(f"✓ Kelvin wave simulation: {kelvin_states.shape}")
    
    # Test Rossby wave
    rossby_initial = model.initialize_rossby_wave(amplitude=3.0, mode=1)
    rossby_states = model.integrate(rossby_initial)
    
    print(f"✓ Rossby wave simulation: {rossby_states.shape}")
    
    # Test Gaussian perturbation
    gaussian_initial = model.initialize_gaussian_perturbation(amplitude=4.0)
    gaussian_states = model.integrate(gaussian_initial)
    
    print(f"✓ Gaussian perturbation: {gaussian_states.shape}")
    
    # Check for numerical issues
    has_nans = any([
        np.isnan(kelvin_states).any(),
        np.isnan(rossby_states).any(), 
        np.isnan(gaussian_states).any()
    ])
    
    if has_nans:
        print("✗ NaN detected in simulations!")
        return False
    else:
        print("✓ All simulations clean (no NaNs)")
    
    # Test xarray conversion
    ds = model.to_xarray(kelvin_states)
    print(f"✓ XArray conversion: {len(ds.time)} time steps")
    
    return True

def test_stability():
    """Test model stability with different time steps."""
    print("\n=== Testing Stability ===")
    
    dt_tests = [2.0, 5.0, 10.0, 15.0]  # minutes
    
    for dt_min in dt_tests:
        try:
            test_model = create_matsuno_model(
                equivalent_depth=25.0,
                dt_minutes=dt_min,
                nt=60  # 1 hour
            )
            
            # Simple test
            initial = test_model.initialize_kelvin_wave(amplitude=1.0)
            states = test_model.integrate(initial)
            
            # Check growth
            initial_max = np.max(np.abs(states[0]))
            final_max = np.max(np.abs(states[-1]))
            growth = final_max / initial_max if initial_max > 0 else np.inf
            
            cfl_max = max(test_model.cfl_x, test_model.cfl_y)
            stable = not np.isnan(final_max) and growth < 2.0
            
            status = "✓" if stable else "✗"
            print(f"{status} dt={dt_min:4.1f}min, CFL={cfl_max:.3f}, growth={growth:.2f}")
            
        except Exception as e:
            print(f"✗ dt={dt_min:4.1f}min FAILED: {str(e)[:40]}...")

if __name__ == "__main__":
    success = test_basic_functionality()
    test_stability()
    
    if success:
        print(f"\n🎉 Matsuno model ready for use!")
        print(f"   Run 'jupyter notebook matsuno_examples.ipynb' to see examples")
    else:
        print(f"\n❌ Model tests failed")