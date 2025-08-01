"""
Matsuno's Shallow Water Equations on Equatorial Beta Plane

Implementation of the linearized shallow water equations following Matsuno (1966)
for equatorial wave dynamics. Initially configured for atmospheric applications
with future ocean compatibility in mind.

References:
- Matsuno, T. (1966). Quasi-geostrophic motions in the equatorial area. 
  Journal of the Meteorological Society of Japan, 44(1), 25-43.
- Gill, A. E. (1982). Atmosphere-Ocean Dynamics. Academic Press.
"""

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MatsunoConfig:
    """Configuration parameters for Matsuno shallow water model."""
    
    # Domain parameters
    lat_min: float = -15.0      # Southern boundary (degrees)
    lat_max: float = 15.0       # Northern boundary (degrees)
    lon_min: float = 0.0        # Western boundary (degrees)
    lon_max: float = 120.0      # Eastern boundary (degrees)
    nlat: int = 61              # Number of latitude points
    nlon: int = 241             # Number of longitude points
    
    # Physical parameters
    g: float = 9.81             # Gravitational acceleration (m/s²)
    earth_radius: float = 6.371e6  # Earth radius (m)
    omega: float = 7.292e-5     # Earth rotation rate (rad/s)
    
    # Atmospheric stratification (default - can be modified for ocean)
    equivalent_depth: float = 25.0  # Equivalent depth (m) - atmospheric first mode
    
    # Beta plane parameters (computed)
    beta: float = None          # Will be computed: 2*omega*cos(lat_ref)/R
    lat_ref: float = 0.0        # Reference latitude for beta plane (equator)
    
    # Time stepping
    dt: float = 300.0           # Time step (seconds) - 5 minutes for atmosphere
    nt: int = 288               # Number of time steps (1 day)
    
    # Numerical parameters
    robert_filter: float = 0.01  # Robert-Asselin filter coefficient
    
    def __post_init__(self):
        """Compute derived parameters."""
        if self.beta is None:
            # Beta parameter at equator: β = 2Ω cos(φ)/R ≈ 2Ω/R at equator
            self.beta = 2 * self.omega / self.earth_radius


class MatsunoModel:
    """Matsuno shallow water model on equatorial beta plane."""
    
    def __init__(self, config: MatsunoConfig):
        self.config = config
        self._setup_grid()
        self._compute_parameters()
    
    def _setup_grid(self):
        """Set up the computational grid."""
        # Latitude and longitude arrays
        self.lat = jnp.linspace(self.config.lat_min, self.config.lat_max, self.config.nlat)
        self.lon = jnp.linspace(self.config.lon_min, self.config.lon_max, self.config.nlon)
        
        # Grid spacing
        self.dlat = (self.config.lat_max - self.config.lat_min) / (self.config.nlat - 1)
        self.dlon = (self.config.lon_max - self.config.lon_min) / (self.config.nlon - 1)
        
        # Convert to meters for derivatives
        self.dy = self.dlat * jnp.pi / 180 * self.config.earth_radius
        self.dx = self.dlon * jnp.pi / 180 * self.config.earth_radius
        
        # Create 2D coordinate grids
        self.lon_2d, self.lat_2d = jnp.meshgrid(self.lon, self.lat, indexing='ij')
        
        # Distance from equator (in meters) for beta plane
        self.y_2d = (self.lat_2d - self.config.lat_ref) * jnp.pi / 180 * self.config.earth_radius
    
    def _compute_parameters(self):
        """Compute model parameters."""
        # Wave speed
        self.c = jnp.sqrt(self.config.g * self.config.equivalent_depth)
        
        # Coriolis parameter on beta plane: f = β*y
        self.f_2d = self.config.beta * self.y_2d
        
        # Equatorial deformation radius
        self.L_eq = jnp.sqrt(self.c / self.config.beta)
        
        # Non-dimensional parameters for stability analysis
        self.cfl_x = self.c * self.config.dt / self.dx
        self.cfl_y = self.c * self.config.dt / self.dy
    
    def tendency(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Compute tendencies for Matsuno's shallow water equations.
        
        Equations (linearized):
        ∂u/∂t - βy*v = -g*∂η/∂x
        ∂v/∂t + βy*u = -g*∂η/∂y  
        ∂η/∂t = -H*(∂u/∂x + ∂v/∂y)
        
        Args:
            state: [u, v, η] with shape (3, nlat, nlon)
            
        Returns:
            tendencies: [du/dt, dv/dt, dη/dt] with same shape
        """
        u, v, eta = state[0], state[1], state[2]
        
        # Compute spatial derivatives using centered differences
        # Zonal derivatives (∂/∂x)
        du_dx = self._derivative_x(u)
        deta_dx = self._derivative_x(eta)
        
        # Meridional derivatives (∂/∂y) 
        dv_dy = self._derivative_y(v)
        deta_dy = self._derivative_y(eta)
        
        # Matsuno's shallow water equations
        du_dt = self.config.beta * self.y_2d.T * v - self.config.g * deta_dx
        dv_dt = -self.config.beta * self.y_2d.T * u - self.config.g * deta_dy
        deta_dt = -self.config.equivalent_depth * (du_dx + dv_dy)
        
        return jnp.stack([du_dt, dv_dt, deta_dt])
    
    def _derivative_x(self, field: jnp.ndarray) -> jnp.ndarray:
        """Compute zonal derivative using centered differences."""
        # Periodic boundary conditions in longitude
        df_dx = jnp.zeros_like(field)
        df_dx = df_dx.at[:, 1:-1].set((field[:, 2:] - field[:, :-2]) / (2 * self.dx))
        df_dx = df_dx.at[:, 0].set((field[:, 1] - field[:, -1]) / (2 * self.dx))
        df_dx = df_dx.at[:, -1].set((field[:, 0] - field[:, -2]) / (2 * self.dx))
        return df_dx
    
    def _derivative_y(self, field: jnp.ndarray) -> jnp.ndarray:
        """Compute meridional derivative using centered differences."""
        # Zero gradient boundary conditions at north/south boundaries
        df_dy = jnp.zeros_like(field)
        df_dy = df_dy.at[1:-1, :].set((field[2:, :] - field[:-2, :]) / (2 * self.dy))
        df_dy = df_dy.at[0, :].set((field[1, :] - field[0, :]) / self.dy)
        df_dy = df_dy.at[-1, :].set((field[-1, :] - field[-2, :]) / self.dy)
        return df_dy
    
    def leapfrog_step(self, state_prev: jnp.ndarray, state_curr: jnp.ndarray) -> jnp.ndarray:
        """
        Perform one leapfrog time step with Robert-Asselin filter.
        
        Args:
            state_prev: State at time n-1
            state_curr: State at time n
            
        Returns:
            state_next: State at time n+1
        """
        # Compute tendencies at current time
        tendency_curr = self.tendency(state_curr)
        
        # Leapfrog step: state(n+1) = state(n-1) + 2*dt*tendency(n)
        state_next = state_prev + 2 * self.config.dt * tendency_curr
        
        # Apply Robert-Asselin filter to suppress computational mode
        # state(n) = state(n) + ε*(state(n-1) - 2*state(n) + state(n+1))
        filter_coeff = self.config.robert_filter
        state_filtered = state_curr + filter_coeff * (state_prev - 2*state_curr + state_next)
        
        return state_next, state_filtered
    
    def forward_euler_step(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Perform one forward Euler step (for initialization).
        
        Args:
            state: Current state
            
        Returns:
            new_state: State after one time step
        """
        tendency = self.tendency(state)
        return state + self.config.dt * tendency
    
    def integrate(self, initial_state: jnp.ndarray) -> jnp.ndarray:
        """
        Integrate the model forward in time.
        
        Args:
            initial_state: Initial conditions [u, v, η] with shape (3, nlat, nlon)
            
        Returns:
            states: Time series of states with shape (nt+1, 3, nlat, nlon)
        """
        # Initialize storage
        states = jnp.zeros((self.config.nt + 1, 3, self.config.nlat, self.config.nlon))
        states = states.at[0].set(initial_state)
        
        # First step using forward Euler
        state_1 = self.forward_euler_step(initial_state)
        states = states.at[1].set(state_1)
        
        # Subsequent steps using leapfrog
        state_prev = initial_state
        state_curr = state_1
        
        for i in range(2, self.config.nt + 1):
            state_next, state_filtered = self.leapfrog_step(state_prev, state_curr)
            states = states.at[i].set(state_next)
            
            # Update for next iteration
            state_prev = state_filtered  # Use filtered state as previous
            state_curr = state_next
        
        return states
    
    def initialize_kelvin_wave(self, amplitude: float = 1.0, 
                             wavelength: float = 1000e3,
                             lon_center: float = 60.0) -> jnp.ndarray:
        """
        Initialize an equatorial Kelvin wave.
        
        Kelvin wave structure:
        - Eastward propagating (phase speed = c)
        - Trapped at equator with scale L_eq
        - u and η in phase, v = 0
        
        Args:
            amplitude: Wave amplitude (m for η, m/s for u)
            wavelength: Wavelength in meters
            lon_center: Longitude center (degrees)
            
        Returns:
            state: Initial state [u, v, η]
        """
        # Wave parameters
        k = 2 * jnp.pi / wavelength  # Wavenumber
        
        # Longitude phase (in radians)
        lon_rad = (self.lon_2d - lon_center) * jnp.pi / 180
        phase = k * self.config.earth_radius * lon_rad
        
        # Equatorial structure (Gaussian decay)
        equatorial_structure = jnp.exp(-0.5 * (self.y_2d / self.L_eq)**2)
        
        # Kelvin wave fields
        eta = amplitude * equatorial_structure.T * jnp.cos(phase.T)
        u = (self.c / self.config.g) * self.config.g * eta / self.c  # Geostrophic balance
        v = jnp.zeros_like(u)  # No meridional velocity for Kelvin waves
        
        return jnp.stack([u, v, eta])
    
    def initialize_rossby_wave(self, amplitude: float = 1.0,
                             wavelength: float = 2000e3,
                             mode: int = 1,
                             lon_center: float = 60.0) -> jnp.ndarray:
        """
        Initialize an equatorial Rossby wave.
        
        Args:
            amplitude: Wave amplitude
            wavelength: Wavelength in meters
            mode: Meridional mode number (n = 1, 2, 3, ...)
            lon_center: Longitude center (degrees)
            
        Returns:
            state: Initial state [u, v, η]
        """
        # Wave parameters
        k = 2 * jnp.pi / wavelength
        
        # Rossby wave dispersion: ω = -βk/(k² + (2n+1)β/c)
        # For simplicity, use approximate structure
        
        lon_rad = (self.lon_2d - lon_center) * jnp.pi / 180
        phase = k * self.config.earth_radius * lon_rad
        
        # Meridional structure (Hermite polynomials approximated)
        y_norm = self.y_2d / self.L_eq
        if mode == 1:
            meridional_structure = y_norm.T * jnp.exp(-0.5 * (y_norm.T)**2)
        else:
            # Higher modes - simplified
            meridional_structure = (y_norm.T**mode) * jnp.exp(-0.5 * (y_norm.T)**2)
        
        # Rossby wave fields (westward propagating)
        eta = amplitude * meridional_structure * jnp.cos(phase.T)
        u = 0.5 * amplitude * meridional_structure * jnp.sin(phase.T)
        v = amplitude * y_norm.T * jnp.exp(-0.5 * (y_norm.T)**2) * jnp.sin(phase.T)
        
        return jnp.stack([u, v, eta])
    
    def initialize_gaussian_perturbation(self, amplitude: float = 1.0,
                                       lon_center: float = 60.0,
                                       lat_center: float = 0.0,
                                       sigma_lon: float = 10.0,
                                       sigma_lat: float = 5.0) -> jnp.ndarray:
        """
        Initialize a Gaussian height perturbation.
        
        Args:
            amplitude: Perturbation amplitude (m)
            lon_center: Longitude center (degrees)
            lat_center: Latitude center (degrees)
            sigma_lon: Longitude standard deviation (degrees)
            sigma_lat: Latitude standard deviation (degrees)
            
        Returns:
            state: Initial state [u, v, η]
        """
        # Gaussian perturbation
        lon_dist = (self.lon_2d - lon_center)**2 / (2 * sigma_lon**2)
        lat_dist = (self.lat_2d - lat_center)**2 / (2 * sigma_lat**2)
        
        eta = amplitude * jnp.exp(-(lon_dist + lat_dist)).T
        u = jnp.zeros_like(eta)
        v = jnp.zeros_like(eta)
        
        return jnp.stack([u, v, eta])
    
    def to_xarray(self, states: jnp.ndarray) -> xr.Dataset:
        """
        Convert model output to xarray Dataset.
        
        Args:
            states: Model states with shape (nt+1, 3, nlat, nlon)
            
        Returns:
            ds: xarray Dataset with coordinates and metadata
        """
        # Time coordinate
        time = jnp.arange(self.config.nt + 1) * self.config.dt / 3600  # hours
        
        # Create dataset
        ds = xr.Dataset(
            {
                'u': (['time', 'lat', 'lon'], states[:, 0, :, :]),
                'v': (['time', 'lat', 'lon'], states[:, 1, :, :]),
                'eta': (['time', 'lat', 'lon'], states[:, 2, :, :]),
            },
            coords={
                'time': time,
                'lat': self.lat,
                'lon': self.lon,
            },
            attrs={
                'title': 'Matsuno Shallow Water Model Output',
                'description': 'Equatorial wave dynamics on beta plane',
                'equivalent_depth': self.config.equivalent_depth,
                'wave_speed': float(self.c),
                'beta': self.config.beta,
                'deformation_radius': float(self.L_eq),
                'dt_seconds': self.config.dt,
                'CFL_x': float(self.cfl_x),
                'CFL_y': float(self.cfl_y),
            }
        )
        
        return ds


def create_matsuno_model(equivalent_depth: float = 25.0,
                        dt_minutes: float = 5.0,
                        nt: int = 288) -> MatsunoModel:
    """
    Create a Matsuno model with specified parameters.
    
    Args:
        equivalent_depth: Equivalent depth (m)
        dt_minutes: Time step in minutes
        nt: Number of time steps
        
    Returns:
        model: Configured MatsunoModel instance
    """
    config = MatsunoConfig(
        equivalent_depth=equivalent_depth,
        dt=dt_minutes * 60.0,  # Convert to seconds
        nt=nt
    )
    return MatsunoModel(config)