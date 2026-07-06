#  Synthetic Diagnostic for Linear Plasma (SPEKTRE)

Internship project at the Institut Jean Lamour (Université de Lorraine).  
Synthetic microwave diagnostic for the SPEKTRE linear plasma device.

## Context

The goal is to reconstruct the electron density profile ne(r) of a linear plasma from the travel time of microwave rays.  
Various density profiles have been tested. Rays are traced using Hamiltonian ray tracing equations, and the density profile is recovered via a Abel inversion.

## Files

| File | Description |
|------|-------------|
| `cone_ui.py` | Interactive UI — Hamiltonian ray tracing (RK4) in a fan-beam emitter-receiver geometry for three density profiles (parabolic, Gaussian, nested shells); displays ray paths, travel time τ(α) and frequency ω(α) per ray, and isopeak curves. Does not compute synthetic inversion data — see `general_ray_tracing.py` for that. |
| `bouguer_inverse.py` | Bouguer-Abel inversion (onion-peeling) reconstructing ne(r) from synthetic (τ, ω) pairs: converts launch angle α to impact parameter, then solves layer-by-layer for ωp²(r) via Brent's method |
| `general_ray_tracing.py` | Hamiltonian ray tracing solver (RK4) for a layered parabolic plasma; generates the synthetic (τ, ω) dataset used as input to `bouguer_inverse.py`, with configurable noise on τ |
| `plot_transition_couches.py` | Visualizes the monotonicity transition of τ(ωp²) in the forward model across multiple plasma layers |

Cache files (`cone_ui_cache_*.npz`, `cone_cache/`) store precomputed ray tracing results to speed up the UI.

## Dependencies

The only non-standard dependency is **scipy** (numpy and matplotlib are usually pre-installed):

```bash
pip install scipy
```

If starting from scratch:
```bash
pip install numpy scipy matplotlib
```

## Running the interactive UI

```bash
python cone_ui.py
```

The UI lets you adjust plasma parameters (density, profile width, number of layers) and visualize ray paths, travel times, and the reconstructed density profile in real time.

### Cache files

Ray tracing is computationally expensive. To avoid long computation times, `cone_ui.py` uses a cache system — results are saved as `.npz` files and reloaded instantly on subsequent runs.

**The precomputed cache files are included in this repository** (`cone_ui_cache_*.npz` and `cone_cache/`). It is strongly recommended to download them alongside `cone_ui.py`, otherwise the UI will recompute everything from scratch on first launch, which can take several minutes.

If you change any plasma or geometry parameters (density, profile width, plasma radius, emitter-receiver distance, number of layers, etc.), the cache will no longer match and the UI will automatically trigger a new computation for the updated configuration. The result is then saved as a new cache file for future use.

## Author

Paul Hausherr — L3 Physics, Université de Lorraine  
3rd year physics bachelor internship — Institut Jean Lamour, 2026
