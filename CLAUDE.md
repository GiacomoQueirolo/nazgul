# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository (the **Nazgul** package, `EAGLE_Lensing/`).

## Environment Setup

```bash
source ../activate.sh   # activates nazgul_env conda env and sets PYTHONPATH
```

Create the environment from scratch:
```bash
mamba env create -f nazgul_env.yaml
pip install -e .
```

## Git

Two branches: `main` (shared with collaborators) and `sam` (personal working branch). `RingBearer/` (simulation data) and `__pycache__/` are gitignored.

## Testing

```bash
# Nazgul unit tests
pytest src/nazgul/test/

# Single test file
pytest src/nazgul/test/test_project_gal_AMR.py
```

## Architecture

Nazgul is a pipeline that takes hydrodynamical cosmological simulations (EAGLE, COLIBRE) and produces gravitational lens models and population statistics.

### Package layout (`src/nazgul/`)

All runtime parameters live in `configurations.py` — change `std_simsuite`, `min_z`, `max_z`, `min_mass`, `pixel_num`, `min_thetaE`, `scale_tE`, `z_source_max`, `verbose`, `forecast_telescope` there before running any pipeline step. The current default simsuite is `COLIBRE` (`SimSuiteNames[1]`); set to `"EAGLE"` or `"ANL_TEST"` as needed. Set `nazgul_path_origin` when loading pickles synced from a different machine (e.g. COSMA → local); `_resolve_gal_path()` in `mount_doom/cracks_of_doom.py` uses it to remap stored paths.

Data flows through these stages:

1. **Translator** (`Translator/`) — dispatch layer over simulation-specific readers. `translator.py` exposes a uniform `PartGal` class; it dynamically imports the correct submodule (`EAGLE/`, `COLIBRE/`, or `ANL_TEST/`) based on `simsuite`. Each submodule implements `SimPartGal`, `get_kw_SimPartGal`, `Gal2MXYZ`, `get_rnd_SPG`, `get_all_SPG`, and `gal_path2kwGal`. Available simulations: EAGLE = `["RefL0025N0752", "RefL0012N0188", "RefTuto"]`; COLIBRE = `["L0025N0752"]` with subsim `{"L0025N0752": ["THERMAL_AGN_m5"]}`; ANL_TEST = `["SIS"]`. `Translator/__init__.py` also exports `std_sim`, `std_subsim`, `test_sim`, and `tutorial_sim` (= `RefTuto`, used by the Tutorial).

2. **Galaxy projection** (`project_gal.py`) — wraps `PartGal` in `ProjGal` (projection index 0/1/2 = x/y/z axis), computes 2-D density maps via AMR (`AMR2D_PLL.py`), locates the maximum density coordinate, and tests supercriticality for a source up to `z_source_max`. Results are cached per projection as `projection_{index}.pkl`.

3. **Particle lenses** (`particle_lenses.py`) — converts particle positions and masses into lenstronomy-compatible lens parameters. Two profiles: Arcsinh parallel (`AS`, default, `kwlens_part = {"type":"AS","theta_cAS":5e-3}`) and point-mass parallel (`PM`). Class hierarchy: `PartLens_basis` is the abstract base; `PartLens(PartLens_basis)` is the standard implementation used by `GalLens`; `PartLensExpanded(PartLens_basis)` additionally supports `kw_add_lenses` for line-of-sight perturber profiles.

4. **Defaults & observation config** (`mount_doom/cracks_of_doom.py`) — centralises per-run defaults: `default_kwlens_part_AS = {"type":"AS","theta_cAS":5e-3}`, `default_kwlens_part_PM = {"type":"PM"}`, `kwargs_source_default` (magnitude-based Sérsic ellipse, `{"magnitude":25.,...}`), `kwargs_band_sim` (HST-like), and `kw_prior_z_source_stnd`. Also provides `LoadLens(path)` for loading cached `GalLens` pickles and `_resolve_gal_path()` for cross-machine path remapping.

5. **GalLens** (`mount_doom/generate_gal_lens.py`) — `BasicLensPart` subclass that owns the full per-galaxy lens computation: calls `galaxy_projection`, sets up `PartLens`, computes `alpha_map`, `kappa_map`, `hessian`, `psi_map` (all lazily computed and cached), and persists to `RingBearer/{SimSuite}/{Sim}/snap_{NNN}/{GnXSGnY}/Sub/Sub_*.pkl`. Pass `ignore_OoBErr=True` to suppress out-of-bound errors from lenstronomy.

6. **LensSystem** (`mount_doom/lens_system.py`) — wraps `GalLens` and adds optional extra lenses (`kwargs_add_lenses`) via lenstronomy's `LensModel`. Handles simulated observations via `SimAPI`, source sampling within the tangential caustic, and lensed image generation.

7. **Modelling** (`modelling_*.py`, `model_*.py`) — fits parametric models (SIS, SIE, 2SIS, with/without line-of-sight perturbers) to the simulated lens images using lenstronomy's inference engine. Outputs go to `tmp/modelling_*/`. Analysis of fitted chains is done in `combined_modelling_results.py` (all lenses) and `combined_modelling_results_one_lens.py` (single lens) using chainconsumer; both include LOS shear statistics (`shear_magnitude`, `shear_stdev`).

8. **LensPop** (`LensPop/`) — pre-computed lens population catalogs for DES, Euclid, LSST. Active telescope is set via `forecast_telescope` in `configurations.py`.

9. **Isodensity analysis** (`isodens.py`, `isodens_stat.py`) — fits elliptical isophotes to kappa or psi maps using `photutils.isophote` (`Ellipse`, `EllipseGeometry`). `fit_isodens(lens)` returns ellipse parameters (axis ratio, position angle, boxy/diskiness `b4`) as a function of radius; `isodens_stat.py` is a batch script that runs it across all computed lenses and builds population distributions.

### Data storage tree

All persistent results live under `src/nazgul/RingBearer/` (gitignored):

```
RingBearer/
  {SimSuite}/                     e.g. EAGLE/
    {Sim}/                        e.g. RefL0025N0752/
      {subsim}/                   e.g. THERMAL_AGN_m5/ (COLIBRE only)
        CatGal/                   galaxy catalogue pickles
        CatLens/                  lens catalogue pickles (from stat_lenses.py)
        snap_{NNN}/
          ParticleData/           HDF5 snapshot files (symlinks on HPC)
          {GnXSGnY}/
            Gal/                  PartGal pickle
            Projection/           projection_{0,1,2}.pkl  (ProjGal)
            Sub/                  Sub_*Npix*Part*Prj*.pkl (GalLens)
            Dom/                  parametric model fits
            LensSystem/           LensSystem pickles
```

### Key classes and the `reload` flag

`PartGal` proxies all attribute access to its internal `SimPartGal`. `ProjGal` proxies to `PartGal`. `GalLens` inherits from `BasicLensPart`/`BasicGal`. All persistent classes use `dill` for serialization. Particle lens hierarchy: `PartLens_basis → PartLens` (standard) or `PartLensExpanded` (supports extra LOS lens profiles).

`reload=True` loads a cached `.pkl` from disk; `reload=False` recomputes from scratch (slow — involves re-reading HDF5 particle data and re-running AMR). Large attributes (`Gal`, `PartLens`, `cosmo`, `kwargs_lens`, `lens_prof`) are stripped before serialization and reconstructed on `unpack()` / `LoadLens()`.

### Entry points

```python
# Batch: all galaxies matching filters (min_z/max_z/min_mass are the basic set;
# also accepts min_vel_disp, min_hmr, min_mass_stars, sim, subsim, etc.)
from nazgul.mount_doom.generate_gal_lens import wrapper_get_all_lens, wrapper_get_rnd_lens
all_lenses = wrapper_get_all_lens(kw_galpart={"min_z":.49,"max_z":.51,"min_mass":1e12}, reload=True)

# Random single lens (useful for testing)
lens = wrapper_get_rnd_lens(reload=True)

# Single galaxy
from nazgul.Translator.translator import PartGal
from nazgul.mount_doom.generate_gal_lens import GalLens
Gal  = PartGal({"Gn":23,"SGn":0}, simsuite="EAGLE", sim="RefL0025N0752", snap="27", reload=True)
lens = GalLens(Gal, projection_index=0, reload=True)
lens.run()
lens.kappa_map   # 2-D numpy array

# Load all previously computed lenses for a snapshot
from nazgul.stat_lenses import get_all_gallens
lenses = get_all_gallens(snaps=[27], sim="RefL0025N0752")
```

### Adding a new simulation

Create `Translator/{NewSim}/` with `__init__.py`, `particle_galaxy.py` (subclassing `BasicPartGal`; must implement `SimPartGal`, `get_kw_SimPartGal`, `Gal2MXYZ`, `get_rnd_SPG`, `get_all_SPG`, `gal_path2kwGal`, `get_z_snap`), and `pathfinder.py`. Add the new suite name to `SimSuiteNames` in `configurations.py`. The top-level `translator.py` dispatches to `get_rnd_PG`/`get_all_PG` wrappers that call your submodule's `get_rnd_SPG`/`get_all_SPG`.

## EAGLE data setup

Run `python src/nazgul/Translator/EAGLE/setup_eagle_data.py` to interactively download EAGLE particle snapshots. Credentials are stored (via dill) in `Translator/EAGLE/.eagle_account.dll` (gitignored).

## Tutorial

`src/nazgul/Tutorial/Tutorial.ipynb` — end-to-end walkthrough using pre-packaged data under `Tutorial/data_Tuto/`.
