"""
Stat of particles of galaxies used for modelling
"""
import gc
import dill
import numpy as np
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

from Modelling.lib_models import load_kwargs_input, load_kwargs_result, get_model_plot, get_red_chi2
from python_tools.get_res import load_whatever
from python_tools.tools import short_SciNot

res_dir  = Path("results/models/simNoShear/")
pkl_path = res_dir / "chi2_part_stats.pkl"
reload   = True
# ── collect results ───────────────────────────────────────────────────────────
try:
    assert reload
    records = load_whatever(pkl_path)
except:
    
    records = []   # list of dicts, one per successfully processed lens
    
    for g in glob(str(res_dir / "snap_*/kw_res.dll")):
        model_res_dir = Path(g).parent
        try:
            kwargs_result = load_kwargs_result(model_res_dir)
            model_plot    = get_model_plot(model_res_dir, kwargs_result=kwargs_result)
            chi2_val      = get_red_chi2(model_plot, verbose=False)
    
            gln     = glob(str(model_res_dir) + "/link_gallens*")[0]
            gallens = load_whatever(gln)
            gal     = load_whatever(gallens.Gal_path)
            gal.unpack()
    
            if not hasattr(gal, "N_stars"):
                gal.initialise_parts()
                gal.N_stars = len(gal.stars.particle_ids)
                gal.N_gas   = len(gal.gas.particle_ids)
                gal.N_dm    = len(gal.dark_matter.particle_ids)
                gal.N_bh    = len(gal.black_holes.particle_ids)
                gal.store_gal()
    
            records.append({
                "model_res_dir": str(model_res_dir),
                "chi2":    chi2_val,
                "N_stars": getattr(gal, "N_stars", 0),
                "N_gas":   getattr(gal, "N_gas",   0),
                "N_dm":    getattr(gal, "N_dm",    0),
                "N_bh":    getattr(gal, "N_bh",    0),
                "M_stars": getattr(gal, "M_stars", 0.),
                "M_gas":   getattr(gal, "M_gas",   0.),
                "M_dm":    getattr(gal, "M_dm",    0.),
                "M_bh":    getattr(gal, "M_bh",    0.),
            })
    
            gal.slim_down()
            del gal, gallens, model_plot, kwargs_result
            gc.collect()
    
        except Exception as e:
            print(f"Failed {model_res_dir}:\n  {e}")

# ── derive aggregate arrays ───────────────────────────────────────────────────

def _arr(key):
    return np.array([r[key] for r in records])

chi2    = _arr("chi2")
n_star  = _arr("N_stars");  m_star = _arr("M_stars")
n_gas   = _arr("N_gas");    m_gas  = _arr("M_gas")
n_dm    = _arr("N_dm");     m_dm   = _arr("M_dm")
n_bh    = _arr("N_bh");     m_bh   = _arr("M_bh")
n_tot   = n_star + n_gas + n_dm + n_bh
m_tot   = m_star + m_gas + m_dm + m_bh

# ── save to dill ──────────────────────────────────────────────────────────────

stats = {
    "records":  records,          # per-lens raw data
    "chi2":     chi2,
    "N_stars":  n_star,  "M_stars": m_star,
    "N_gas":    n_gas,   "M_gas":   m_gas,
    "N_dm":     n_dm,    "M_dm":    m_dm,
    "N_bh":     n_bh,    "M_bh":    m_bh,
    "N_tot":    n_tot,   "M_tot":   m_tot,
}
with open(pkl_path, "wb") as f:
    dill.dump(stats, f)
print(f"Saved stats → {pkl_path}")

# ── print summary ─────────────────────────────────────────────────────────────

rows = [
    ("tot",   n_tot,  m_tot),
    ("stars", n_star, m_star),
    ("gas",   n_gas,  m_gas),
    ("dm",    n_dm,   m_dm),
    ("bh",    n_bh,   m_bh),
]
col_w = 14
print(f"\n{'':10s} {'<N>':>{col_w}} {'<M> [Msun]':>{col_w}} {'<M/N> [Msun]':>{col_w}}")
print("-" * (10 + 3 * (col_w + 1)))
for label, n_arr, m_arr in rows:
    ratio = np.where(n_arr > 0, m_arr / n_arr, np.nan)
    print(f"{label:10s} "
          f"{short_SciNot(np.nanmean(n_arr)):>{col_w}} "
          f"{short_SciNot(np.nanmean(m_arr)):>{col_w}} "
          f"{short_SciNot(np.nanmean(ratio)):>{col_w}}")

# ── plot ──────────────────────────────────────────────────────────────────────

species = [
    ("Total", n_tot,  m_tot),
    ("Stars", n_star, m_star),
    ("Gas",   n_gas,  m_gas),
    ("DM",    n_dm,   m_dm),
    ("BH",    n_bh,   m_bh),
]

fig, axes_grid = plt.subplots(len(species), 2, figsize=(8, 4 * len(species)))

for i, (label, n_arr, m_arr) in enumerate(species):
    ax_n, ax_m = axes_grid[i]

    ax_n.scatter(chi2, n_arr, c="k", s=20)
    ax_n.set_ylabel(f"N {label}")
    ax_n.set_xlabel(r"$\chi^2_{\rm red}$")
    ax_n.set_title(label)

    ax_m.scatter(chi2, m_arr, c="steelblue", s=20)
    ax_m.set_ylabel(f"M {label} [$M_\\odot$]")
    ax_m.set_xlabel(r"$\chi^2_{\rm red}$")

plt.tight_layout()
nm = res_dir / "chi2VsNpart.png"
plt.savefig(nm, dpi=150)
print(f"Saved plot → {nm}")
plt.close(fig)