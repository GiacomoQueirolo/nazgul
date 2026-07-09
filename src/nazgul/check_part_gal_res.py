"""
Stat of particles of galaxies used for modelling
"""
import numpy as np
from glob import glob
from Modelling.lib_models import load_kwargs_input,load_kwargs_result,get_model_plot, get_red_chi2
from python_tools.get_res import load_whatever
from pathlib import Path

from python_tools.tools import short_SciNot

res_dir = "results/models/simNoShear/"
chi2 = []
n_star = []
n_gas = []
n_dm = []
n_bh = []
n_tot = []

m_star = []
m_gas = []
m_dm = []
m_bh = []
m_tot = []

for g in glob(f"{res_dir}/snap_*/kw_res.dll"):
    model_res_dir = Path(g).parent
    kwargs_result  = load_kwargs_result(model_res_dir)
    try:
        model_plot     = get_model_plot(model_res_dir,kwargs_result=kwargs_result)
        chi2.append(get_red_chi2(model_plot,verbose=False))
        gln = glob(str(model_res_dir)+"/link_gallens*")[0]
        gallens = load_whatever(gln)
        gallens.unpack()
        gal = gallens.Gal.unpack()
        print(gal.N_stars,gal.M_stars,gal.N_gas,gal.M_gas,gal.N_dm,gal.M_dm,gal.N_bh,gal.M_dm)
        n_star.append(gal.N_stars)
        n_gas.append(gal.N_gas)
        n_dm.append(gal.N_dm)
        n_bh.append(gal.N_bh)
        n_tot.append(gal.N_stars+gal.N_gas+gal.N_dm+gal.N_bh)
        
        m_star.append(gal.M_stars)
        m_gas.append(gal.M_gas)
        m_dm.append(gal.M_dm)
        m_bh.append(gal.M_bh)
        m_tot.append(gal.M_stars+gal.M_gas+gal.M_dm+gal.M_bh)

    except Exception as e:
        print(f"Failed {e}:\n{model_res_dir}")

chi2 = np.array(chi2)
n_star = np.array(n_star)
n_gas = np.array(n_gas)
n_dm = np.array(n_dm)
n_bh = np.array(n_bh)
n_tot = np.array(n_tot)

m_star = np.array(m_star)
m_gas = np.array(m_gas)
m_dm = np.array(m_dm)
m_bh = np.array(m_bh)
m_tot = np.array(m_tot)

m_starXpart = m_star/n_star
m_gasXpart = m_gas/n_gas
m_dmXpart = m_dm/n_dm
m_bhXpart = m_bh/n_bh
m_totXpart = m_tot/n_tot


print("<N>:",short_SciNot(np.nanmean(n_tot)))
print("<M>:",short_SciNot(np.nanmean(m_tot)))
print("<N_stars>:",short_SciNot(np.nanmean(n_star)))
print("<N_gas>:",short_SciNot(np.nanmean(n_gas)))
print("<N_dm>:",short_SciNot(np.nanmean(n_dm)))
print("<N_bh>:",short_SciNot(np.nanmean(n_bh)))
print("<N_tot>:",short_SciNot(np.nanmean(n_tot)))

print("<M_stars>:",short_SciNot(np.nanmean(m_star)))
print("<M_gas>:",short_SciNot(np.nanmean(m_gas)))
print("<M_dm>:",short_SciNot(np.nanmean(m_dm)))
print("<M_bh>:",short_SciNot(np.nanmean(m_bh)))
print("<M_tot>:",short_SciNot(np.nanmean(m_tot)))


print("<M_stars_part>:",short_SciNot(np.nanmean(m_starXpart)))
print("<M_gas_part>:",short_SciNot(np.nanmean(m_gasXpart)))
print("<M_dm_part>:",short_SciNot(np.nanmean(m_dmXpart)))
print("<M_bh_part>:",short_SciNot(np.nanmean(m_bhXpart)))
print("<M_tot_part>:",short_SciNot(np.nanmean(m_totXpart)))

import matplotlib.pyplot as plt
"""
fig,axes = plt.subplots(1,3,figsize=(10,5))
ax = axes[0]
ax.scatter(chi2,n_tot,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"N part. tot")
ax = axes[1]
ax.scatter(chi2,m_tot,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot")
ax = axes[2]
ax.scatter(chi2,m_totXpart,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot/N part. tot")

plt.tight_layout()
nm = f"{res_dir}/chi2VsNpart.png"
plt.savefig(nm)
print(f"Saved {nm}")
plt.close(fig)"""


fig,axeses = plt.subplots(5,3,figsize=(10,25))
axes = axeses[0]
ax = axes[0]
ax.scatter(chi2,n_tot,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"N part. tot")
ax = axes[1]
ax.scatter(chi2,m_tot,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot")
ax = axes[2]
ax.scatter(chi2,m_totXpart,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot/N part. tot")


axes = axeses[1]
n,m,mxpart = n_star,m_star,m_starXpart
nm = "Star"

ax = axes[0]
ax.set_title(nm)
ax.scatter(chi2,n,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"N part. tot")
ax = axes[1]
ax.scatter(chi2,m,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot")
ax = axes[2]
ax.scatter(chi2,mxpart,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot/N part. tot")

axes = axeses[2]
n,m,mxpart = n_gas,m_gas,m_gasXpart
nm = "Gas"
ax = axes[0]
ax.set_title(nm)
ax.scatter(chi2,n,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"N part. tot")
ax = axes[1]
ax.scatter(chi2,m,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot")
ax = axes[2]
ax.scatter(chi2,mxpart,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot/N part. tot")


axes = axeses[3]
n,m,mxpart = n_dm,m_dm,m_dmXpart
nm = "DM"
ax = axes[0]
ax.set_title(nm)
ax.scatter(chi2,n,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"N part. tot")
ax = axes[1]
ax.scatter(chi2,m,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot")
ax = axes[2]
ax.scatter(chi2,mxpart,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot/N part. tot")

axes = axeses[4]
n,m,mxpart = n_bh,m_bh,m_bhXpart
nm = "BH"
ax = axes[0]
ax.set_title(nm)
ax.scatter(chi2,n,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"N part. tot")
ax = axes[1]
ax.scatter(chi2,m,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot")
ax = axes[2]
ax.scatter(chi2,mxpart,c="k")
ax.set_xlabel(r"$\chi^2$")
ax.set_ylabel(r"M tot/N part. tot")

plt.tight_layout()
nm = f"{res_dir}/chi2VsNpart.png"
plt.savefig(nm)
print(f"Saved {nm}")
plt.close(fig)