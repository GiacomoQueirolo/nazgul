# count effective lenses
import dill
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

from python_tools.get_res import load_whatever
from nazgul.pathfinder import get_catlensdir
from nazgul.project_gal import ProjectionError
actual_lenses= []
computed_sublenses = glob("RingBearer/EAGLE/RefL0025N0752/snap_0??/Gn*SGn*/Sub/Sub_*Prj?.pkl")
N_computed_sublenses = len(computed_sublenses)
# Study thetaE distribution
thetaEs = []
z_source = []
z_source_min = []
z_lens = []
masses = []
gals   = []
for sub_lns in computed_sublenses:
    ln = load_whatever(sub_lns)
    ln.unpack()
    try:
        ln.run()
        gals.append(ln.Gal)
        actual_lenses.append(ln.pkl_path) # sub-lens
        thetaEs.append(ln.thetaE.value)
        z_source.append(ln.z_source)
        
        z_source_min.append(ln.z_source_min)
        z_lens.append(ln.z_lens)
        masses.append(ln.Gal.M_tot)
    except ProjectionError as PE:
        # ignore galaxies which are not lenses
        pass
N_actual_lenses= len(actual_lenses)
print("\n\n")
print("Actual lenses:"+str(N_actual_lenses))
print(str(int(N_actual_lenses*100/N_computed_sublenses))+"% of computed galaxies")
catdir = get_catlensdir()
cat_lens = {"lens_path":actual_lenses,
            "thetaE":thetaEs,
            "gal_path":computed_sublenses}
cat_file = catdir/"LensCat.pkl"
with open(cat_file,"wb") as f:
    dill.dump(cat_lens,f)
print(f"Saved {cat_file}")

thetaEs  = np.array(thetaEs)
masses   = np.array(masses)
z_lens   = np.array(z_lens)
z_source = np.array(z_source)
z_source_min = np.array(z_source_min)

plt.title(r"$\theta_E$ of Lenses")
plt.hist(thetaEs,bins=20)
plt.xlabel(r"$\theta_E$ ['']")
plt.ylabel("N (tot="+str(N_actual_lenses)+")")
fig_tE = str(catdir)+"/Distr_thetaE.png"
plt.savefig(fig_tE)
print(f"Saving {fig_tE}")
plt.close()

plt.title(r"z source")
plt.hist(z_source,bins=20)
plt.xlabel(r"z source")
plt.ylabel("N (tot="+str(N_actual_lenses)+")")
fig_tE = str(catdir)+"/Distr_zsource.png"
plt.savefig(fig_tE)
print(f"Saving {fig_tE}")

plt.close()


plt.title(r"M lens [solar masses]")
plt.hist(masses,bins=20)
plt.xlabel(r"M")
plt.ylabel("N (tot="+str(N_actual_lenses)+")")
fig_tE = str(catdir)+"/Distr_mlens.png"
plt.savefig(fig_tE)
print(f"Saving {fig_tE}")
plt.close() 

plt.scatter(masses,thetaEs,c=z_lens,cmap="viridis")
plt.colorbar(label=r"z$_{lens}$")
plt.xlabel(r"M [M$_\odot$]")
plt.ylabel(r"$\theta_E$ ['']")
plt.title("Relation between total galaxy masses and Einstein radius (N="+str(N_actual_lenses)+")")
plt.tight_layout()
fig_MtE = str(catdir)+"/MvstE.png"
plt.savefig(fig_MtE)
print(f"Saving {fig_MtE}")
plt.close()

#Cropping mass outliers
i_crop = masses<np.median(masses)+3*np.std(masses)
plt.scatter(masses[i_crop],thetaEs[i_crop],c=z_source_min[i_crop],cmap="viridis")
plt.colorbar(label=r"z$_{source,min}$")
plt.xlabel(r"M [M$_\odot$]")
plt.ylabel(r"$\theta_E$ ['']")
plt.title("Relation between total galaxy masses and Einstein radius (N="+str(len(thetaEs[i_crop]))+")")
plt.tight_layout()
fig_MtEzs = str(catdir)+"/MvstE_zs.png"
plt.savefig(fig_MtEzs)
print(f"Saving {fig_MtEzs}")
plt.close()


# considering the star masses 
# (to comp. w SEAGLE selection M_* > 1.76 *1e10 Msun)
m_s = []
for g in gals:
    # ugly but works
    ms = float(str(g).split("and Mass in ")[1].split("Stars:")[1].split(" ")[0] )
    m_s.append(ms)
m_s = np.array(m_s)
plt.hist(m_s,bins=15)
plt.title(r"Star mass of galaxy [solar masses]")
plt.xlabel(r"M$_{*}* [M$_\odot$]")
fig_ms = str(catdir)+"/Mstars.png"
plt.savefig(fig_ms)
plt.close()