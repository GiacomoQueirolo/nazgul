# count effective lenses
import dill
from glob import glob
import matplotlib.pyplot as plt

from python_tools.get_res import load_whatever

from nazgul.mount_doom.generate_particle_lens_dom import LensPart
from nazgul.pathfinder import get_catlensdir
from nazgul.project_gal import ProjectionError
actual_lenses= []
N_actual_lenses= 0
computed_sublenses = glob("RingBearer/EAGLE/RefL0025N0752/snap_0??/Gn*SGn*/Sub/Sub_*.pkl")
N_computed_sublenses = len(computed_sublenses)
# Study thetaE distribution
thetaEs = []

for sub_lns in computed_sublenses:
    #Gal = load_whatever(gal)
    
    ln = load_whatever(sub_lns)
    ln.run()
    try:
        LP  = LensPart(ln.Gal)
        LP.run()
        actual_lenses.append(LP.lenspart.pkl_path) # sub-lens
        N_actual_lenses+=1
        thetaEs.append(LP.thetaE.value)
    except ProjectionError as PE:
        # ignore galaxies which are not lenses
        pass
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

plt.title(r"$\theta_E$ of Lenses")
plt.hist(thetaEs,bins=12)
plt.xlabel(r"$\theta_E$ ['']")
plt.ylabel("N (tot="+str(N_actual_lenses)+")")
fig_tE = str(catdir)+"/Distr_thetaE.png"
plt.savefig(fig_tE)
print(f"Saving {fig_tE}")