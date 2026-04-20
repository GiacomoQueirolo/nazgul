import dill
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from nazgul.Translator.particle_galaxy import clip_coord

from nazgul.mount_doom.generate_gal_lens import get_kw_lenspart,get_kw_galpart,get_all_PG
from nazgul.Translator.translator import Gal2kwMXYZ
from nazgul.Translator.EAGLE.fnct import get_snap
def compute_axis_ratio(Gal):

    Mstar = Gal.stars["mass"]*u.Msun
    # Particle pos
    Xstar,Ystar,Zstar =  np.transpose(Gal.stars["coords"])*u.Mpc.to("kpc")*u.kpc #kpc

    # clip particle outliers
    Ms,Xs,Ys,Zs = clip_coord(Mstar,Xstar,Ystar,Zstar)
     
    Cx,Cy,Cz = Gal.centre*u.Mpc.to("kpc")*u.kpc/(Gal.xy_propr2comov) # (now) kpc 
    
    Xs -= Cx
    Ys -= Cy
    Zs -= Cz
    
    mass = Mstar.value
    # center positions first!
    x = Xs.value
    y = Ys.value
    z = Zs.value
    
    pos = np.transpose([x,y,z])
    I = np.zeros((3,3))
    for i in range(len(pos)):
        r = pos[i]
        I += mass[i] * np.outer(r, r)

    # eigenvalues
    eigvals = np.linalg.eigvalsh(I)
    eigvals = np.sort(eigvals)[::-1]  # λ1 ≥ λ2 ≥ λ3

    a = np.sqrt(eigvals[0])
    b = np.sqrt(eigvals[1])
    c = np.sqrt(eigvals[2])

    return c / b
    

kw_galpart={"min_z":.09,
           "max_z":0.11,
           "min_mass":1e11}
kw_galpart  = get_kw_galpart(kw_galpart)
all_Gal     = get_all_PG(**kw_galpart)
all_axis_ratio = []
gals_names = []
z_l = []
for Gal in all_Gal:
    gals_names.append(Gal.name)
    z_l.append(Gal.z)
    try:
        all_axis_ratio.append(Gal.axis_ratio)
    except:
        Gal.run()
        axis_ratio = compute_axis_ratio(Gal)
        all_axis_ratio.append(axis_ratio)
        Gal.axis_ratio = axis_ratio
        Gal.store_gal()
z_l = list(set(z_l))
if len(z_l)==1:
    snap = get_snap(z=z_l[0])
else:
    snap = ""
    for z in z_l:
        snap+= get_snap(z=z)+"_"
    snap = snap[:-1]
    
all_axis_ratio= np.array(all_axis_ratio)

nm2 = f"tmp/axis_ratio_snap{snap}.dll"
with open(nm2,"wb") as f:
    dill.dump({"gal_names":gals_names,
               "z":z_l,
               "axis_ratio":all_axis_ratio},f)
print(f"Saved {nm2}")

    
plt.title(f"axial ratio c/b for snap {snap}/z {z_l}")
plt.hist(all_axis_ratio)
plt.xlabel("c/b")
plt.axvline(0.7,c="k",label="lower limit from Vyvere '22")
plt.legend()
nm = f"tmp/axis_ratio_snap{snap}.png"
plt.savefig(nm)
print(f"Saving {nm}")
