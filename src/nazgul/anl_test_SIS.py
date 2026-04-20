import matplotlib.pyplot as plt

from nazgul.Translator.translator import PartGal 
from nazgul.mount_doom.generate_gal_lens import GalLens
pg = PartGal({"n_smpl":1e6},sim="SIS",simsuite="ANL_TEST") 
pg.run () 
pg.store_gal()
lensgal = GalLens(pg,0)
lensgal.run()


lensgal.unpack()
fig,axis = plt.subplots(1,2)
plt.suptitle(r"$\alpha$ map for SIS simulation")
axis[0].imshow(lensgal.alpha_map[0])
axis[1].imshow(lensgal.alpha_map[1])
plt.savefig("tmp/alpha_sis.png")

lensgal._unpack_Gal()

cosmo = lensgal.Gal.cosmo
Ds_p = cosmo.angular_diameter_distance(lensgal.Gal.z_source)
Ds = cosmo.angular_diameter_distance(lensgal.z_source)
Dls = cosmo.angular_diameter_distance_z1z2(lensgal.z_lens,lensgal.z_source)
Dls_p = cosmo.angular_diameter_distance_z1z2(lensgal.z_lens,lensgal.Gal.z_source)

rescale_fact = Ds*Dls_p/(Ds_p*Dls)

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck18 as default_cosmo

from python_tools.tools import to_dimless,short_SciNot,ensure_unit,mkdir
from nazgul.pathfinder  import get_sim_dir

from nazgul.lib_cosmo import SigCrit
SigCrit_gal = SigCrit(cosmo =lensgal.cosmo,z_lens=lensgal.z_lens,z_source=lensgal.Gal.z_source)
SigCrit_rescaled = lensgal.SigCrit/rescale_fact
print(f"% error in SigCrit : {100*np.abs(SigCrit_gal-SigCrit_rescaled)/SigCrit_gal}")

tE_rescaled = (lensgal.thetaE*rescale_fact) #/lensgal.Gal.theta_E
print(f"% error in theta_E : {100*(lensgal.Gal.theta_E-tE_rescaled.value)/lensgal.Gal.theta_E}")