# to verify that the projection works
from project_gal_AMR import *
from remade_gal import get_rnd_NG
from python_tools.get_res import LoadClass
from Gen_PM_PLL_AMR import LensPart,plot_all,min_thetaE
from Gen_PM_PLL_AMR import kwlens_part_AS,cutoff_radius,z_source_max,pixel_num
def sample_z_source(self,z_source_min,z_source_max):
        # this is here to allow modularity 
        # for now a simple uniform sample, but we could define something more fancy
        z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        return z_source
    
if __name__ == "__main__":
    Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_25//Gn3SGn0.pkl")
    kw_parts         = Gal2kwMXYZ(Gal) # kwargs of Msun,XYZ in kpc (explicitely) centered around Centre of Mass (CM)
    z_source_max  = 4
    arcXkpc = Gal.cosmo.arcsec_per_kpc_proper(Gal.z_lens)
    kw_res = projection_main_AMR(Gal,kw_parts,z_source_max,sample_z_source,min_thetaE,
                    arcXkpc,verbose=True,save_res=False,reload=False)