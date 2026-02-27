from   astropy.cosmology import FlatLambdaCDM
import astropy.constants as const
import astropy.units as u
import numpy as np

default_cosmo       = FlatLambdaCDM(H0=67.7,Om0=0.3)

def DsDds(cosmo,z_d,z_s):
    if np.isinf(z_s):
        return 1*u.dimensionless_unscaled
    Ds  = cosmo.angular_diameter_distance(z_s)
    Dds = cosmo.angular_diameter_distance_z1z2(z_d,z_s)
    return Ds/Dds

def SigCrit(cosmo,z_lens,z_source):
    cosmo_dd    = cosmo.angular_diameter_distance(z_lens).to("kpc")   #kpc
    ratio_DsDds = DsDds(cosmo,z_lens,z_source)
    Sigma_Crit  = ratio_DsDds*(const.c**2)/(4*np.pi*const.G*cosmo_dd) #
    return Sigma_Crit.to("Msun /(kpc kpc)")
    
def SigCrArc2(cosmo,z_lens,z_source):
    arcXkpc           = cosmo.arcsec_per_kpc_proper(z_lens) # ''/kpc
    Sigma_Crit        = SigCrit(cosmo=cosmo,z_lens=z_lens,z_source=z_source)
    Sigma_Crit_arcs2  = Sigma_Crit.to("Msun /(kpc kpc)")/(arcXkpc*arcXkpc)
    return Sigma_Crit_arcs2

"""
# big brain time: 

cosmo= FlatLambdaCDM(H0=70, Om0=0.3)

arcXkpc = cosmo.arcsec_per_kpc_proper(2)


cosmo.angular_diameter_distance(2)
Out[6]: <Quantity 1726.62069147 Mpc>

1/arcXkpc.to("rad/Mpc")
Out[7]: <Quantity 1726.62069147 Mpc / rad>
"""