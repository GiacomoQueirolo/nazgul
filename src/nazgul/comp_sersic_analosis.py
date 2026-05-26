import astropy.units as u
import numpy as np
import  matplotlib.pyplot as plt
from astropy.cosmology import Planck13
from analosis.utilities.useful_functions import Utilities

R_sersic_nazgul,n_sersic_nazgul,mag_nazgul = 0.1,3,25

cosmo = Planck13
util = Utilities(cosmo, "./tmp/")

z_l   = 0.271 # SEAGLE_I
z_source = 2 # arbitrary
n_smpl = 1000
distances = {"os": cosmo.angular_diameter_distance(z_l)}

mean_radius = 3e-3 #[Mpc] #PFmod
radius = np.random.lognormal(np.log(mean_radius), np.log(2)/2,n_smpl)*u.Mpc
# this ensures that 95% of the events have a radius that is at most
# a factor two larger or smaller than the mean radius.
#radius = np.max(radius.value, mean_radius/2) #PFmod: ensures large enough source
radius[radius.value<mean_radius/2] = mean_radius*u.Mpc/2
R_sersic = radius / distances['os'] # [rad]
R_sersic = util.angle_conversion(R_sersic, 'to arcsecs')*u.arcsec

# absolute magnitude: assume that the luminosity is proportional to the
# galaxy's area; for r = mean_radius we have M = reference_magnitude
reference_magnitude = -22 #PFmod
absolute_magnitude = reference_magnitude - 5 * np.log10(radius.value / mean_radius)

D = (1 + z_source)**2 * distances['os'] # luminosity distance to s [Mpc]
magnitude = absolute_magnitude + 5 * np.log10(D.value) + 25 # 25 = log10(Mpc/10pc)

mean_sersic_index = 4
n_sersic = np.random.lognormal(np.log(mean_sersic_index), np.log(1.5)/2,n_smpl)


fig,axes = plt.subplots(1,3,figsize=(15,5))
plt.suptitle("Comparing Source parameters: Nazgul vs Analosis")
axes[0].hist(R_sersic,bins=30)
axes[0].set_xlabel(r"R$_{\rm{Sersic}}$")
axes[0].axvline(R_sersic_nazgul,ls="--",label=r"R$_{\rm{Sersic, nazgul}}$",c="k")
axes[1].hist(n_sersic,bins=30)
axes[1].set_xlabel(r"n$_{\rm{Sersic}}$")
axes[1].axvline(n_sersic_nazgul,ls="--",label=r"n$_{\rm{Sersic, nazgul}}$",c="k")
axes[2].hist(magnitude,bins=30)
axes[2].set_xlabel(r"mag")
axes[2].axvline(mag_nazgul,ls="--",label=r"mag$_{\rm{nazgul}}$",c="k")

for ax in axes:
    ax.legend()
nm= "tmp/cmp_sersic_nazVsAnalosis.png"
plt.savefig(nm)
print(f"Saving {nm}")

