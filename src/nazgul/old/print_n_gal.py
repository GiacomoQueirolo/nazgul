# copy from plot_GSMF
# adapted to plot N of galaxies at different redshifts
# then output a table of indexes to use to get them, with 
# -  mass
# - coordinates
# - redshift

import numpy as np
import matplotlib.pyplot as plt

#from plot_GSMF import con # so i don't have to repeat it
from sql_connect import exec_query_orig
from fnct import std_sim,get_simsize,sim_path

sim_name = std_sim 
sim_size = get_simsize(sim_name)
min_mass = "1e12" # select higher masses bc 1) lenses 2) else we have too many points

myQuery = "SELECT \
    COUNT(gal.GalaxyID) as N_gal, \
    SUM(gal.Mass) as Mtot_gal, \
    gal.Redshift as z \
FROM \
    # %s_Subhalo as gal \
# WHERE \
#     (gal.Redshift between 0 and 2) and \
#     gal.Mass > %s \
# GROUP BY \
#     gal.Redshift \
# ORDER BY \
#     gal.Redshift"%(sim_name,min_mass)

# Execute
myData = exec_query_orig(myQuery)
plt.scatter(myData["z"],np.log(myData["N_gal"]))
plt.xlabel("z")
plt.ylabel("log(N_gal)")
plt.savefig(sim_path+"/high_Ngalvsz.png")
plt.close()
plt.scatter(myData["z"],np.log(myData["Mtot_gal"]))
plt.ylabel("log(Sum(M_gal(zi)))")
plt.xlabel("z")
plt.savefig(sim_path+"/high_Mgalvsz.png")
print("Tot n gal considered",np.sum(myData["N_gal"]))
