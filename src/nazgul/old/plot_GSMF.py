# from The_eagle_simulations_of_galaxy_formation__public_release_of_halo_and_galaxy_catalogues.pdf
# p 15 
# python example to get Gal Stellar Mass function
import numpy as np
import matplotlib.pyplot as plt
from sql_connect import exec_query
# Array
mySims = np.array([('RefL0100N1504', 100.) , ('AGNdT9L0050N0752', 50.) , ('RecalL0025N0752', 25.)])
# This u s e s t h e e a g l e S q l T o o l s module t o c o n n e c t t o t h e d a t a b a s e w i t h your username and password .
# I f t h e password i s not g i v e n , t h e module w i l l prompt f o r i t .

if __name__=="__main__":
    for sim_name, sim_size in mySims:
        print(sim_name)
        myQuery = "SELECT \
        0.1+floor(log10(AP.Mass_Star)/0.2)*0.2 as mass, \
        count(*) as num \
        FROM \
            %s_SubHalo as SH,\
            %s_Aperture as AP \
        WHERE \
            SH.GalaxyID = AP.GalaxyID and \
            AP.ApertureSize = 30 and \
            AP.Mass_Star > 1e8 and \
            SH.SnapNum = 27 \
        GROUP BY \
            0.1+floor(log10(AP.Mass_Star)/0.2)*0.2 \
        ORDER BY \
            mass"%(sim_name,sim_name)
        # Execut
        myData = exec_query(myQuery)
        # Normalize by volume and b i n w i d t h .
        hist = myData['num'][:]/float(sim_size)**3.
        hist = hist/0.2
        plt.plot(myData['mass'],np.log10(hist),label=sim_name,linewidth =2)
    # Label p l o t .
    plt.xlabel(r'log$_{10}$M${_*}$[M$_{\odot}$]',fontsize =20)
    plt.ylabel(r'log$_{10}$dn/dlog$_{10}$(M$_{*}$ )[ cMpc $ ^{−3}$ ]',fontsize=20)
    plt.tight_layout()
    plt.legend()
    
    plt.savefig('GSMF.pdf')
    plt.close()
