#WIP
import numpy as np
from fnct import Galaxy,std_sim,sim_path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nazgul.get_gal_indexes import get_rnd_gal
# obtained from http://virgodb.dur.ac.uk:8080/Eagle/MyDB
# with the followin command:
"""
SELECT   
        gal.Redshift as z,   
        gal.Image_Face as face, 
        gal.CentreOfMass_x as x, 
        gal.CentreOfMass_y as y, 
        gal.CentreOfMass_z as z,
        gal.GroupNumber as Gn,
        gal.SubGroupNumber as SGn
   FROM
        RefL0025N0752_SubHalo as gal,   
        RefL0025N0752_SubHalo as ref   
   WHERE   
        ref.GalaxyID=1848116 and -- GalaxyID at z=1   
        ((gal.SnapNum > ref.SnapNum and ref.GalaxyID   
        between gal.GalaxyID and gal.TopLeafID) or    
        (gal.SnapNum <= ref.SnapNum and gal.GalaxyID    
        between ref.GalaxyID and ref.TopLeafID))   
   ORDER BY   
        gal.Redshift
"""

#centre3,gn,sgn = np.array([14.434582,24.12927,19.225077]),22,0

gl = get_rnd_gal(sim=std_sim,min_z=1.9,max_z=2.02,reuse_previous=True)
# Galaxy(Gn=gn,SGn=sgn,CntX=cntre3[0],CntY=centre3[1],CntZ=centre3[2],z=0)

def plot_gal(gl):
    xyz_dm  = gl.dm["coords"].T
    xyz_str = gl.stars["coords"].T
    xyz_gas = gl.gas["coords"].T
    xyz_bh  = gl.bh["coords"].T
    
    x_dm  = xyz_dm[0]
    x_str = xyz_str[0]
    x_gas = xyz_gas[0]
    x_bh  = xyz_bh[0]
    
    
    m_str = gl.stars["mass"]
    m_dm  =  gl.dm["mass"]
    m_gas = gl.gas["mass"]
    m_bh  = gl.bh["mass"]
    
    # bh can be ignored
    
    b = 40
    plt.style.use('classic')
    plt.hist(x_str,bins=b,weights=m_str, color="yellow",label="stars",alpha=.3)
    plt.hist(x_gas,bins=b,weights=m_gas, color="violet",label="gas",alpha=.3)
    plt.hist(x_dm,bins=b,weights=m_dm,color="grey",label="dm",alpha=.3)
    plt.yscale("log")
    plt.ylabel(r"M [M$_\odot$]")
    plt.xlabel("X coord [Mpc]")
    plt.legend()
    plt.title("Mass Histogram For Different Particles of 1 EAGLE Gal.")
    nm = sim_path+"/mHistGal1.png"
    print(f"Saving {nm}")
    plt.savefig(nm)
    plt.close()
    
    
    xy_str = xyz_str[:-1]
    xy_dm  = xyz_dm[:-1]
    xy_gas = xyz_gas[:-1]
    xy_bh  = xyz_bh[:-1]
    dxy = 3*np.max(np.std(xy_dm,axis=1))
    xm,ym=np.mean(xy_dm,axis=1)

    fg,axes = plt.subplots(2,2)
    ax = axes[0][0]
    ax.set_title("Stars particles")
    im0 = ax.scatter(*xy_str,c=np.log(m_str),alpha=.2,cmap="coolwarm_r",marker=".")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label="log(Star Mass)")

    ax = axes[0][1]
    ax.set_title("DM particles")
    im0 = ax.scatter(*xy_dm,c=np.log(m_dm),alpha=.2,cmap="winter",marker=".")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label="log(DM Mass)")

    ax = axes[1][0]
    ax.set_title("Gas particles")
    im0 = ax.scatter(*xy_gas,c=np.log(m_gas),alpha=.2,cmap="coolwarm_r",marker=".")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label="log(Gas Mass)")
    
    ax = axes[1][1]
    ax.set_title("Blackholes particles")
    im0 = ax.scatter(*xy_bh,c=np.log(m_bh),alpha=.2,cmap="coolwarm_r",marker=".")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label="log(BH Mass)")

    for axi in axes:
        for axii in axi:    
            axii.set_xlim(xm-dxy,xm+dxy)
            axii.set_ylim(ym-dxy,ym+dxy)
            axii.set_xlabel("X [Mpc]")
            axii.set_ylabel("Y [Mpc]")
                
    nm = sim_path+"/PartDistrGal.png"
    print(f"Saving {nm}")
    plt.savefig(nm)
    plt.close()
    
    
