# further simplification by taking 2D histogram
# instead of KDE
# using chatgpt to get it faster 

import os
import csv
import glob
import pickle
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from argparse import ArgumentParser 
from scipy.stats import gaussian_kde

from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel import convergence_integrals

#from lenstronomy.Util import util

from fnct import gal_dir,std_sim,test_sim
from get_gal_indexes import get_rnd_gal

from python_tools.tools import mkdir,get_dir_basename
from python_tools.get_res import load_whatever


z_source_max = 5
pixel_num    = 100j
verbose      = True
plot_dnsmap  = True
################################################
# debugging funct.

from proj_part import get_radius,get_z_source,get_dP

def get_dens_map_rotate_hist(Gal,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,plot=plot_dnsmap):
    # try all projection in order to obtain a lens
    proj_index = 0
    res = None
    res = get_dens_map_hist(Gal=Gal,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose,plot=plot)
    raise RuntimeError("DEBUG--Arrived here")
    """
    while proj_index<3:
        try:
            res = get_dens_map_hist(Gal=Gal,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose)
            break
        except AttributeError as Ae:
            print("Error : ")
            print(Ae)
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if res is None:
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
        return res
    """
        
def get_dens_map_hist(Gal,proj_index=0,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,save_res=True,plot=True):
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)

    # given a projection, produce the density map
    # fails if it can't produce a supercritical lens w. z_source<z_source_max
    
    Xstar,Ystar,Zstar = Gal.stars["coords"].T # in Mpc/h
    Xgas,Ygas,Zgas    = Gal.gas["coords"].T   # in Mpc/h
    Xdm,Ydm,Zdm       = Gal.dm["coords"].T    # in Mpc/h
    Xbh,Ybh,Zbh       = Gal.bh["coords"].T    # in Mpc/h
    
    Mstar = Gal.stars["mass"] # in Msun 
    Mgas  = Gal.gas["mass"]  # in Msun 
    Mdm   = Gal.dm["mass"] # in Msun 
    Mbh   = Gal.bh["mass"] # in Msun 
    
    # center around the center of the galaxy
    # correct from cMpc/h to Mpc/h
    # then from Mpc/h to Mpc
    # center of mass is given in Comiving coord 
    # see https://arxiv.org/pdf/1510.01320 D.23 -> given that it is not corrected, it should prob
    # also be corrected for h -> or maybe not?
    Cx,Cy,Cz= Gal.centre*u.Mpc/(Gal.xy_propr2comov*Gal.h) # this should be now in Mpc
    

    print("DEBUG")
    fig, ax = plt.subplots(3)
    for XX,YY,MM,name in zip([Xstar,Xgas,Xdm,Xbh],[Ystar,Ygas,Ydm,Ybh],[Mstar,Mgas,Mdm,Mbh],["star","gas","dm","bh"]):
        x,y = XX*u.Mpc-Cx,YY*u.Mpc-Cy
        """radius = get_radius(x,y)
        xmin = -radius
        ymin = -radius
        xmax = +radius
        ymax = +radius
        """
        ax[0].hist(x,bins=nx,ls='dashed', lw=3,facecolor="None",label=name)#,range=[xmin, xmax])
        ax[0].set_xlabel("X [kpc]")
        #ax[0].set_xlim([xmin,xmax])
        ax[1].hist(y,bins=ny,ls='dashed', lw=3,facecolor="None",label=name)#,range=[ymin, ymax])
        ax[1].set_xlabel("Y [kpc]")
        #ax[1].set_xlim([ymin,ymax])
        ax[2].hist(MM/1e8,bins=nx,ls='dashed', lw=3,facecolor="None",label=name)
        ax[2].set_xlabel("M [1e8 SolMass]")
        ax[2].legend()
    namefig = f"{Gal.proj_dir}/hist1D_{proj_index}_part.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig)

    
    # Concatenate particle properties
    # the unit is Mpc/h -> has to be converted to Mpc. Meaning that the value has to be divided by h 
    # Wrong, they are already given corrected for h
    x = np.concatenate([Xdm, Xstar, Xgas, Xbh])*u.Mpc#/Gal.h # now in Mpc
    y = np.concatenate([Ydm, Ystar, Ygas, Ybh])*u.Mpc#/Gal.h # now in Mpc
    z = np.concatenate([Zdm, Zstar, Zgas, Zbh])*u.Mpc#/Gal.h # now in Mpc
    #  print("QUESTION: do we have to convert also the mass by h") -> I think we have to 
    m = np.concatenate([Mdm, Mstar, Mgas, Mbh])*u.Msun #/Gal.h

    """
    # From https://academic.oup.com/mnras/article/470/1/771/3807086
    # I think that 
    # 1) smoothing make sense for hydrodym, maybe less so for lens modelling
    # 2) we could test how important it is:
    #    - w/o smoothing (all point particles)
    #    - w. smoothing: - Gaussian
    #                    - isoth-sphere
    SmoothStar = Gal.stars["smooth"]  # in Mpc/h
    SmoothGas  = Gal.gas["smooth"]    # in Mpc/h
    SmoothDM   = np.zeros_like(Mdm)   # in Mpc/h -> no smoothing for DM
    SmoothBH   = Gal.bh["smooth"]     # in Mpc/h
    smooth     = np.concatenate([SmoothDM,SmoothStar,SmoothGas,SmoothBH])
    """
    # DEBUG
    max_diam = np.max([np.max(x.value) - np.min(x.value),np.max(y.value) - np.min(y.value),np.max(z.value) - np.min(z.value)])*u.Mpc
    print("DEBUG","max_diam",max_diam)
    
    # projection along given indexes
    # xy : ind=0
    # xz : ind=1
    # yz : ind=2
    if proj_index==0:
        _=True # all as usual
    elif proj_index==1:
        y  = copy(z)
        Cy = copy(Cz)
    elif proj_index==2:
        x  = copy(y)
        Cx = copy(Cy)
        y  = copy(z)
        Cy = copy(Cz)
    x -=Cx
    y -=Cy

    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.asarray(m, dtype=float)
    """
    x      = np.asarray(x.to("kpc").value) #kpc
    y      = np.asarray(y.to("kpc").value) #kpc
    radius = get_radius(x,y)               #kpc
    m      = np.asarray(m.to("solMass").value, dtype=float)  # M_sol
    # Redshift: 
    z_lens = Gal.z
    if verbose:
        print("z_lens",z_lens)
    cosmo   = FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)
    print("DEBUG")
    print(cosmo)
    print("H0",Gal.h*100, "Om0",1-Gal.h)
    
    if verbose:
        print("<Xs>",np.mean(x))
        print("tot mass",np.sum(m))
    
    # I think the following is wrong: it should be centered around 0 bc X,Y already recentered
    """
    xmin = x-radius
    ymin = y-radius
    xmax = x+radius
    ymax = y+radius
    """
    xmin = -radius
    ymin = -radius
    xmax = +radius
    ymax = +radius

    
    print("DEBUG")
    fig, ax = plt.subplots(3)
    ax[0].hist(x,bins=nx,range=[xmin, xmax])
    ax[0].set_xlabel("X [kpc]")
    #ax[0].set_xlim([xmin,xmax])
    ax[1].hist(y,bins=ny,range=[ymin, ymax])
    ax[1].set_xlabel("Y [kpc]")
    #ax[1].set_xlim([ymin,ymax])
    ax[2].hist(m/1e8,bins=nx)
    ax[2].set_xlabel("M [1e8 SolMass]")
    namefig = f"{Gal.proj_dir}/hist1D_{proj_index}.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig)
    # numpy.histogram2d returns H with shape (nx_bins, ny_bins) where H[i,j]
    # counts x-bin i and y-bin j. We transpose to (ny, nx) so rows are y.
    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny],
                                       range=[[xmin, xmax], [ymin, ymax]],
                                       weights=m,density=False)  # if density=True, it normalises it to the total density
    # H is then the distribution of mass for each bin, not the density
    mass_grid = H.T.copy() # Solar Masses
    # H shape: (nx, ny) -> transpose to (ny, nx)

    # area of the (dx/dy) edges of bins:
    dx = (xmax - xmin) / nx #kpc
    dy = (ymax - ymin) / ny #kpc
    # density_ij = M_ij/(Area_bin_ij)
    density = mass_grid / (dx * dy)

    if plot:
        extent = [xmin,xmax,ymin,ymax]
        plt.imshow(np.log10(density),extent=extent, cmap=plt.cm.gist_earth_r,norm="log")
        plt.colorbar()
        #plt.scatter(x,y,c="w",marker=".")
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        namefig = f"{Gal.proj_dir}/hist_densmap_proj_{proj_index}.png"
        plt.savefig(namefig)
        plt.close()
        print("Saved "+namefig)
    # define the z_source:
    # dens now is already in Msun/kpc^2
    """
    dens_Ms_arcsec2 = dens/(dP**2)  # Msun /''^2 
    dens_Ms_kpc2    = dens_Ms_arcsec2*(arcXkpc**2) # Msun/kpc^2
    """   
    print("DEBUG ")
    print("M(density)",np.sum(mass_grid))
    print("M(m)",np.sum(m))
    print("M(gal)/h",Gal.M_tot/Gal.h)
    print("M(gal2)/h",Gal.M/Gal.h)
    dens_Ms_kpc2    = density*u.Msun/(u.kpc*u.kpc)
    print("dx,dy",dx,dy)
    print("area",dx*dy,"kpc^2")
    print("<density>",np.mean(dens_Ms_kpc2))
    z_source        = get_z_source(cosmo,z_lens,dens_Ms_kpc2=dens_Ms_kpc2,z_source_max=z_source_max,verbose=verbose)
    if z_source==0:
        raise AttributeError("Rerun trying different projection")
        
    dP = get_dP(radius*u.kpc,pixel_num) # ''/pix -> to double check that this is correct
    # store the results
    res = [dens_Ms_kpc2,radius,dP,cosmo]
    if save_res:
        with open(Gal.dens_res,"wb") as f:
            pickle.dump(res,f)
        print("Saved "+Gal.dens_res)
    # still consider the dP -> has to convert from kpc/pix to ''/pix
    return res

def sersic_brightness(x,y,n=4,I=10):
    # rotate the galaxy by the angle self.pa
    #x = np.cos(self.pa)*(x-self.ys1)+np.sin(self.pa)*(y2-self.ys2)
    #y = -np.sin(self.pa)*(y1-self.ys1)+np.cos(self.pa)*(y2-self.ys2)
    # include elliptical isophotes
    try:
        # ugly but useful
        x=x.value
        y=y.value
    except:
        pass
    r = np.sqrt((x)**2+(y)**2)
    # brightness at distance r
    bn = 1.992*n - 0.3271
    re = 5.0
    brightness = I*np.exp(-bn*((r/re)**(1.0/n)-1.0))
    return brightness


if __name__=="__main__":
    parser = ArgumentParser(description="Project particles into a mass sheet - histogram version")
    parser.add_argument("-dn","--dir_name",dest="dir_name",type=str, help="Directory name",default="proj_part_hist")
    parser.add_argument("-pxn","--pixel_num",dest="pixel_num",type=int, help="Pixel number",default=pixel_num.imag)
    parser.add_argument("-zsm","--z_source_max",dest="z_source_max",type=float, help="Maximum source redshift",default=z_source_max)
    parser.add_argument("-nrr", "--not_rerun", dest="rerun", 
                        default=True,action="store_false",help="if True, rerun code")
    parser.add_argument("-pl", "--plot", dest="plot", 
                        default=False,action="store_true",help="Plot dens map")
    parser.add_argument("-v", "--verbose", dest="verbose", 
                        default=False,action="store_true",help="verbose")
    args          = parser.parse_args()
    pixel_num     = args.pixel_num*1j
    rerun         = args.rerun
    dir_name      = args.dir_name
    verbose       = args.verbose
    z_source_max  = args.z_source_max
    plot          = args.plot
    if rerun:
        #print("DEBUG -- USING test sym")
        #Gal = get_rnd_gal(sim=test_sim,check_prev=False,reuse_previous=False,min_mass="1e13",max_z="1")
        Gal = get_rnd_gal(sim=std_sim,check_prev=False,reuse_previous=False,min_mass="1e13",max_z="1")
        Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}_{Gal.Name}/"
        mkdir(Gal.proj_dir)
        Gal.dens_res = f"{Gal.proj_dir}/dens_res.pkl"
    else:
        # find an already "random" galaxy
        dens_res_path = glob.glob(gal_dir+"/snap_*/"+dir_name+"_*/dens_res.pkl")
        dens_res = np.random.choice(dens_res_path)
        class empty_class():
            def __init__(self):
                return None
        Gal = empty_class()
        Gal.dens_res = dens_res
        Gal.proj_dir = get_dir_basename(dens_res)[0]
        
    if verbose:
        print("Assumptions: We are considering the maximum source redshift to be ",z_source_max)
        if int(pixel_num.imag)<500:
            print("Warning: running test")
        elif int(pixel_num.imag)>=1000:
            print("Warning: running very long")


    try:
        if rerun:
            raise RuntimeError("Rerunning anyway")
        dens_Ms_kpc2,radius,dP,cosmo = load_whatever(Gal.dens_res)
        
        if len(dens_Ms_kpc2)!=int(pixel_num.imag):
            print("DEBUG")
            print(len(dens_Ms_kpc2),int(pixel_num.imag))
            print("Num pixel != of the wanted number of pixel, Rerunning")
            raise RuntimeError()
    except:
        dens_Ms_kpc2,radius,dP,cosmo = get_dens_map_rotate_hist(Gal=Gal,pixel_num=pixel_num,
                                                           z_source_max=z_source_max,verbose=True,plot=plot)#verbose=verbose)
    Xg, Yg  = np.mgrid[-radius:radius:pixel_num, -radius:radius:pixel_num] # kpc
    arcXkpc = cosmo.arcsec_per_kpc_proper(Gal.z) # ''/kpc
    
    # create lensed image:
    # dPix it's given by the pixel_num  
    cosmo_dd  = cosmo.angular_diameter_distance(z_lens).to("kpc")   #kpc
    cosmo_ds  = cosmo.angular_diameter_distance(z_source).to("kpc") #kpc
    cosmo_dds = cosmo.angular_diameter_distance_z1z2(z1=z_lens,z2=z_source).to("kpc") #kpc
    
    # Sigma_Crit = D_s*c^2/(4piG * D_d*D_ds)
    Sigma_Crit  = (cosmo_ds*const.c**2)/(4*np.pi*const.G*cosmo_dds*cosmo_dd)
    Sigma_Crit  = Sigma_Crit.to("Msun /(kpc kpc)")    
    #Sigma_Crit /= arcXkpc**2  # Msun / ''^2 
    #Sigma_Crit *= dP**2       # Msun/pix^2
    kappa_grid  = dens_Ms_kpc2/Sigma_Crit # 1 
    
    # NOTE: a lens is such only if kappa_map > 1 at least at one point
    # 1st assumption: we will realistically assume that such point is the center of the galaxy -> position the source there
    # check that 
    assert(np.any(kappa_grid>1))#this should be true given our get_z_source function
    # if not: check if there is a realistic (higher)  z_source  to make such that the 
    
    # add masking
    # add padding
    
    # I don't have to go trough the potential->deflection_from_kappa_grid ->later needed when we'll have to add the LOS effects
    """
        #num_pot    = convergence_integrals.potential_from_kappa_grid(kappa_grid, dP) # ''^2 / pix^2 -> this is odd but that's just how it is computed
        #print("numpot,unit",num_pot.unit)
        #numPot_div_dP = num_pot/dP # -> ''/pix 
        #raise("The latest unit problem is here: what exactly should it be the correct way to obtain alpha in arcsecs?")
        #print("numPot_div_dP,unit",numPot_div_dP.unit)
        #num_aRa,num_aDec = np.gradient(numPot_div_dP)/dP # this smh divide(?) by pix? and therefore the unit of the output must be only ''  -> set it by hand
        -> the function doesn't respect the units
    """
    ##
    """
    Deflection angle :math:`\\vec {\\alpha }}` from a convergence grid :math:`\\kappa`.
    
    .. math::
        {\\vec {\\alpha }}({\\vec {\\theta }})={\\frac {1}{\\pi }}
        \\int d^{2}\\theta ^{\\prime }{\\frac {({\\vec {\\theta }}-{\\vec {\\theta }}^{\\prime })
        \\kappa ({\\vec {\\theta }}^{\\prime })}{|{\\vec {\\theta }}-{\\vec {\\theta }}^{\\prime }|^{2}}}
    
    The computation is performed as a convolution of the Green's function with the convergence map using FFT.
    
    :param kappa: convergence values for each pixel (2-d array)
    :param grid_spacing: scale of an individual pixel (per axis) of grid
    :return: numerical deflection angles in x- and y- direction over the convergence grid points    
    """
    num_aRa,num_aDec = convergence_integrals.deflection_from_kappa_grid(kappa_grid,dP.value) # this function does not respect dimensions
    num_aRa  *=u.arcsec
    num_aDec *=u.arcsec
    #print("num_aRa,unit",num_aRa.unit) #*u.arcsec
    
    #ra,dec = util.array2image(RAg),util.array2image(DECg)
    print("DEBUG: shape Xg,num_aRa",np.shape(Xg),np.shape(num_aRa))
    
    ra  = Xg.reshape(num_aRa.shape)*arcXkpc/u.pix  # not entirely sure this unit is correct, but anyway it's just book-keeping
    dec = Yg.reshape(num_aDec.shape)*arcXkpc/u.pix  
    print("DEBUG")
    plt.imshow(ra.value)
    plt.colorbar()
    plt.title("Ra source")
    im_name = f"{Gal.proj_dir}/ra_src.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(dec.value)
    plt.colorbar()
    plt.title("Dec source")
    im_name = f"{Gal.proj_dir}/dec_src.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(np.log10(sersic_brightness(ra,dec)) )
    plt.colorbar()
    plt.title("log Source")
    im_name = f"{Gal.proj_dir}/src.pdf"
    plt.savefig(im_name)
    plt.close()
    
    plt.imshow(num_aRa.value)
    plt.colorbar()
    plt.title("Ra deflection")
    im_name = f"{Gal.proj_dir}/alpha_ra.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(num_aDec.value)
    plt.colorbar()
    plt.title("Dec deflection")
    im_name = f"{Gal.proj_dir}/alpha_dec.pdf"
    plt.savefig(im_name)
    plt.close()
    print("DEBUG")
    
    ra_im  = ra.value-num_aRa.value
    dec_im = dec.value-num_aDec.value
    lensed_im = sersic_brightness(ra_im,dec_im)
    plt.imshow(np.log10(lensed_im))
    plt.colorbar()
    plt.title("Log Lensed Sersic image")
    im_name = f"{Gal.proj_dir}/lensed_im.pdf"
    plt.savefig(im_name)
    plt.close()
    print("Saving "+im_name)
    
    # for convenience, I link the result to the tmp dir
    os.unlink("./tmp/"+dir_name)
    os.symlink(Gal.proj_dir[:-1],"./tmp/.")
    
    print("Success")
