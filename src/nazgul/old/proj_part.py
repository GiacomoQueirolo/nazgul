# project mass to a 2D 

# smotthing: read appending A of 
    # https://academic.oup.com/mnras/article/470/1/771/3807086

# to do: rewrite code in order to have a funct. to create mass distribution along all axes
# then complicate it by saving stuff

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

from fnct import gal_dir,std_sim
from get_gal_indexes import get_rnd_gal

from python_tools.tools import mkdir,get_dir_basename
from python_tools.get_res import load_whatever


z_source_max = 5
verbose = True
pixel_num = 100j
################################################
# debugging funct.
"""
def _namestr(obj, namespace):
    try:
        return [name for name in namespace if namespace[name] is obj]
    except:
        return "name not found"

def _debug(var=None,namespace=globals()):
    print("DEBUG")
    if var is not None:
        name_var = _namestr(var,namespace)
        if len(name_var)==1:
            name_var = name_var[0]
        print(name_var,":",var)
"""
################################################

def basic_get_radius(RAs,DECs):
    # define a "radius" (more like 1/2 of the edge-lenght) of the grid used to sample the density
    
    # We want to have the same pixelscale in the 2Dim
    # the obvioious way would give RA and DEC that might not be on the same range
    # but we create a grid with the same number of points for both
    # we rather would go as such: redefine the ranges such that the number of pixels 
    # and the ranges are the same (but there might be some empty, ie 0 density, pixels
    # for either of the two dimensions)
    ramin  = RAs.min()
    ramax  = RAs.max()
    #rangeRa = ramax - ramin
    decmin = DECs.min()
    decmax = DECs.max()
    #rangeDec = decmax-decmin
    # we have the advantage that the center is set to 0 ->
    radius =  max([0-ramin,ramax-0,0-decmin,decmax-0])
    # verify:
    assert(ramin>=-radius)
    assert(decmin>=-radius)
    assert(ramax<=radius)
    assert(decmax<=radius)
    return radius

def get_radius(RAs,DECs,sigmas=6):
    # cut-out outlier particles 
    rad_max = basic_get_radius(RAs,DECs)
    # we take 6 <sigmas> of 
    rad_min = sigmas*(np.std(RAs)+np.std(DECs))/2
    return np.min([rad_max,rad_min])


def get_z_source(cosmo,z_lens,dens_Ms_kpc2,z_source_max=z_source_max,verbose=verbose):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        # to do : deal with this
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")
        #return 0
    try:
        dens_Ms_kpc2.value
    except:
        # dens_Ms_kpc2 is already given in Msun/kpc^2
        dens_Ms_kpc2 *= u.Msun/(u.kpc**2)
    assert dens_Ms_kpc2.unit==u.solMass/(u.kpc**2)
    print("DEBUG z_lens",z_lens)
    print("DEBU cosmo",cosmo)
    print("DEBUG cosmo.angular_diameter_distance(z_lens)",cosmo.angular_diameter_distance(z_lens))
    print("DEBUG NOTE: the approx MW surf.dens. is 2*1e9Msun/kpc^2")
    print("DEBUG np.max(dens_Ms_kpc2)",np.max(dens_Ms_kpc2))
    print("DEBUG 4*np.pi*const.G",4*np.pi*const.G)
    print("DEBUG cosmo.angular_diameter_distance(z_lens)",cosmo.angular_diameter_distance(z_lens))
    print("DEBUG (const.c**2) ",(const.c**2) )
    max_DsDds = np.max(dens_Ms_kpc2)*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2) 
    print("DEBUG\n","np.max(dens_Ms_kpc2)",np.max(dens_Ms_kpc2.to("1e9Msun/kpc^2")))
    print("DEBUG\n","max_DsDds",max_DsDds)
    max_DsDds = max_DsDds.to("") # assert(max_DsDds.unit==u.dimensionless_unscaled) -> equivalent
    max_DsDds = max_DsDds.value # dimensionless
    print("DEBUG\n","max_DsDds",max_DsDds)
    #z_source_range = np.linspace(z_lens,z_source_max,100) # it's a very smooth funct->
    min_DsDds = cosmo.angular_diameter_distance(z_source_max)/cosmo.angular_diameter_distance_z1z2(z_lens,z_source_max) # this is the minimum
    min_DsDds = min_DsDds.to("") # dimensionless
    min_DsDds = min_DsDds.value
    
    z_source_range = np.linspace(z_lens+0.09,z_source_max,100) # it's a very smooth funct->
    DsDds = np.array([cosmo.angular_diameter_distance(z_s).to("Mpc").value/cosmo.angular_diameter_distance_z1z2(z_lens,z_s).to("Mpc").value for z_s in z_source_range])
    if not min_DsDds<max_DsDds:
        # to do: deal with this kind of output
        if verbose:
            print("Warning: the minimum z_source needed to have a lens is higher than the maximum allowed z_source")
            # debug:
            # verify the computation
            print("DEBUG")
            plt.plot(z_source_range,DsDds,ls="-",c="k",label=r"D$_{\text{s}}$/D$_{\text{ds}}$(z$_{source}$)")
            plt.axhline(max_DsDds,ls="--",c="r",label=r"max(dens)*4$\pi$*G*$D_l$/c$^2$")
            plt.legend()
            name = "tmp/DsDds.pdf"
            plt.savefig(name)
            print("Saved "+name)
        return 0
    else:
        # note that the successful test means only that there is AT LEAST 1 PIXEL that is supercritical
        minimise     = np.abs(DsDds-max_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        # select a random source within the range
        z_source = np.random.uniform(z_source_min,z_source_max,1)[0]
        if verbose:
            print("Minimum z_source:",np.round(z_source_min,2))
            print("Chosen z_source:", np.round(z_source,2))
        return z_source
        
def get_dens_map_rotate(Gal,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,plot=False):
    # try all projection in order to obtain a lens
    proj_index = 0
    res = None
    while proj_index<3:
        try:
            res = get_dens_map_main(Gal=Gal,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose,plot=plot)
            break
        except AttributeError:
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if res is None:
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
        return res
        
def get_dP(radius_kpc,pixel_num,arcXkpc=None,cosmo=None,Gal=None):
    if arcXkpc is None:
        if cosmo is None or Gal is None:
            raise RuntimeError("Give either arcXkpc or cosmo and Gal")
        arcXkpc = cosmo.arcsec_per_kpc_proper(Gal.z) # ''/kpc
            
    try:
        radius_kpc.value 
    except AttributeError:
        # hoping the radius is actually inserted in kpc
        radius_kpc *= u.kpc 
    return 2*radius_kpc*arcXkpc.to("arcsec/kpc")/(int(pixel_num.imag)*u.pix) #''/pix
    
def get_dens_map_main(Gal,proj_index=0,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,save_res=True,plot=True):
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
    
    
    # Concatenate particle properties
    # the unit is Mpc/h -> has to be converted to Mpc. Meaning that the value has to be divided by h
    x = np.concatenate([Xdm, Xstar, Xgas, Xbh])*u.Mpc/Gal.h # now in Mpc
    y = np.concatenate([Ydm, Ystar, Ygas, Ybh])*u.Mpc/Gal.h # now in Mpc
    z = np.concatenate([Zdm, Zstar, Zgas, Zbh])*u.Mpc/Gal.h # now in Mpc
    #  print("QUESTION: do we have to convert also the mass by h") -> I think we have to
    m = np.concatenate([Mdm, Mstar, Mgas, Mbh])*u.Msun/Gal.h

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
    print("DEBUG\n","max_diam",max_diam)
    
    # center around the center of the galaxy
    # correct from cMpc/h to Mpc/h
    # then from Mpc/h to Mpc
    Cx,Cy,Cz= Gal.centre*u.Mpc/(Gal.xy_propr2comov*Gal.h) # this should be now in Mpc
    
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
        Cz = copy(Cz)
    x -=Cx
    y -=Cy

    
    # Redshift: 
    z_lens = Gal.z
    if verbose:
        print("z_lens",z_lens)
    cosmo   = FlatLambdaCDM(H0=Gal.h*100, Om0=1-Gal.h)
    #arcXkpc = cosmo.arcsec_per_kpc_proper(Gal.z) # ''/kpc

    # We don't need to sample it in arcsec, we can keep it in kpc
    """
    # note: gal coords are in Mpc -> conv to ''
    RAs  = x*arcXkpc # ''
    RAs  = RAs.to("arcsec") 
    DECs = y*arcXkpc # ''
    DECs = DECs.to("arcsec") 
    
    if verbose:
        print("<RAs>",np.mean(RAs))
        print("tot mass",np.sum(m))
    
    # fit the mass distribution w KDE
    kde       = gaussian_kde(np.array([RAs.value,DECs.value]),weights=m.value)# Msun/pix^2 (but reported dimensionless)

    radius    = get_radius(RAs,DECs)
    dP        = 2*radius/(int(pixel_num.imag)*u.pix) # ''/pix
    RAg, DECg = np.mgrid[-radius:radius:pixel_num, -radius:radius:pixel_num]
    positions = np.vstack([RAg.ravel(), DECg.ravel()])
    fit_kde   = kde(positions)*u.Msun/(u.pix**2)  # Msun/pix^2 -> the kde give density as function of the pixel number, not the coordinates
    """
    if verbose:
        print("<Xs>",np.mean(x))
        print("tot mass",np.sum(m))
    
    # fit the mass distribution w KDE
    x_kde,y_kde = x.to("kpc").value,y.to("kpc").value
    kde         = gaussian_kde(np.array([x_kde,y_kde]),weights=m.value)# Msun/kpc^2 (but reported dimensionless)

    radius    = get_radius(x_kde,y_kde) #kpc
    Xg, Yg    = np.mgrid[-radius:radius:pixel_num, -radius:radius:pixel_num]
    positions = np.vstack([Xg.ravel(), Yg.ravel()])
    fit_kde   = kde(positions)*u.Msun/(u.kpc**2)  # Msun/kpc^2 
    #-> the kde give density as function of the pixel number, not the coordinates
    if verbose:
        print("fit_kde sum",np.sum(fit_kde))
        print("fit_kde median",np.median(fit_kde))
        print("fit_kde max",np.max(fit_kde))
    dens    = np.reshape(fit_kde, Xg.shape).T # Msun/pix^2 
    if plot:
        # plot dens map
        """
        fig, ax = plt.subplots(2)
        ax[0].contour(Xg, Xg, dens.value)
        ax[1].imshow(dens.value, cmap=plt.cm.gist_earth_r)
        namefig = f"{Gal.proj_dir}/kde_densmap.pdf"
        plt.savefig(namefig)
        plt.close()
        print("Saved "+namefig)
        """
        try:
            ramin  = x_kde.min()
            ramax  = x_kde.max()
            decmin = y_kde.min()
            decmax = y_kde.max()
            extent = [ramin,ramax,decmin,decmax]
            dens   = fit_kde.reshape(Xg.shape).T
            plt.imshow(dens.value,extent=extent, cmap=plt.cm.gist_earth_r)
            plt.scatter(x_kde,y_kde,c="w",marker=".")
            plt.contour(Xg, Yg, dens.T.value,extent=extent)
            plt.xlim([ramin,ramax])
            plt.ylim([decmin,decmax])
            namefig = f"{Gal.proj_dir}/kde_densmap.pdf"
            plt.savefig(namefig)
            plt.close()
            print("Saved "+namefig)
        except TypeError as e:
            print("while plotting, encountered the following error:")
            print(e)
            print("Ignored and continued w/o plotting")




    # define the z_source:
    # dens now is already in Msun/kpc^2
    """
    dens_Ms_arcsec2 = dens/(dP**2)  # Msun /''^2 
    dens_Ms_kpc2    = dens_Ms_arcsec2*(arcXkpc**2) # Msun/kpc^2
    """   
    dens_Ms_kpc2    = dens 
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
    parser = ArgumentParser(description="Project particles into a mass sheet")
    parser.add_argument("-dn","--dir_name",dest="dir_name",type=str, help="Directory name",default="proj_part")
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
        dens_Ms_kpc2,radius,dP,cosmo = get_dens_map_rotate(Gal=Gal,pixel_num=pixel_num,
                                                           z_source_max=z_source_max,verbose=verbose,plot=plot)
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
    """
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
    """
    fg,ax=plt.subplots(2,3,figsize=(16,8))
    ax[0][0].imshow(ra.value)
    #ax[0].colorbar()
    ax[0][0].set_title("Ra source")
    #im_name = f"{Gal.proj_dir}/ra_src.pdf"
    #im_name = f"tmp/ra_src.pdf"
    #plt.savefig(im_name)
    #plt.close()
    ax[0][1].imshow(dec.value)
    #ax[1].colorbar()
    ax[0][1].set_title("Dec source")
    #im_name = f"{Gal.proj_dir}/dec_src.pdf"
    #im_name = f"tmp/dec_src.pdf"
    #.savefig(im_name)
    #plt.close()
    ax[0][2].imshow(np.log10(sersic_brightness(ra,dec)) )
    #ax[2].colorbar()
    ax[0][2].set_title("log Source")
    #im_name = f"{Gal.proj_dir}/src.pdf"
    """
    im_name = f"src.pdf"
    plt.savefig(im_name)
    plt.show()
    plt.close()
    
    fg,ax=plt.subplots(1,3,figsize=(16,8))
    """
    
    ax[1][0].imshow(num_aRa)
    #plt.colorbar()
    ax[1][0].set_title("Ra deflection")
    ax[1][1].imshow(num_aDec)
    ax[1][1].set_title("Dec deflection")
    ra_im  = ra.value-num_aRa
    dec_im = dec.value-num_aDec
    lensed_im = sersic_brightness(ra_im,dec_im)
    ax[1][2].imshow(np.log10(lensed_im))
    ax[1][2].set_title("Log Lensed Sersic image")
    #im_name = f"tmp/lensed_im.pdf"
    im_name = f"lensed_im.pdf"
    plt.savefig(im_name)
    plt.show()
    plt.close()
    print("Saving "+im_name)
    
    # for convenience, I link the result to the tmp dir
    os.unlink("./tmp/"+dir_name)
    os.symlink(Gal.proj_dir[:-1],"./tmp/.")
    
    print("Success")