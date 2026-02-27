# copied from proj_part_hist but now using the "ParticleGalaxy" class

raise DeprecationWarning("Use project_gal_AMR.py instead - mostly the coords are now wrong")
# due to the clipping of extremes

import os
import csv
import glob
import pickle
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from argparse import ArgumentParser 
import scipy
from scipy.stats import gaussian_kde

from astropy import units as u
from astropy import constants as const

from lenstronomy.LensModel import convergence_integrals

from ParticleGalaxy import get_rnd_PG,get_z_source,get_dP

from python_tools.tools import mkdir,get_dir_basename
from python_tools.get_res import load_whatever
from python_tools.conversion import e1e2_from_qphi

#from fnct import Sersic
from lenstronomy.LightModel.Profiles.sersic import SersicElliptic
z_source_max = 4
pixel_num    = 600j #200j
verbose      = True
plot_dnsmap  = True
def_radius   = 100*u.kpc #70*u.kpc

from project_gal import dir_name, prep_Gal_projpath
def get_dens_map_rotate_hist(Gal,radius=def_radius,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,plot=plot_dnsmap):
    # try all projection in order to obtain a lens
    proj_index = 0
    """
    #DEBUG
    res = get_dens_map_hist(Gal=Gal,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose,plot=plot)
    raise RuntimeError("DEBUG--Arrived here")
    """
    kw_res = None
    while proj_index<3:
        try:
            kw_res = get_dens_map_hist(Gal=Gal,radius=radius,proj_index=proj_index,pixel_num=pixel_num,
                                    z_source_max=z_source_max,verbose=verbose)
            break
        except AttributeError as Ae:
            print("Error : ")
            print(Ae)
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if kw_res is None:
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
        return kw_res

def get_dens_map_hist(Gal,proj_index=0,pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose,save_res=True,plot=True,radius=def_radius,DEBUG=False):
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)

    # given a projection, produce the density map
    # fails if it can't produce a supercritical lens w. z_source<z_source_max
    
    Xstar,Ystar,Zstar = Gal.stars["coords"].T # in Mpc
    Xgas,Ygas,Zgas    = Gal.gas["coords"].T   # in Mpc
    Xdm,Ydm,Zdm       = Gal.dm["coords"].T    # in Mpc
    Xbh,Ybh,Zbh       = Gal.bh["coords"].T    # in Mpc
    
    Mstar = Gal.stars["mass"] # in Msun 
    Mgas  = Gal.gas["mass"]  # in Msun 
    Mdm   = Gal.dm["mass"] # in Msun 
    Mbh   = Gal.bh["mass"] # in Msun 
    
    # center around the center of the galaxy
    # center of mass is given in Comiving coord 
    # see https://arxiv.org/pdf/1510.01320 D.23 
    # ->  it's given in cMpc (not cMpc/h) fsr
    Cx,Cy,Cz = Gal.centre*u.Mpc/(Gal.xy_propr2comov) # (now) Mpc
    if DEBUG:
        print("DEBUG")
        # mass factor for particles
        fact_M = 5e4
        fig, ax = plt.subplots(3)
        for XX,YY,MM,name in zip([Xstar,Xgas,Xdm,Xbh],[Ystar,Ygas,Ydm,Ybh],[Mstar,Mgas,Mdm,Mbh],["star","gas","dm","bh"]):
            xx,yy = XX*u.Mpc-Cx,YY*u.Mpc-Cy
            mm    = MM/fact_M
            ax[0].hist(xx,bins=nx,alpha=.5,label=name)
            ax[1].hist(yy,bins=ny,alpha=.5,label=name)
            ax[2].hist(mm,alpha=.5,label=name)
        ax[0].axvline(0,ls="--",label="centre")
        ax[1].axvline(0,ls="--",label="centre")
        ax[0].set_xlabel("X [kpc]")
        ax[1].set_xlabel("Y [kpc]")
        ax[2].set_xlabel(f"M [{str(fact_M)} SolMass]")
        ax[2].legend()
        namefig = f"./tmp/NG_hist1D_{proj_index}_part.png"
        plt.tight_layout()
        plt.savefig(namefig)
        plt.close()
        print("Saved "+namefig)
    
    # Concatenate particle properties
    # already in proper/physical units (corrected for h as well)
    x = np.concatenate([Xdm, Xstar, Xgas, Xbh])*u.Mpc#/Gal.h # now in Mpc
    y = np.concatenate([Ydm, Ystar, Ygas, Ybh])*u.Mpc#/Gal.h # now in Mpc
    z = np.concatenate([Zdm, Zstar, Zgas, Zbh])*u.Mpc#/Gal.h # now in Mpc
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
    if DEBUG:
        max_diam = np.max([np.max(x.value) - np.min(x.value),np.max(y.value) - np.min(y.value),np.max(z.value) - np.min(z.value)])*u.Mpc
        print("DEBUG","max_diam",np.round(max_diam,2))
        print(Cx,Cy,Cz)
        print(np.mean(x),np.mean(y),np.mean(z))
        print(np.sum(x*m)/np.sum(m),np.sum(y*m)/np.sum(m),np.sum(y*m)/np.sum(m))
    x -= Cx
    y -= Cy
    z -= Cz
    if DEBUG:
        print("DEBUG np.mean(x),Cx",np.mean(x),Cx)
        print("DEBUG np.mean(y),Cy",np.mean(y),Cy)
        print("DEBUG np.median(y),Cy",np.median(y))
        print("DEBUG np.std(y)",np.std(y))
        print("DEBUG np.mean(z),Cz",np.mean(z),Cz)
    # projection along given indexes
    # xy : ind=0
    # xz : ind=1
    # yz : ind=2
    if proj_index==0:
        _=True # all as usual
    elif proj_index==1:
        y  = copy(z)
    elif proj_index==2:
        x  = copy(y)
        y  = copy(z)
    x  = np.asarray(x.to("kpc").value) #kpc
    y  = np.asarray(y.to("kpc").value) #kpc
    m  = np.asarray(m.to("solMass").value, dtype=float)  # M_sol
    #radius = def_radius #kpc 
    if radius==def_radius:
        print("NOTE: taking a small radius -",radius)
    elif radius is None:
        raise RuntimeError("TODO: implement")
        radius = get_radius(x,y) #kpc
        
    # Redshift: 
    z_lens = Gal.z
    if verbose:
        print("z_lens",z_lens)
    cosmo = Gal.cosmo
    if DEBUG:
        print("DEBUG")
        print("cosmo",cosmo)
        print("H0",Gal.h*100, "Om0",1-Gal.h)
    
    if DEBUG:
        print("<X> [kpc]",np.round(np.mean(x),3))
        print("<Y> [kpc]",np.round(np.mean(y),3))
        print("radius ",np.round(radius,3))
        print("tot mass [1e8 M_sol]",np.round(np.sum(m)/1e8,3))
        
    # X,Y already recentered around 0
    xmin = -radius.value
    ymin = -radius.value
    xmax = +radius.value
    ymax = +radius.value

    if DEBUG:    
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
        #namefig = f"{Gal.proj_dir}/hist1D_{proj_index}.png"
        namefig = f"./tmp/NG_hist1D_{proj_index}.png"
        plt.tight_layout()
        plt.savefig(namefig)
        plt.close()
        print("Saved "+namefig)
    # numpy.histogram2d returns H with shape (nx_bins, ny_bins) where H[i,j]
    # counts x-bin i and y-bin j. We transpose to (ny, nx) so rows are y.
    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny],
                                       range=[[xmin, xmax], [ymin, ymax]],
                                       weights=m,density=False)  
    #                           if density=True, it normalises it to the total density
    
    # H is then the distribution of mass for each bin, not the density
    mass_grid = H.T.copy() # Solar Masses
    # H shape: (nx, ny) -> transpose to (ny, nx)

    # area of the (dx/dy) bins:
    dx = np.diff(xedges) #kpc #==(xmax - xmin) / nx 
    dy = np.diff(yedges) #kpc
    # density_ij = M_ij/(Area_bin_ij)
    density = mass_grid / (dx * dy)
    if plot:

        # DEBUG
        try:
            log10_dens = np.log10(density)
            extent = [xmin,xmax,ymin,ymax]
            if np.isneginf(log10_dens).any():
                print("log10(dens) has -inf")
                log10_dens[np.where(np.isneginf(log10_dens))[0]]= -1e4
            plt.imshow(log10_dens,extent=extent, cmap=plt.cm.gist_earth_r,norm="log",label="Density [Msun/kpc^2]")
            plt.colorbar()
            #plt.scatter(x,y,c="w",marker=".")
            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            #
            if DEBUG:
                namefig = f"tmp/NG_proj_hist_densmap_{proj_index}.png"
            else:
                namefig = f"{Gal.proj_dir}/hist_densmap_proj_{proj_index}.png"
            plt.savefig(namefig)
            plt.close()
            print("Saved "+namefig)
        except ValueError as e:
                
            print("Failed plot due to error "+str(e))
            print("extent",extent)
            
            tmp_name = "tmp/log10dens_del.pkl"
            print("Saving log10_dens in "+tmp_name)
            with open(tmp_name,"wb") as f:
                pickle.dump(log10_dens,f)

    # define the z_source:
    # dens now is already in Msun/kpc^2
    """
    dens_Ms_arcsec2 = dens/(dP**2)  # Msun /''^2 
    dens_Ms_kpc2    = dens_Ms_arcsec2*(arcXkpc**2) # Msun/kpc^2
    """ 
    dens_Ms_kpc2    = density*u.Msun/(u.kpc*u.kpc)
    if DEBUG:
        print("DEBUG ")
        print("M(density)",np.sum(mass_grid))
        print("M(m)",np.sum(m))
        print("M(gal)",Gal.M_tot)
        print("M(gal2)",Gal.M)
        print("dx,dy",dx,dy)
        print("bin area",dx*dy,"kpc^2")
        print("<density>",np.mean(dens_Ms_kpc2))
        print("max(density)",np.max(dens_Ms_kpc2))
    z_source = get_z_source(cosmo=cosmo,z_lens=z_lens,dens_Ms_kpc2=dens_Ms_kpc2,
                            z_source_max=z_source_max,verbose=verbose)
    if z_source==0:
        raise AttributeError("Rerun trying different projection")
        
    # dP to convert from kpc/pix to ''/pix
    dP = get_dP(radius*u.kpc,pixel_num,cosmo=cosmo,Gal=Gal) # ''/pix -> to double check that this is correct
    # store the results
    #res = [dens_Ms_kpc2,radius,dP,[dx,dy],z_source,cosmo,proj_index]
    kw_res = {"dens_Ms_kpc2":dens_Ms_kpc2,
              "radius":radius,
              "dP":dP,
              "dx":dx,
              "dy":dy,
              "z_source":z_source,
              "z_lens":z_lens,
              "pixel_num":pixel_num,
              "cosmo":cosmo,
              "proj_index":proj_index}
    if save_res:
        with open(Gal.dens_res,"wb") as f:
            pickle.dump(kw_res,f)
        print("Saved "+Gal.dens_res)
    return kw_res




if __name__=="__main__":
    parser = ArgumentParser(description="Project particles into a mass sheet - histogram version")
    parser.add_argument("-dn","--dir_name",dest="dir_name",type=str, help="Directory name",default=dir_name)
    parser.add_argument("-pxn","--pixel_num",dest="pixel_num",type=int, help="Pixel number",default=pixel_num.imag)
    parser.add_argument("-r","--radius",dest="radius",type=int, help="Cutout radius [kpc]",default=def_radius.value)
    parser.add_argument("-zsm","--z_source_max",dest="z_source_max",type=float, help="Maximum source redshift",default=z_source_max)
    #parser.add_argument("-nrr", "--not_rerun", dest="rerun", 
    #                    default=True,action="store_false",help="if True, rerun code")
    parser.add_argument("-pl", "--plot", dest="plot", 
                        default=False,action="store_true",help="Plot dens map")
    parser.add_argument("-v", "--verbose", dest="verbose", 
                        default=False,action="store_true",help="verbose")
    args          = parser.parse_args()
    pixel_num     = args.pixel_num*1j
    radius        = args.radius*u.kpc
    #rerun         = args.rerun
    dir_name      = args.dir_name
    verbose       = args.verbose
    z_source_max  = args.z_source_max
    plot          = args.plot
    """
    if rerun:
        #print("DEBUG -- USING test sym")
        #Gal = get_rnd_gal(sim=test_sim,check_prev=False,reuse_previous=False,min_mass="1e13",max_z="1")
        Gal = get_rnd_PG()#sim=std_sim,check_prev=False,reuse_previous=False,min_mass="1e13",max_z="1")
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
    """    
    Gal    = get_rnd_PG()
    z_lens = Gal.z
    """
    print("DEBUG")
    NG = Gal
    fig, ax = plt.subplots(3)
    nx = 100
    for name,part in zip(["stars","dm","gas"],[NG.stars,NG.dm,NG.gas]):
        coords = part["coords"]
        x,y,z  = coords.T
        print(np.std(coords,axis=0))
        ax[0].hist(x,bins=nx,alpha=.5,label=name)#,range=[xmin, xmax])
        ax[1].hist(y,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
        ax[2].hist(z,bins=nx,alpha=.5,label=name)#,range=[ymin, ymax])
    ax[0].set_xlabel("X [kpc]")
    ax[2].set_xlabel("Z [kpc]")
    ax[1].set_xlabel("Y [kpc]")
    ax[2].legend()
    namefig = f"tmp/hist_by_hand_parts.png"
    plt.tight_layout()
    plt.savefig(namefig)
    plt.close()
    print("Saved "+namefig) 
    print("DEBUG")
    """
    print("pixel_num:",pixel_num)
    print("cutout radius:",radius)
    print("Gal:",str(Gal))
    Gal = prep_Gal_projpath(Gal)
    if verbose:
        print("Assumptions: We are considering the maximum source redshift to be ",z_source_max)
        if int(pixel_num.imag)<500:
            print("Warning: running test")
        elif int(pixel_num.imag)>=1000:
            print("Warning: running very long")
    kw_res = get_dens_map_rotate_hist(Gal=Gal,pixel_num=pixel_num,
                                      z_source_max=z_source_max,
                                      verbose=True,radius=radius)#plot=plot,verbose=verbose)
    dens_Ms_kpc2 = kw_res["dens_Ms_kpc2"]
    radius       = kw_res["radius"]
    dP           = kw_res["dP"]
    dx           = kw_res["dx"]
    dy           = kw_res["dy"]
    z_source     = kw_res["z_source"]
    cosmo        = kw_res["cosmo"]
    proj_index   = kw_res["proj_index"]
    """
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
    """
    Xg, Yg    = np.mgrid[-radius:radius:pixel_num, -radius:radius:pixel_num]*u.kpc
    arcXkpc   = cosmo.arcsec_per_kpc_proper(z_lens) # ''/kpc
    
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
    #if DEBUG:
    #    print("DEBUG: shape Xg,num_aRa",np.shape(Xg),np.shape(num_aRa))

    ra  = Xg*arcXkpc #arcsec
    dec = Yg*arcXkpc #arcsec
    """
    print("DEBUG")
    plt.imshow(ra.value)
    plt.colorbar()
    plt.title("Ra source")
    #im_name = f"{Gal.proj_dir}/ra_src.pdf"
    im_name = f"tmp/ra_src.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(dec.value)
    plt.colorbar()
    plt.title("Dec source")
    #im_name = f"{Gal.proj_dir}/dec_src.pdf"
    im_name = f"tmp/dec_src.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(np.log10(sersic_brightness(ra,dec)) )
    plt.colorbar()
    plt.title("log Source")
    #im_name = f"{Gal.proj_dir}/src.pdf"
    im_name = f"tmp/src.pdf"
    plt.savefig(im_name)
    plt.close()
    
    plt.imshow(num_aRa.value)
    plt.colorbar()
    plt.title("Ra deflection")
    #im_name = f"{Gal.proj_dir}/alpha_ra.pdf"
    im_name = f"tmp/alpha_ra.pdf"
    plt.savefig(im_name)
    plt.close()
    plt.imshow(num_aDec.value)
    plt.colorbar()
    plt.title("Dec deflection")
    #im_name = f"{Gal.proj_dir}/alpha_dec.pdf"
    im_name = f"tmp/alpha_dec.pdf"
    plt.savefig(im_name)
    plt.close()
    print("DEBUG")
    
    ra_im  = ra.value-num_aRa.value
    dec_im = dec.value-num_aDec.value
    lensed_im = sersic_brightness(ra_im,dec_im)
    plt.imshow(np.log10(lensed_im))
    plt.colorbar()
    plt.title("Log Lensed Sersic image")
    #im_name = f"tmp/lensed_im.pdf"
    im_name = f"{Gal.proj_dir}/lensed_im.pdf"
    plt.savefig(im_name)
    plt.close()
    print("Saving "+im_name)
    """
    """
    # define the source to be behind the most dense pixel (not necessarily==CMS)
    # -> could consider not exactly behind but at least within a small radius of it    
    # find the "center" of the galaxy -> most dens pixel
    #index_maxk = np.where(kappa_grid==np.max(kappa_grid))
    # this could find a single high pixel -> take the average of n pixels to "smooth it out"
    # and find the real centre (although with lower res -> we don't care for a few pixels of difference)
    #   first: find the smallest rescale factor that is a multiple of the size of the image
    rescale_factor = [i  for i in np.range(5,15) if len(kappa_grid)%i==0 and len(kappa_grid[0])%i==0]
    assert len(rescale_factor)!=0
    rescale_factor = rescale_factor[0]
    rescaled_kappa = scipy.ndimage.zoom(kappa_grid, 1./rescale_factor, order=3)
    index_max_rescaled_kappa = np.where(rescaled_kappa==np.max(rescaled_kappa))
    index_maxk = [(index_max_rescaled_kappa[0]+.5)*rescale_factor,(index_max_rescaled_kappa[1]+.5)*rescale_factor]

    print("dx,dy",dx,dy)
    print("shape kappa",np.shape(kappa_grid))
    cntx_meas  = -(index_maxk[0] -int((len(kappa_grid))/2.))*dx
    cnty_meas  = -(index_maxk[1] -int((len(kappa_grid[0]))/2.))*dy
    cntx_meas  = cntx_meas[0]
    cnty_meas  = cnty_meas[0]
    
    cntx_meas_arcsec = cntx_meas*arcXkpc.to("arcsec/kpc").value
    cnty_meas_arcsec = cnty_meas*arcXkpc.to("arcsec/kpc").value
    print("Center measured")
    print(cntx_meas,cnty_meas,"[kpc]")
    print(cntx_meas_arcsec,cnty_meas_arcsec,"['']")
    #print("WARNING - INVERTING X AND Y FOR THE SOURCE")

    #source = Sersic(I=10,cnty=cnty_meas_arcsec,cntx=cntx_meas_arcsec,pa=45,q=.65,n=4)
    """ # all this was highly biased
    # find CM again: -> kappa grid is density off by a constant, cancelled out by the normalisation
    ra_cm  = np.sum(ra* kappa_grid)/np.sum(kappa_grid)
    dec_cm = np.sum(dec* kappa_grid)/np.sum(kappa_grid)
    #source = Sersic(I=10,cntx=ra_cm,cnty=dec_cm,pa=45,q=.65,n=4)
    source  = SersicElliptic()
    e1,e2   = e1e2_from_qphi(q=.65,phi=45)
    
    #index_maxk = np.abs(ra_cm-ra).argmin(axis=0)[0],np.abs(dec_cm-dec).argmin(axis=1)[0]
    index_cmsk = np.abs(ra_cm-ra[:,0]).argmin(),np.abs(dec_cm-dec[0]).argmin()
    # consider instead the maxima as center for the source
    ira_max,idec_max = np.where(kappa_grid==np.max(kappa_grid))
    ira_max = ira_max[0]
    idec_max = idec_max[0]
    ra_max  = ra[ira_max][idec_max]
    dec_max = dec[ira_max][idec_max]

    fg,axes    = plt.subplots(2,3,figsize=(16,8))
    rad_arcsec = radius*arcXkpc
    rad_arcsec = rad_arcsec.value
    extent_arcs = [-rad_arcsec,rad_arcsec,-rad_arcsec,rad_arcsec] #arcsec
    extent_kpc  = [-radius.value,radius.value,-radius.value,radius.value] #kpc
    
    ax  = axes[0][0]
    im0 = ax.imshow(np.log10(kappa_grid.value),origin="lower",extent=extent_kpc)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical')
    ax.contour(kappa_grid.value,cmap=plt.cm.inferno,alpha=.8,extent=extent_kpc)
    #ax.scatter(index_cmsk[1],index_cmsk[0],marker="x",c="r",label="CMS")
    #ax.scatter(idec_max,ira_max,marker="x",c="g",label=r"Max($\kappa$)")
    ax.scatter(dec_cm/arcXkpc,ra_cm/arcXkpc,marker="x",c="b",label="CMS")
    ax.scatter(dec_max/arcXkpc,ra_max/arcXkpc,marker="x",c="g",label=r"Max($\kappa$)")
    ax.set_xlim(extent_kpc[0],extent_kpc[1])
    ax.set_ylim(extent_kpc[2],extent_kpc[3])
    ax.set_title(r"log $\kappa$")
    ax.legend()

    ax  = axes[0][1]
    im0 = ax.imshow(np.log10(kappa_grid.value),origin="lower",extent=extent_arcs,)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical')
    ax.contour(kappa_grid.value,cmap=plt.cm.inferno,extent=extent_arcs)
    #ax.scatter(index_cmsk[0],index_cmsk[1],marker="x",c="r",label="CMS")
    ax.scatter(dec_cm,ra_cm,marker="x",c="b",label="CMS arcsec")
    ax.scatter(dec_max,ra_max,marker="x",c="g",label=r"Max($\kappa$) arcsec")
    ax.set_xlim(extent_arcs[0],extent_arcs[1])
    ax.set_ylim(extent_arcs[2],extent_arcs[3])
    ax.set_title(r"log $\kappa$ (arcsec)")
    ax.legend()
    
    #ax.axis("off")
    #Src = np.log10(source.image(ra,dec)) 
    se_image = source.function(x=ra.value,y=dec.value,center_x=ra_max.value,
                               center_y=dec_max.value, e1=e1,e2=e2,amp=10,n_sersic=4,R_sersic=5)
    Src = np.log10(se_image)
    ax = axes[0][2]
    im0 = ax.imshow(Src,extent=extent_arcs,origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical')
    ax.contour(Src,cmap=plt.cm.gist_earth_r,extent=extent_arcs)
    ax.set_title("Log Source")
    #ax.scatter(index_maxk[1],index_maxk[0],marker="x",c="r",label="x,y meas:"+str(index_maxk[0])+","+str(index_maxk[1]))

    ax  = axes[1][0]
    im0 = ax.imshow(num_aRa.value,extent=extent_arcs,origin="lower")
    ax.set_title("Ra deflection")
    ax  = axes[1][1]
    ax.imshow(num_aDec.value,extent=extent_arcs,origin="lower")
    ax.set_title("Dec deflection")
    ra_im  = ra.value-num_aRa.value
    dec_im = dec.value-num_aDec.value
    #lensed_im =  source.image(ra_im,dec_im)
    lensed_im = source.function(x=ra_im,y=dec_im,center_x=ra_max.value,
                                center_y=dec_max.value,e1=e1,e2=e2,amp=10,n_sersic=4,R_sersic=5)
    Lsnd      = np.log10(lensed_im)
    ax = axes[1][2]
    ax.imshow(Lsnd,extent=extent_arcs,origin="lower")
    ax.contour(Lsnd,cmap=plt.cm.gist_earth_r,extent=extent_arcs)
    ax.set_title("Log Lensed Sersic image")

    for i,axi in enumerate(axes):
        for j,axij in enumerate(axi):
            if i==0 and j==0:
                axij.set_xlabel('X [kpc]')
                axij.set_ylabel('Y [kpc]')
            else:
                axij.set_xlabel('RA ["]')
                axij.set_ylabel('DEC ["]')
            
    im_name = f"tmp/lensed_im.pdf"
    # just to be sure
    try:
        os.remove(im_name)
    except:
        pass
    
    plt.tight_layout()
    plt.suptitle("Gal z="+str(Gal.z))
    plt.savefig(im_name)
    plt.close()
    print("Saving "+im_name)

    # for convenience, I link the result to the tmp dir
    #os.unlink("./tmp/"+dir_name)
    #os.symlink(Gal.proj_dir[:-1],"./tmp/.")
    
    print("Success")
    
