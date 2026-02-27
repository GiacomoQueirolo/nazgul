# Take Gal from ParticleGalaxy and does projection and similar calculations
# functions usefuls for lensing and later imported by Gen_PM_PLL.py

# update 1.1 from project_gal.py form 8/12/25:
# we compute now twice the density map:
# once to recenter around the densest pixel
# the second time to find the densest pixel again ->  needed if we have missed the densest point in the fist cut
# do it in a smarter way: 
# take everything -> bin it (large) -> find densest bin -> cut radius by a factor -> re-bin w. higher precision 
# -> iterate to a threshold mean particle density -> stop : output: max density value and coordinate (recenter EVERYTHING around that) 
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

from python_tools.tools import mkdir,to_dimless,short_SciNot
from ParticleGalaxy import Gal2kwMXYZ,get_CM
# for now keep this and check if still needed
dir_name     = "proj_part_hist"
def prep_Gal_projpath(Gal,dir_name=dir_name):
    # impractical but easy to set up
    Gal.proj_dir = Gal.gal_snap_dir+f"/{dir_name}_{Gal.Name}/"
    mkdir(Gal.proj_dir)
    Gal.projection_path = f"{Gal.proj_dir}/projection.pkl"
    return Gal

from python_tools.get_res import load_whatever
from copy import copy,deepcopy

def proj_parts(kw_parts,proj_index):    
    Xs,Ys,Zs = kw_parts["Xs"],kw_parts["Ys"],kw_parts["Zs"]
    if proj_index==0:
        _   = True  # all as usual
    elif proj_index==1:
        Ys  = copy(Zs)
    elif proj_index==2:
        Xs  = copy(Ys)
        Ys  = copy(Zs)    
    else:
        raise RuntimeError("Projection index can only be 1,2 or 3, not "+str(proj_index))
    return Xs,Ys 

def project_kw_parts(kw_parts,proj_index):    
    Xs,Ys = proj_parts(kw_parts,proj_index) 
    kw_parts_proj = {"Xs":Xs,"Ys":Ys,"Ms":kw_parts["Ms"]} 
    return kw_parts_proj
    
def kwparts2arcsec(kw_parts,arcXkpc):
    if "Zs" in kw_parts:
        # have no sense to have 3D distr in arcsec
        raise RuntimeError("The kw_parts should already be projected")
    RAs = kw_parts["Xs"]*arcXkpc
    DECs = kw_parts["Ys"]*arcXkpc
    return {"RAs":RAs,"DECs":DECs,"Ms":kw_parts["Ms"]}

def projection_main(Gal,kw_parts,pixel_num,z_source_max,arcXkpc=None,verbose=True,save_res=True,reload=True):
    # this is going to be the main function:
    # - for each projection:
    #       - find center and densest bin iteratively:
    #           - output:
    #               -  MD (mode/maximum density) coord
    #               -  MD value
    #               -  ~best 2D density histogram~ -> too large and easy to recover
    #                       - instead: nbins/pixel_num is fixed, give best cutout  
    #       - test if said densest bin is enough to be a SGL  
    #       - return:
    #              - projection
    #              - z_min
    #              - MD coord
    #              - theta_E (approx) centered around MD -> this computed separately from these results
    # try all projection in order to obtain a lens
    proj_index = 0
    kw_res     = None
    # if present and reload: -> not sure if this is to be done
    try:
        assert reload
        kw_res = load_whatever(Gal.projection_path)
        return kw_res
    except AssertionError:
        pass
    except Exception as e :
        if verbose:
            print("Failed to load because "+str(e))
            print("Recomputing projection ...")
        pass
    # else compute it
    if arcXkpc is None:
        arcXkpc = Gal.cosmo.arcsec_per_kpc_proper(Gal.z)
    while proj_index<3:
        try:
            # iterate density histogram
            kw_parts_proj = project_kw_parts(kw_parts=kw_parts,proj_index=proj_index)
            #kw_parts_proj_arcsec = kwparts2arcsec(kw_parts_proj,arcXkpc)
            kw_2Ddens = iterate_dens_map(Gal=Gal,
                                      kw_parts_proj=kw_parts_proj,
                                      pixel_num=pixel_num,
                                      arcXkpc =arcXkpc,
                                      verbose=verbose)    
            kw_z_min = get_min_z_source(Gal=Gal,
                                      kw_2Ddens=kw_2Ddens,
                                      z_source_max=z_source_max,verbose=verbose)
            kw_proj = {"proj_index":proj_index}
            kw_res  = kw_proj|kw_2Ddens|kw_z_min
            break
        except AttributeError as Ae:
            print("Projection Error : "+str(Ae))
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if kw_res is None:
        print("M(gal)",short_SciNot(Gal.M_tot))
        print("z_gal",Gal.z)
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
        if save_res:
            with open(Gal.projection_path,"wb") as f:
                pickle.dump(kw_res,f)
            print("Saved "+Gal.projection_path)
        return kw_res

def iterate_dens_map(Gal,
                      kw_parts_proj,
                      pixel_num,
                      meta_params=None, # to define how to iterate (converg. prms etc)
                      arcXkpc=None,
                      verbose=True):
    # returns: kw_2Ddens["MD_value"][u.Msun/(u.kpc**2),1] 
    #          kw_2Ddens["MD_coord"][arcsec,2]
    #          kw_2Ddens["cutoff_dens"][arcsec,1]
    MD_coord_kpc = None
    radius_kpc   = None #initially take all the particules
    it = 0
    print("DEBUG","kw_parts_proj",kw_parts_proj)
    while True:
        """
        # this is done in get_densmap
        kw_parts_rec = recenter_MD(kw_parts_proj,MD)
        kw_parts_cut = crop_parts(kw_parts_rec,radius_kpc)
        dns_map,coords =  densmap(kw_parts_cut,pixel_num)
        """
        numDens,dens_map,coords = get_densmap(kw_parts_proj,MD_coord_kpc=MD_coord_kpc,cutoff_dens_kpc=radius_kpc,pixel_num=pixel_num)
        MD_value,MD_coord_kpc   = get_MD(dens_map,coords)
        print("DEBUG","MD_coord_kpc",MD_coord_kpc)
        radius_kpc              = update_density_radius(dens_map,coords,meta_params)
        print("DEBUG","radius_kpc",radius_kpc)
        min_numDens             = np.min(numDens)
        if densmap_convergence(it,min_numDens,meta_params):
            break        
        it+=1
    print("DEBUG","kw_parts_proj",kw_parts_proj)

    if arcXkpc is None:
        arcXkpc = Gal.cosmo.arcsec_per_kpc_proper(Gal.z)
    print("DEBUG","arcXkpc",arcXkpc,"MD_coord",MD_coord_kpc*arcXkpc)
    kw_2Ddens = {"MD_value":MD_value,"MD_coord":MD_coord_kpc*arcXkpc,"cutoff_dens":radius_kpc*arcXkpc,"pixel_num":pixel_num}
    return kw_2Ddens
        
def get_densmap(kw_parts_proj,pixel_num,MD_coord_kpc=None,cutoff_dens_kpc=None,verbose=True):
    """
    output: numdens,dns_map,coords
    """ 
    kw_parts_rec      = recenter_kwparts(kw_parts_proj,MD_coord_kpc)
    kw_parts_cut      = crop_kwparts(kw_parts_rec,cutoff_dens_kpc)
    numdens,dns_map,coords = densmap(kw_parts_cut,pixel_num,verbose=verbose)
    return numdens,dns_map,coords

def get_MD(dens_map,coords):
    i_max_x,i_max_y = np.where(dens_map==np.max(dens_map))
    if len(i_max_x)==1:
        MD_value = dens_map[i_max_x,i_max_y]
        xMD,yMD = coords[0][i_max_x],coords[1][i_max_y]
        assert xMD.unit == yMD.unit
        MD_coord_kpc = np.array([xMD.value,yMD.value])*xMD.unit
    else:
        raise RuntimeError("Multiple density maxima found - this should not be the case") 
    return MD_value,MD_coord_kpc  

def recenter_kwparts(kw_parts,cent=None):
    if cent is None:
        return kw_parts
    try:
        X0,Y0 = cent
        kw_parts["Xs"] -= X0
        kw_parts["Ys"] -= Y0
    except KeyError:
        RA0,DEC0 = cent
        kw_parts["RAs"] -= RA0
        kw_parts["DECs"] -= DEC0        
    return kw_parts
def crop_kwparts(kw_parts,radius=None):
    if radius is None:
        return kw_parts
    try:
        iparts = np.where(np.hypot(kw_parts["Xs"],kw_parts["Ys"])<radius)
        kw_parts["Ms"] = kw_parts["Ms"][iparts]
        kw_parts["Xs"] = kw_parts["Xs"][iparts]  
        kw_parts["Ys"] = kw_parts["Ys"][iparts]
    except KeyError:
        iparts = np.where(np.hypot(kw_parts["RAs"],kw_parts["DECs"])<radius)
        kw_parts["Ms"] = kw_parts["Ms"][iparts]
        kw_parts["RAs"] = kw_parts["RAs"][iparts]  
        kw_parts["DECs"] = kw_parts["DECs"][iparts]
    return kw_parts

def densmap(kw_parts,pixel_num,verbose=True):    
    """
    output: dns_map,coords
    """
    Ms = kw_parts["Ms"]
    Xs = kw_parts["Xs"]
    Ys = kw_parts["Ys"]
    # Get density map (dimensional)
    x  = np.asarray(Xs.to("kpc").value) #kpc
    y  = np.asarray(Ys.to("kpc").value) #kpc
    m  = np.asarray(Ms.to("solMass").value)  # M_sol
    # for convergence criterion:
    numDens,_,_ = np.histogram2d(x, y, bins=pixel_num)
    
    H, xedges, yedges = np.histogram2d(x, y, bins=pixel_num,
                                       weights=m,density=False)  
    # if density=True, it normalises it to the total density
    # H is then the distribution of mass for each bin, not the density
    # H shape: (nx, ny) -> transpose to (ny, nx) 
    mass_grid = H.T.copy()*u.Msun # Solar Masses
    xedges   *= u.kpc
    yedges   *= u.kpc
    xmid      = 0.5*(xedges[1:] + xedges[:-1]) #kpc
    ymid      = 0.5*(yedges[1:] + yedges[:-1]) #kpc
    coords    = xmid,ymid # kpc,kpc
    
    # area of the (dx/dy) bins:
    dx = np.diff(xedges) #kpc #==(xmax - xmin) / nx 
    dy = np.diff(yedges) #kpc
    # density_ij = M_ij/(Area_bin_ij)
    density_map = mass_grid / (dx * dy) # Msun/kpc^2 
    return numDens,density_map,coords
    
########
# most important functions to defined the density map:
def densmap_convergence(it,min_numDens,meta_params=None):
    # For now very simple: if minimum numDens<10 and iteration>5 or iteration>100:
    if meta_params is None:
        meta_params = {"min_it":5,"max_it":100,"min_numDens":10}
    if it>meta_params["min_it"]:
        if min_numDens<meta_params["min_numDens"]:
            return True
    if it>meta_params["max_it"]:
        print("Reached maximum iteration")
        return True
    return False

def update_density_radius(dens_map,coords,meta_params=None):
    # for now very simple - increase resolution by 1.3
    x,y = coords
    radx = (np.max(x)-np.min(x))/2
    rady = (np.max(y)-np.min(y))/2
    rad  = 0.5*(radx+rady)
    if meta_params is None:
        meta_params = {"rad_fact":1.333}
    return rad*meta_params["rad_fact"]
#######

def get_min_z_source(Gal,kw_2Ddens,z_source_max,verbose=True):
    # given a projection, return the minimal z_source
    # fails if it can't produce a supercritical lens w. z_source<z_source_max
    
    max_dens = kw_2Ddens["MD_value"]
    # define the z_source_min:        
    z_source_min = _get_min_z_source(cosmo=Gal.cosmo,z_lens=Gal.z,
                                    max_dens=max_dens,
                                    z_source_max=z_source_max,verbose=verbose)
    if z_source_min==0:
        raise AttributeError("Rerun trying different projection")

    kw_zs_min = {"z_source_min":z_source_min}
    return kw_zs_min


def _get_min_z_source(cosmo,z_lens,max_dens,z_source_max,verbose=True):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")
        #return 0
    try:
        max_dens.value
    except:
        # max_dens is already given in Msun/kpc^2
        max_dens *= u.Msun/(u.kpc**2)
    assert max_dens.unit==u.Msun/(u.kpc**2)
    
    max_DsDds = max_dens*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2) 
    max_DsDds = max_DsDds.to("").value # assert(max_DsDds.unit==u.dimensionless_unscaled) -> equivalent

    min_DsDds = cosmo.angular_diameter_distance(z_source_max)/cosmo.angular_diameter_distance_z1z2(z_lens,z_source_max) # this is the minimum
    min_DsDds = min_DsDds.to("").value # dimensionless
    
    z_source_range = np.linspace(z_lens+0.1,z_source_max,100) # it's a very smooth funct->
    DsDds = np.array([cosmo.angular_diameter_distance(z_s).to("Mpc").value/cosmo.angular_diameter_distance_z1z2(z_lens,z_s).to("Mpc").value for z_s in z_source_range])
    if not min_DsDds<max_DsDds:
        # to do: deal with this kind of output
        if verbose:
            print("Warning: the minimum z_source needed to have a lens is higher than the maximum allowed z_source")
            plt.plot(z_source_range,DsDds,ls="-",c="k",label=r"D$_{\text{s}}$/D$_{\text{ds}}$(z$_{source}$)")
            plt.xlabel(r"z$_{\text{source}}$")
            plt.axhline(max_DsDds,ls="--",c="r",label=r"max(dens)*4$\pi$*G*$D_l$/c$^2$="+str( short_SciNot(max_DsDds)))
            plt.legend()
            name = "tmp/DsDds.pdf"
            plt.savefig(name)
            print("max density",short_SciNot(max_dens.value))
            print("Saved "+name)
        return 0
    else:
        # Note: successful test means only that there is AT LEAST 1 PIXEL that is supercritical
        minimise     = np.abs(DsDds-max_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        return z_source_min




def get_rough_radius(cosmo,z_lens,z_source,kw_part_arc,kw_2Ddens,scale=2,verbose=True):
    # -> this should only be used for plotting
    # the idea is simple:
    # we want a very approximate idea of the theta_E of the galaxy
    # to do that, we fit a SIS to its particle distribution 
    # basically in 1D, assuming (wrong but we don't care) spherical symmetry
    # then we scale that by the scale (default=2) and that is our aperture

    Dd      = cosmo.angular_diameter_distance(z_lens).to("Mpc")
    Ds      = cosmo.angular_diameter_distance(z_source).to("Mpc")
    Dds     = cosmo.angular_diameter_distance_z1z2(z_lens,z_source).to("Mpc") 
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc")

    # Warning: these must not be already centered around MD 
    # (or if they are, MD should be updated to 0,0)
    Ms,RAs,DECs  = kw_part_arc["Ms"],kw_part_arc["RAs"],kw_part_arc["DECs"]
    RA_MD,DEC_MD = kw_2Ddens["MD_coord"]
    # note: RA/DEC are given in arcsec, and Ms in Msun
    print("RAs",RAs)
    print("RA_MD",RA_MD)
    
    RA_centered,DEC_centered = RAs-RA_MD,DECs-DEC_MD # centered around MD
    kw_part_RADEC_cnt = {"Ms":Ms,"RA":RA_centered,"DEC":DEC_centered,"cutoff":kw_2Ddens["cutoff_dens"]}
    kw_Ddds = {"Dd":Dd,"Dds":Dds,"Ds":Ds}
    return scale*theta_E_from_particles(verbose=verbose,**kw_part_RADEC_cnt,**kw_Ddds) # arcsec

from scipy.ndimage import gaussian_filter1d
def theta_E_from_particles(Ms, RA, DEC, Dd, Ds, Dds,cutoff,nbins=100,verbose=True,sigma_smooth=2.,nm_sigmaplot="tmp/Sigma.png"):
    # Physical scale of 1 arcsec at Dd
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc") # arcsec/kpc (on the lens plane)
    # Critical density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # Radii in kpc
    thetas = np.sqrt(RA**2 + DEC**2)
    # consider only within cutoff -> what about AMR in 1D instead
    i_cut = np.where(thetas<cutoff) # must have same dimension
    
    thetas_cut = thetas[i_cut]
    Ms_cut     = Ms[i_cut]
    r_kpc  = thetas_cut / arcXkpc
    # Histogram Σ(R)
    i = np.arange(nbins + 1)
    # still bins growing as sqrt
    t_edges = np.max(thetas_cut) * np.sqrt(i / nbins)
    r_edges = t_edges/arcXkpc 
    hist, edges = np.histogram(r_kpc, bins=r_edges, weights=Ms_cut)
    rmid = 0.5*(edges[1:] + edges[:-1]) #kpc

    # convert to Σ(R)
    Sigma_R = hist / (2*np.pi*rmid*np.diff(edges)) 
    # Smooth Σ(R)
    Sigma_R_s = gaussian_filter1d(Sigma_R, sigma_smooth)*Sigma_R.unit
    # Enclosed Σ(<R)
    Menc = np.cumsum(Sigma_R_s * 2*np.pi*rmid*np.diff(edges))
    Sigma_encl = Menc / (np.pi*rmid**2)
    # Solve Σ=Σcrit by interpolation
    theta_E_kpc = np.interp(Sigma_crit, Sigma_encl[::-1], rmid[::-1])
    theta_E_arcsec = theta_E_kpc *arcXkpc

    print("theta_E_arcsec found",short_SciNot(np.round(theta_E_arcsec,2)))
    plt.scatter(rmid,Sigma_encl)
    plt.axhline(Sigma_crit.value,ls="--",c="r",label=r"$\Sigma_{crit}$")
    plt.axvline(to_dimless(theta_E_kpc),label=r"$\theta_E$="+str(short_SciNot(theta_E_kpc)))
    plt.xlabel(r"kpc")
    plt.ylabel(r"$\Sigma$ ["+str(Sigma_encl.unit)+"]")
    plt.title(r"$\Sigma_{encl}$")
    plt.legend()
    plt.savefig(nm_sigmaplot)
    print(f"Saving {nm}"_sigmaplot)
    plt.close()
    return theta_E_arcsec

def Gal2MRADEC(Gal,proj_index,arcXkpc):
    kw_parts = Gal2kwMXYZ(Gal)
    Xs,Ys    = proj_parts(kw_parts,proj_index)
    RAs,DECs = Xs.to("kpc")*arcXkpc,Ys.to("kpc")*arcXkpc
    return kw_parts["Ms"],RAs,DECs

def Gal2kw_samples(Gal,proj_index,kw_2Ddens,arcXkpc,nbins=200,scale_rad=5):
    # we scale the radius by a factor to be sure to include the center
    Ms,RAs,DECs = Gal2MRADEC(Gal,proj_index,arcXkpc=arcXkpc)
    # RA,DEC= arcsec, Ms = Msun
    #print("Some galaxy have a 'shifted' CM")
    RA_cm,DEC_cm = get_CM(Ms,RAs,DECs)
    
    
    print(f"We recenter around the densest point (MD) obtained iteratively (see iterate_dens_map)") 
    RA_MD,DEC_MD = kw_2Ddens["MD_coord"]
    print("Info:  CM vs Densest ")
    print("CM:",np.round(RA_cm,2),np.round(DEC_cm,2))
    print("Dns:",np.round(RA_MD,2),np.round(DEC_MD,2))
    print("Dist:",np.round(np.sqrt((RA_MD-RA_dns)**2+(DEC_cm-DEC_MD)**2),2))

    kw_samples   = {}
    kw_samples["RAs"]  = RAs-RA_MD   #arcsec
    kw_samples["DECs"] = DECs-DEC_MD  #arcsec
    
    kw_samples["Ms"]   = Ms    #Msun
    kw_samples["cm"]   = RA_cm-RA_MD,DEC_cm-DEC_MD  # 
    return kw_samples
