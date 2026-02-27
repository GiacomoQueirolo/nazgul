# Take Gal from ParticleGalaxy and does projection and similar calculations

import dill
import numpy as np
from copy import copy,deepcopy
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.constants as const
from scipy.ndimage import gaussian_filter1d

from python_tools.tools import to_dimless,short_SciNot
from python_tools.get_res import load_whatever
from ParticleGalaxy import get_CM,Gal2kwMXYZ

def proj_parts(kw_parts,proj_index,arcXkpc=None):    
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

def findDens(M,X,Y,rad,nbins=200,XYCM=None):
    # locate the coordinates of the densest bin
    if XYCM is None:
        XYCM = get_CM(M,X,Y)
    X_cm,Y_cm = XYCM
    bins = [np.linspace(X_cm - rad,X_cm+rad,nbins),
            np.linspace(Y_cm - rad,Y_cm+rad,nbins)]
    
    mass_grid, xedges, yedges   = np.histogram2d(X,Y,
                                       bins=bins,
                                       weights=M,
                                       density=False)
    # max density indexes
    ix, iy = np.unravel_index(np.argmax(mass_grid), mass_grid.shape)
    
    # Compute center coordinates
    X_dns  = 0.5 * (xedges[ix] + xedges[ix+1])
    Y_dns  = 0.5 * (yedges[iy] + yedges[iy+1])

    return X_dns,Y_dns

def get_minzsource_proj(Gal,kw_parts,cutoff_radius,pixel_num,z_source_max,verbose=True,save_res=True,reload=True):
    # try all projection in order to obtain a lens
    proj_index = 0
    kw_res     = None
    # if present and reload:
    try:
        assert reload
        kw_res = load_whatever(Gal.proj_zs_path)
        return kw_res
    except AssertionError:
        pass
    except Exception as e :
        if verbose:
            print("Failed to load because "+str(e))
            print("Recomputing min_z_source ...")
        pass
    # else compute it
    while proj_index<3:
        try:
            kw_res = get_min_z_source(Gal=Gal,proj_index=proj_index,
                                      kw_parts=deepcopy(kw_parts),cutoff_radius=cutoff_radius,
                                      pixel_num=pixel_num,z_source_max=z_source_max,verbose=verbose)
            break
        except AttributeError as Ae:
            print("Error : ")
            print(Ae)
            # should only be if the minimum z_source is higher than the maximum z_source
            # try with other proj
            proj_index+=1
    if kw_res is None:
        print("M(gal)",short_SciNot(Gal.M_tot))
        print("z_gal",Gal.z)
        raise RuntimeError("There is no projection of the galaxy that create a lens given the z_source_max")
    else:
            
        if save_res:
            with open(Gal.proj_zs_path,"wb") as f:
                dill.dump(kw_res,f)
            print("Saved "+Gal.proj_zs_path)
        return kw_res

def xyminmax(cutoff_radius):
    cutoff_radius = to_dimless(cutoff_radius)
    # assuming x,y centred around 0
    xmin = -cutoff_radius
    ymin = -cutoff_radius
    xmax = +cutoff_radius
    ymax = +cutoff_radius
    rng  = [[xmin, xmax], [ymin, ymax]] 
    return rng
    
def get_densmap_dimless(x,y,m,pixel_num,cutoff_radius,verbose=True,ret_mgrid=False,plot=False):
    # Get density map (dimensionless)
    # Assumed dimless values
    nx,ny = int(pixel_num.imag),int(pixel_num.imag)
    if nx==0:
        nx,ny = int(pixel_num), int(pixel_num)
    # get range from cutoff_radius
    rng = xyminmax(cutoff_radius)
    
    # numpy.histogram2d returns H with shape (nx_bins, ny_bins) where H[i,j]
    # counts x-bin i and y-bin j. We transpose to (ny, nx) so rows are y.
    H, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny],
                                       range=rng,
                                       weights=m,density=False)  
    # if density=True, it normalises it to the total density
    # H is then the distribution of mass for each bin, not the density
    mass_grid = H.T.copy() # Solar Masses
    # H shape: (nx, ny) -> transpose to (ny, nx)
    if plot:
        plt.close()
        plt.imshow(np.log10(mass_grid))
        plt.title("Gal Name:"+Gal.Name)
        plt.colorbar()
        plt.savefig("tmp/mass_"+str(proj_index)+".png")
        print("DEBUG:"+"tmp/mass_"+str(proj_index)+".png")
        plt.close()
    # area of the (dx/dy) bins:
    dx = np.diff(xedges) #kpc #==(xmax - xmin) / nx 
    dy = np.diff(yedges) #kpc

    # density_ij = M_ij/(Area_bin_ij)
    density = mass_grid / (dx * dy)
    if verbose:
        print("Note: following dimension are assumed")
        print("<dx,dy>",np.round(np.mean(dx),2),np.round(np.mean(dy),2),"kpc")
        print("<Bin Area>",np.round(np.mean(dx*dy),2), "kpc^2") 
        print("<mass>",short_SciNot(np.round(np.mean(mass_grid),2)),"Msun")
        print("<density>",short_SciNot(np.round(np.mean(density),2)),"Msun/kpc^2")
    if ret_mgrid:
        return density,mass_grid
    return density
    
def get_densmap(kw_parts,proj_index,pixel_num,cutoff_radius,cutoff_radius_dens=None,verbose=True):
    # Get density map from kw_parts and projected index (dimensional)
    # note: in principle kw_parts could be not given and re-obtained from Gal with
    #  kw_parts = Gal2kwMXYZ(Gal) from remade_Gal
    Ms  = kw_parts["Ms"]
    # project given the proj_index
    Xs,Ys = proj_parts(kw_parts,proj_index)
    # recenter around densest point 
    if cutoff_radius_dens is None:
        # cutoff_radius_dens is used to have a first (quite alright)
        # estimate of the densest point -> by def == cutoff_radius
        # but not necessarily so
        cutoff_radius_dens = cutoff_radius
    Xdns,Ydns = findDens(Ms,Xs,Ys,cutoff_radius_dens)
    Xs -= Xdns
    Ys -= Ydns
    # Get density map (dimensional)
    x  = np.asarray(Xs.to("kpc").value) #kpc
    y  = np.asarray(Ys.to("kpc").value) #kpc
    m  = np.asarray(Ms.to("solMass").value)  # M_sol
    
    cutoff_radius = to_dimless(cutoff_radius) #kpc
    density = get_densmap_dimless(x,y,m,pixel_num,cutoff_radius,\
                                  verbose=verbose,ret_mgrid=False,plot=False)
    # Dens now is in Msun/kpc^2 
    dens_Ms_kpc2 = density*u.Msun/(u.kpc*u.kpc)
    return dens_Ms_kpc2

def get_min_z_source(Gal,proj_index,kw_parts,pixel_num,z_source_max,cutoff_radius,verbose=True):
    # given a projection, return the minimal z_source
    # fails if it can't produce a supercritical lens w. z_source<z_source_max

    # Get density map (dimensional)
    dens_Ms_kpc2 = get_densmap(kw_parts,proj_index,pixel_num,cutoff_radius,verbose=verbose)

    # define the z_source:        
    z_source_min = _get_min_z_source(cosmo=Gal.cosmo,z_lens=Gal.z,dens_Ms_kpc2=dens_Ms_kpc2,
                            z_source_max=z_source_max,verbose=verbose)
    if z_source_min==0:
        raise AttributeError("Rerun trying different projection")

    kw_res = {"z_source_min":z_source_min,
              "proj_index":proj_index}
    return kw_res


def _get_min_z_source(cosmo,z_lens,dens_Ms_kpc2,z_source_max,verbose=True):
    # the lens has to be supercritical
    # dens>Sigma_crit = (c^2/4PiG D_d(z_lens) ) D_s(z_source)/D_ds(z_lens,z_source)
    # -> D_s(z_source)/D_ds(z_lens,z_source) < 4PiG D_d(z_lens) *dens/c^2
    # D_s(z_source)/D_ds(z_lens,z_source) is not easy to compute analytically, but we can sample it
    if z_lens>z_source_max:
        raise ValueError("The galaxy redshift is higher than the maximum allowed source redshift")
        #return 0
    try:
        dens_Ms_kpc2.value
    except:
        # dens_Ms_kpc2 is already given in Msun/kpc^2
        dens_Ms_kpc2 *= u.Msun/(u.kpc**2)
    assert dens_Ms_kpc2.unit==u.Msun/(u.kpc**2)
    
    max_DsDds = np.max(dens_Ms_kpc2)*4*np.pi*const.G*cosmo.angular_diameter_distance(z_lens)/(const.c**2) 
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
            print("max density",short_SciNot(np.max(dens_Ms_kpc2)))
            print("Saved "+name)
        return 0
    else:
        # Note: successful test means only that there is AT LEAST 1 PIXEL that is supercritical
        minimise     = np.abs(DsDds-max_DsDds) 
        z_source_min = z_source_range[np.argmin(minimise)]
        return z_source_min


def get_rough_radius(cosmo,z_source,z_lens,kw_part_arc,scale=2,verbose=True):
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

    
    Ms,RAs,DECs  = kw_part_arc["Ms"],kw_part_arc["RAs"],kw_part_arc["DECs"]
    RA_cm,DEC_cm = get_CM(Ms,RAs,DECs)
    # note: RA/DEC are given in arcsec, and Ms in Msun
    RA_centered,DEC_centered = RAs-RA_cm,DECs-DEC_cm
    kw_part_RADEC_cnt = {"Ms":Ms,"RA":RA_centered,"DEC":DEC_centered}
    kw_Ddds = {"Dd":Dd,"Dds":Dds,"Ds":Ds}
    return scale*theta_E_from_particles(verbose=verbose,**kw_part_RADEC_cnt,**kw_Ddds) # arcsec

def theta_E_from_particles(Ms, RA, DEC, Dd, Ds, Dds, nbins=100,verbose=True,sigma_smooth=2.):
    # Physical scale of 1 arcsec at Dd
    arcXkpc = u.rad.to("arcsec")*u.arcsec/Dd.to("kpc") # arcsec/kpc (on the lens plane)
    # Critical density
    Sigma_crit = (const.c**2 / (4*np.pi*const.G) * (Ds/(Dd*Dds))).to("Msun/kpc^2")
    # Radii in kpc
    thetas = np.sqrt(RA**2 + DEC**2)
    r_kpc  = thetas / arcXkpc
    # Histogram Σ(R)
    t_max = np.max(thetas)/10
    i = np.arange(nbins + 1)
    t_edges = t_max * np.sqrt(i / nbins)
    r_edges = t_edges/arcXkpc 
    hist, edges = np.histogram(r_kpc, bins=r_edges, weights=Ms)
    rmid = 0.5*(edges[1:] + edges[:-1]) #kpc

    # Smooth Σ(R)
    Sigma_R = hist / (2*np.pi*rmid*np.diff(edges))  # convert to Σ(R)
    Sigma_R_s = gaussian_filter1d(Sigma_R, sigma_smooth)*Sigma_R.unit
    # Enclosed Σ(<R)
    Menc = np.cumsum(Sigma_R_s * 2*np.pi*rmid*np.diff(edges))
    Sigma_encl = Menc / (np.pi*rmid**2)
    # Solve Σ=Σcrit by interpolation
    theta_E_kpc = np.interp(Sigma_crit, Sigma_encl[::-1], rmid[::-1])
    theta_E_arcsec = theta_E_kpc *arcXkpc

    print("--DEBUG")
    print("theta_E_arcsec found",theta_E_arcsec)
    plt.scatter(rmid,Sigma_encl)
    plt.axhline(Sigma_crit.value,ls="--",c="r",label=r"$\Sigma_{crit}$")
    plt.axvline(to_dimless(theta_E_kpc),label=r"$\theta_E$="+str(short_SciNot(theta_E_kpc)))
    plt.xlabel(r"kpc")
    plt.ylabel(r"$\Sigma$ ["+str(Sigma_encl.unit)+"]")
    plt.title(r"$\Sigma_{encl}$")
    plt.legend()
    nm_tmp = "tmp/Sigma.png"
    plt.savefig(nm_tmp)
    plt.close()
    return theta_E_arcsec

def Gal2MRADEC(Gal,proj_index,arcXkpc):
    kw_parts = Gal2kwMXYZ(Gal)
    Xs,Ys    = proj_parts(kw_parts,proj_index)
    RAs,DECs = Xs.to("kpc")*arcXkpc,Ys.to("kpc")*arcXkpc
    return kw_parts["Ms"],RAs,DECs
