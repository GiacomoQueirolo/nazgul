# Idea is using the photutils isophotal fitting tools to obtain an ellitpical fit of the isocontours
# from : 
# https://github.com/astropy/photutils-datasets/blob/main/notebooks/isophote/isophote_example3.ipynb
# seems very straightforward

"""
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
g = EllipseGeometry(530., 511, 10., 0.1, 10./180.*np.pi)
g.find_center(data)

ellipse = Ellipse(data, geometry=g)
isolist = ellipse.fit_image()
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model

from Gen_PM_PLL import LoadLens #LensPart,kwlens_part_AS,cutoff_radius,z_source_max,pixel_num
from python_tools.get_res import LoadClass
from ParticleGalaxy import Gal2kwMXYZ 
from project_gal  import proj_parts,findDens,get_densmap,xyminmax

    
from scipy.optimize import curve_fit

# temp : 
from Gen_PM_PLL import LensPart,PMLens,thetaE_AS_prefact,thetaE_AS,get_lens_model_AS

def linlaw(x, a, b) :
    return a + x * b


def get_kwisodens(Lens,cutoff_rad=None,pixel_num=None,verbose=True):
    kw_parts = Gal2kwMXYZ(Lens.Gal)
    # cutoff_radius_dens is used to have a first (quite alright)
    # estimate of the densest point
    if cutoff_rad is None:
        cutoff_rad = Lens.cutoff_radius
    if pixel_num is None:
        pixel_num  = Lens.pixel_num # here in case I need to change it
    density = get_densmap(kw_parts,Lens.proj_index,pixel_num=pixel_num,
                          cutoff_radius=cutoff_rad,
                          cutoff_radius_dens=100*u.kpc,verbose=verbose)
    kappa = (density/Lens.SigCrit).value
    
    
    # x0, y0, sma(semimajor), eps(ellipticity=1-b/a), pa
    geom = EllipseGeometry(kappa.shape[0], kappa.shape[1], 10., 0.5, 0./180.*np.pi)
    geom.find_center(kappa)
    ellipse = Ellipse(kappa, geometry=geom)
    isolist = ellipse.fit_image()

    model_kappa = build_ellipse_model(kappa.shape, isolist)
    return {"isolist":isolist,"geom":geom,"kappa":kappa,"model_kappa":model_kappa}




def plot_isodens(Lens,savedir=None,cutoff_rad=None,pixel_num=None,verbose=True):
    if savedir is None:
        savedir = Lens.savedir
    if cutoff_rad is None:
        cutoff_rad = Lens.cutoff_radius 
    else:
        try:
            cutoff_rad.value
        except:
            raise RuntimeError("Given cutoff_rad in kpc explicitely")
    if pixel_num is None:
        pixel_num  = Lens.pixel_num # here in case I need to change it

    kw_isodens = get_kwisodens(Lens,cutoff_rad=cutoff_rad,pixel_num=pixel_num,verbose=verbose)
    isolist,geom,kappa,model_kappa = kw_isodens["isolist"],kw_isodens["geom"],kw_isodens["kappa"],kw_isodens["model_kappa"]
    xmin,xmax,ymin,ymax = list(np.array(xyminmax(cutoff_rad)).ravel())
    extent              = [xmin,xmax,ymin,ymax] 

    residual = kappa - model_kappa
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    im_i = ax1.imshow(np.log10(kappa),cmap=plt.cm.inferno,origin="lower",extent=extent)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=r"$\kappa_{sim}$")

    #ax1.set_xlim(limits)
    #ax1.set_ylim(limits)
    ax1.set_title(r"log$_{10}$($\kappa$)")
    
    im_i = ax2.imshow(np.log10(model_kappa),cmap=plt.cm.inferno,origin="lower",extent=extent)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=r"$\kappa_{iso}$")
    #ax2.set_xlim(limits)
    #ax2.set_ylim(limits)
    ax2.set_title(r"log$_{10}$($\kappa_{Model}$)")

    vm = np.median(residual) +2*np.std(residual)
    print("testing vm residual:",vm)
    im_i = ax3.imshow(residual,cmap="bwr",extent=extent,origin="lower",vmin=-vm,vmax=vm)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im_i, cax=cax, orientation='vertical',label=r"$\kappa_{sim}$-$\kappa_{iso}$")

    ax3.set_title("Residual")
    for ax in ax1,ax2,ax3:
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel("kpc")
        ax.set_ylabel("kpc")
    
    
    # overplot a few isophotes on the residual map
    iso1 = isolist.get_closest(10.)
    iso2 = isolist.get_closest(40.)
    iso3 = isolist.get_closest(100.)
    
    x, y, = iso1.sampled_coordinates()
    Nx,Ny = kappa.shape
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    x, y, = iso2.sampled_coordinates()
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    x, y, = iso3.sampled_coordinates()
    x_plot = xmin + (x / Nx) * (xmax - xmin)
    y_plot = ymin + (y / Ny) * (ymax - ymin)
    ax3.plot(x_plot, y_plot, color='black')
    name_plot = savedir+"/isodens_model.pdf"
    print(f"Saving {name_plot}")
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    kpcPix  = cutoff_rad/pixel_num
    sma_kpc = isolist.sma*kpcPix # semi-major axis in kcp

    
    plt.figure(figsize=(10, 5))
    plt.figure(1)
    
    plt.subplot(221)
    plt.errorbar(sma_kpc, 1-isolist.eps, yerr=isolist.ellip_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('Axis Ratio')
    
    plt.subplot(222)
    plt.errorbar(sma_kpc, isolist.pa/np.pi*180., yerr=isolist.pa_err/np.pi* 80., fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('PA (deg)')
    plt.subplot(223)
    plt.errorbar(sma_kpc, (isolist.x0-geom.x0)*kpcPix.value, yerr=isolist.x0_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('X0-Xcnt [kpc]')
    
    plt.subplot(224)
    plt.errorbar(sma_kpc, (isolist.y0-geom.y0)*kpcPix.value, yerr=isolist.y0_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('Y0-Ycnt [kpc]')
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    name_plot = savedir+"/isodens_prms1.pdf"
    print(f"Saving {name_plot}")
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.figure(1)
    limits = [0., 100., -0.1, 0.1]
    
    plt.subplot(221)
    plt.axis(limits)
    plt.errorbar(sma_kpc, isolist.a3, yerr=isolist.a3_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('A3')
    
    plt.subplot(222)
    plt.axis(limits)
    plt.errorbar(sma_kpc, isolist.b3, yerr=isolist.b3_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('B3')
    
    plt.subplot(223)
    plt.axis(limits)
    plt.errorbar(sma_kpc, isolist.a4, yerr=isolist.a4_err, fmt='o', markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('A4')
    
    plt.subplot(224)
    plt.axis(limits)
    plt.errorbar(sma_kpc, isolist.b4, fmt='o', yerr=isolist.b4_err, markersize=4)
    plt.xlabel('Semimajor axis [kpc]')
    plt.ylabel('B4')
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    
    
    name_plot = savedir+"/isodens_prms2.pdf"
    print(f"Saving {name_plot}")
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()


    plt.title(r"Plot of $\kappa$")
    
    plt.scatter(sma_kpc.value,isolist.intens,c="k")
    plt.legend()
    plt.xlabel(r'Semimajor axis [kpc])')
    plt.ylabel(r'$\kappa$')
    name_plot = savedir+"/isodens_kappa.pdf"
    print(f"Saving {name_plot}")
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    # fit as linear in log
    popt_log,pcov_log = curve_fit(linlaw,np.log10(sma_kpc.value[1:]),np.log10(isolist.intens[1:]))
    ydatafit = linlaw(np.log10(sma_kpc.value[1:]), *popt_log)
    kw_loglogfit = {"popt_log":popt_log,"pcov_log":pcov_log,"fity":ydatafit,"fitx":np.log10(sma_kpc.value[1:])}
    plt.title(r"LogLog plot of $\kappa$")
    str_fit = "log10(kappa) ="+str(np.round(popt_log[0],2))+"log10(sma)^"+str(np.round(popt_log[1],2))
    plt.plot(np.log10(sma_kpc.value[1:]),ydatafit, c="b",ls="--",label="Fit:"+str_fit)
    
    plt.scatter(np.log10(sma_kpc.value),np.log10(isolist.intens),c="k")
    plt.legend()
    plt.xlabel(r'log$_{10}$(Semimajor axis [kpc])')
    plt.ylabel(r'log$_{10}$($\kappa$)')
    name_plot = savedir+"/isodens_loglogk.pdf"
    print(f"Saving {name_plot}")
    plt.tight_layout()
    plt.savefig(name_plot)
    plt.close()

    kw_res = {"isodens":kw_isodens,"loglogfit":kw_loglogfit}
    return kw_res


if __name__=="__main__":
    # for now applied to a "known" lens galaxy
    #Lens = LoadLens("sim_lens/RefL0025N0752/snap23_G3.0//lensing/G3SGn0_Npix200_PartAS.pkl")
    Lens = LoadLens("sim_lens/RefL0025N0752/snap18_G5.0//lensing/G5SGn0_Npix200_PartAS.pkl")
       
    #name_proj = Lens.savedir+"/proj_mass.pdf"
    savedir = "tmp/"
    kw_res = plot_isodens(Lens,savedir,cutoff_rad=50*u.kpc)
    
    kw_loglogfit = kw_res["loglogfit"]
    fity    = kw_loglogfit["fity"]
    logr    = kw_loglogfit["fitx"]
    gamma_fit_r = fity/logr
    kw_isodens = kw_res["isodens"]
    isolist = kw_isodens["isolist"]
    y = np.log10(isolist.intens[1:])
    gamma_r = y/logr
    print("this doesn't make sense because gamma is constant as given by the fit")
    plt.plot(logr,gamma_fit_r,c="k",label=r"fit $\gamma$") #ill behave at log(r)=0 by construction
    plt.plot(logr,gamma_r,c="r",label=r" $\gamma$") 
    gamma_fit_fix = kw_loglogfit["popt_log"][-1]
    plt.plot(logr,gamma_fit_fix*np.ones_like(logr),c="b",ls="--",label=r"$\gamma_{opt.}$="+str(np.round(gamma_fit_fix,2))) 
    plt.title(r"$\gamma(r)=\frac{\gamma log(r)}{log(r)}$")
    plt.xlabel(r"log$_{10}$r")
    plt.ylabel(r"$\gamma$(r)")
    plt.legend()
    namefig = "tmp/gamma_r.pdf"
    print("Saving "+namefig)
    plt.savefig(namefig)