"""Plotting functions for the Particle Lenses
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from python_tools.fwhm import get_fwhm


def _plot_caustics(LPClass,
                   kwargs_lens,
                   kw_extents=None,
                   fast_caustic = True,
                   savename="test_caustics.png"):
    if kw_extents is None:
        kw_extents = LPClass.kw_extents 
    xmin,xmax,ymin,ymax = kw_extents["extent_arcsec"]
    kw_crit             = LPClass.critical_curve_caustics
    cl_rad_x,cl_rad_y   = kw_crit["critical_lines"]["radial"]
    cc_rad_x,cc_rad_y   = kw_crit["caustics"]["radial"]
    cl_tan_x,cl_tan_y   = kw_crit["critical_lines"]["tangential"]
    cc_tan_x,cc_tan_y   = kw_crit["caustics"]["tangential"]

    cent_caust_tan = np.mean(cc_tan_x),np.mean(cc_tan_y)
    fig,ax = plt.subplots()
    ax.scatter(cc_rad_x,cc_rad_y,c="b",marker=".",label="Radial Caustics")
    ax.scatter(cc_tan_x,cc_tan_y,c="r",marker=".",label="Tangential Caustics")
    ax.scatter(cl_rad_x,cl_rad_y,c="cyan",marker=".",label="Radial Crit. Curve")
    ax.scatter(cl_tan_x,cl_tan_y,c="darkorange",marker=".",label="Tangential Crit. Curve")
    # not used anymore
    #ax.scatter(*cent_caust_tan,c="k",marker="x",label="Tang. Center Caustic")
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    plt.gca().set_aspect('equal')
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.legend()
    ax.set_title("Caustics and Critical Curves") 
    plt.tight_layout()
    print(f"Saving {savename}") 
    plt.savefig(savename)
    plt.close()
    
def plot_caustics(Model,fast_caustic = True,savename="test_caustics.png",kw_extents=None):
    return _plot_caustics(Model,Model.kwargs_lens,
                          fast_caustic=fast_caustic,savename=savename,kw_extents=kw_extents)


def plot_kappamap(kappa1,extent_kpc,title1="",savename="kappa.png",skip_show=False):
    fig,axes = plt.subplots(2,figsize=(8,16))

    ax  = axes[0]
    im0 = ax.imshow(kappa1,origin="lower",extent=extent_kpc)
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(title1) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')


    # take advantage of the circular simmetry and obtain the projection
    k1_proj = kappa1[int(len(kappa1)/2)]
    x = np.linspace(extent_kpc[0],extent_kpc[1],len(k1_proj))
    _xcnt = np.median(x)
    ax = axes[1]
    ax.plot(x,k1_proj,c="k")
    fwhm_k1 = get_fwhm(k1_proj,x) 
    hmax  = max(k1_proj)/2.
    ax.axvline(_xcnt,c="g",alpha=.5)

    ax.plot([_xcnt-fwhm_k1/2,_xcnt+fwhm_k1/2],[hmax,hmax],ls="-.",c="r",label="FWHM="+str(np.round(fwhm_k1,3)))
    ax.legend()
    ax.set_title(title1 +" projection at x=0")
    plt.suptitle("Density distribution")
    print(f"Saving {savename}")
    plt.savefig(savename)
    if not skip_show:
        plt.show()
    plt.close()


def plot_lensed_im_and_kappa(Model,savename="lensed_im.pdf",kw_extents=None):
    kappa = Model.kappa_map
    if kw_extents is None:
        kw_extents = Model.kw_extents
    fg,axes = plt.subplots(1,2,figsize=(10,5))
    ax = axes[0]

    extent_kpc    = kw_extents["extent_kpc"]
    extent_arcsec = kw_extents["extent_arcsec"]
    
    im0   = ax.matshow(kappa,origin='lower',extent=extent_kpc,cmap="hot")
    ax.set_xlabel("X [kpc]")
    ax.set_ylabel("Y [kpc]")
    ax.set_title(r"Convergence "+Model.Gal.Name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label=r"$\kappa$")

    lnsd_im  = Model.image_sim 
    ax = axes[1]
    im0   = ax.matshow(np.log10(lnsd_im),origin='lower',extent=extent_arcsec)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    
    ax.set_title("Lensed image "+Model.Gal.Name)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fg.colorbar(im0, cax=cax, orientation='vertical',label=r"log$_{10}$ flux [arbitrary]")
    plt.suptitle(r"With z$_{\text{lens}}$="+str(np.round(Model.z_lens,2))+" z$_{\text{source}}$="+str(np.round(Model.z_source,2)))
    plt.tight_layout()
    print(f"Saving {savename}") 
    plt.savefig(savename)
    plt.close("all")
    
def plot_all(Model,savename_lensed="lensed_im.pdf",savename_kappa="kappa.png",savename_caustics="caustics.png",fast_caustic=True,skip_caustic=False):
    kw_extents = Model.kw_extents
    plot_lensed_im_and_kappa(Model,savename=Model.savedir/savename_lensed,kw_extents=kw_extents)
    if not skip_caustic:
        plot_caustics(Model,fast_caustic=fast_caustic,savename=Model.savedir/savename_caustics,kw_extents=kw_extents)
    plt.close("all")
    return 0
