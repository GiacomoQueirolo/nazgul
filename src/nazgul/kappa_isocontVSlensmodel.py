# compare output from model_sim with isodensity contour results from isodens
import glob,sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import lenstronomy.Util.util as util
from lenstronomy.LensModel.lens_model import LensModel
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nazgul.pathfinder import std_sim
from nazgul.mount_doom.cracks_of_doom import LoadLens
from nazgul.isodens import get_kwisodens

from python_tools.get_res import load_whatever
from python_tools.tools import to_dimless
#from model_sim_lens import lens_model_list
# temp. modelling path

from nazgul.modelling_severals import setup_lens
from nazgul.mount_doom.lens_system import LensSystem

def get_kappa_model(lens):
    # get the kappa map from the lens model
    # NOT the one obtained by the simulation
    _ra,_dec = lens.gallens._radec

    kwargs_res = load_whatever(f"{lens.model_res_dir}/kw_res.json")

    kwargs_lens = kwargs_res["kwargs_lens"]
    kw_input = load_whatever(f"{lens.model_res_dir}/kw_input.dll")
    
    lens_model   = LensModel(lens_model_list=kw_input["kwargs_model"]["lens_model_list"])
    kappa_model  = lens_model.kappa(_ra,_dec, kwargs_lens)
    kappa_model  = util.array2image(kappa_model)
    return kappa_model

def get_kappa_iso_and_sim(lens):
    kw_res = get_kwisodens(lens)
    return kw_res["model"],kw_res["kappa"],kw_res["cutoff_rad"]
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot isocontours of kappa map from simulation vs modelling")
    parser.add_argument('-lp','--lens_path',type=str,
                        dest="lens_path",
                        help="Path to pre-computed LensPart class instance")
    args         = parser.parse_args()
    lens_path    = args.lens_path

    gal_lens = LoadLens(lens_path)
    Lens = LensSystem.from_GalLens(gal_lens)
    Lens = setup_lens(Lens)
    kappa_model                    = get_kappa_model(Lens)
    kappa_iso,kappa_sim,cutoff_rad = get_kappa_iso_and_sim(gal_lens)
    
    xmin = -cutoff_rad
    ymin = -cutoff_rad
    xmax = +cutoff_rad
    ymax = +cutoff_rad
    extent_cutoff = [xmin,xmax,ymin,ymax] 

    if cutoff_rad<=to_dimless(gal_lens.radius):
        raise RuntimeError("not accounting for this case")
    # the sim is over this
    kw_extents  = gal_lens.kw_extents
    extent_full = kw_extents["extent_kpc"]

    # I am not sure if they have the same cutout -  to test
    #print("DEBUG")
    #print(extent_cutoff)
    #print(extent_full)
    
    fig,axes  = plt.subplots(2,3,figsize=(17,12))
    ims       = []
    ax        = axes[0][0]
    kw_imshow = {"origin":"lower",
                 "cmap":plt.cm.inferno,
                 "extent":extent_full}

    fig.suptitle(r"$\kappa$ map")
    ax      = axes[0][0]
    ax.set_title("Simulation")
    resc_kappa_sim = kappa_sim # + np.min(kappa_sim[kappa_sim>0])
    ims.append(ax.imshow((resc_kappa_sim),**kw_imshow))
    ax      = axes[0][1]
    ax.set_title("Iso. fit")
    resc_kappa_iso = kappa_iso # + np.min(kappa_iso[kappa_iso>0])
    ims.append(ax.imshow((resc_kappa_iso),**kw_imshow))
    ax      = axes[0][2]
    ax.set_title("Lens model")
    ims.append(ax.imshow((kappa_model),**kw_imshow))

    ax      = axes[1][0]
    kw_imshow["cmap"] = "seismic" #or bwr
    ax.set_title("Iso. fit - Lens Model")
    ims.append(ax.imshow(kappa_iso-kappa_model,**kw_imshow))
    ax      = axes[1][1]
    ax.set_title("Iso. fit - Sim.")
    ims.append(ax.imshow(kappa_iso-kappa_sim,**kw_imshow))
    ax      = axes[1][2]
    ax.set_title("Lens model - Sim.")
    ims.append(ax.imshow(kappa_model-kappa_sim,**kw_imshow))

    i=0
    for axi in axes:
        for axii in axi:
            axii.set_xlabel("RA")
            axii.set_ylabel("DEC")
            divider = make_axes_locatable(axii)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if i<3:
                #fig.colorbar(ims[i], cax=cax, orientation='vertical',label=r"log$_{10}(\kappa)$")
                fig.colorbar(ims[i], cax=cax, orientation='vertical',label=r"$\kappa$")
            else:
                fig.colorbar(ims[i], cax=cax, orientation='vertical',label=r"$\Delta \kappa$")
            i+=1
    nm_fig = f"{Lens.model_res_dir}/kappa_isoVsLens.png"
    print(f"Saving {nm_fig}")
    plt.tight_layout()
    plt.savefig(nm_fig)
    
