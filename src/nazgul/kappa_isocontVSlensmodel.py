# compare output from model_sim with isodensity contour results from isodens
import glob
import numpy as np
import matplotlib.pyplot as plt
import lenstronomy.Util.util as util
from lenstronomy.LensModel.lens_model import LensModel
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nazgul.fnct import std_sim
from nazgul.generate_particle_lens import LoadLens,get_extents
from nazgul.isodens import get_kwisodens

from python_tools.get_res import load_whatever
from python_tools.tools import to_dimless
#from model_sim_lens import lens_model_list
# temp. modelling path
_res_dir = "./tmp/modelling_sim_lenses/"

def get_kappa_model(lens):
    # get the kappa map from the lens model
    # NOT the one obtained by the simulation
    _radec = lens.imageModel.ImageNumerics.coordinates_evaluate   
    _ra,_dec = _radec

    try:
        kwargs_res = load_whatever(f"{_res_dir}/{lens.name}kw_res.json")
    except:
        kwargs_res = load_whatever(f"{_res_dir}/{lens.name}/kw_res.json")

    kwargs_lens = kwargs_res["kwargs_lens"]
    try:
        kw_input = load_whatever(f"{_res_dir}/{lens.name}kw_input.dll")
    except:
        kw_input = load_whatever(f"{_res_dir}/{lens.name}/kw_input.dll")
    
    lens_model   = LensModel(lens_model_list=kw_input["kwargs_model"]["lens_model_list"])
    kappa_model  = lens_model.kappa(_ra,_dec, kwargs_lens)
    kappa_model  = util.array2image(kappa_model)
    return kappa_model

def get_kappa_iso_and_sim(lens):
    kw_res = get_kwisodens(lens)
    return kw_res["model_kappa"],kw_res["kappa"],kw_res["cutoff_rad"]
    
if __name__=="__main__":
    Lens = LoadLens("sim_lens/RefL0025N0752/snap24_G21.0/test_sim_lens_AMR/G21SGn0_Npix200_PartAS.pkl")
    kappa_model                    = get_kappa_model(Lens)
    kappa_iso,kappa_sim,cutoff_rad = get_kappa_iso_and_sim(Lens)
    
    xmin = -cutoff_rad
    ymin = -cutoff_rad
    xmax = +cutoff_rad
    ymax = +cutoff_rad
    extent_cutoff = [xmin,xmax,ymin,ymax] 

    if cutoff_rad<=to_dimless(Lens.radius):
        raise RuntimeError("not accounting for this case")
    # the sim is over this
    kw_extents  = get_extents(Lens.arcXkpc,Lens)
    extent_full = kw_extents["extent_kpc"]

    # I am not sure if they have the same cutout -  to test
    #print("DEBUG")
    #print(extent_cutoff)
    #print(extent_full)
    
    fig,axes  = plt.subplots(2,3,figsize=(15,12))
    ims       = []
    ax        = axes[0][0]
    kw_imshow = {"origin":"lower",
                 "cmap":plt.cm.inferno,
                 "extent":extent_full}

    fig.suptitle(r"$\kappa$ map")
    ax      = axes[0][0]
    ax.set_title("Simulation")
    ims.append(ax.imshow(np.log10(kappa_sim),**kw_imshow))
    ax      = axes[0][1]
    ax.set_title("Iso. fit")
    ims.append(ax.imshow(np.log10(kappa_iso),**kw_imshow))
    ax      = axes[0][2]
    ax.set_title("Lens model")
    ims.append(ax.imshow(np.log10(kappa_model),**kw_imshow))

    ax      = axes[1][0]
    kw_imshow["cmap"] = "seismic" #or bwr
    ax.set_title("Iso. fit - Lens Model")
    ims.append(ax.imshow(np.log10(kappa_iso-kappa_model),**kw_imshow))
    ax      = axes[1][1]
    ax.set_title("Iso. fit - Sim.")
    ims.append(ax.imshow(np.log10(kappa_iso-kappa_sim),**kw_imshow))
    ax      = axes[1][2]
    ax.set_title("Lens model - Sim.")
    ims.append(ax.imshow(np.log10(kappa_model-kappa_sim),**kw_imshow))

    i=0
    for axi in axes:
        for axii in axi:
            axii.set_xlabel("RA")
            axii.set_xlabel("DEC")
            divider = make_axes_locatable(axii)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(ims[i], cax=cax, orientation='vertical',label=r"log$_{10}(\kappa)$")
            i+=1
    nm_fig = f"tmp/kappa_isoVsLens_{Lens.name}.png"
    print(f"Saving {nm_fig}")
    plt.tight_layout()
    plt.savefig(nm_fig)
    
