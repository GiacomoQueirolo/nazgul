# Model all lenses where LOS is not simulated but w. LOS in the model
# to study the internal shear (á la Etherington)
import os,gc
import argparse
import warnings
import numpy as np
import sys,json,dill
from corner import corner
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.Plots import chain_plot

from nazgul.plot_PL import plot_kappamap
from nazgul.lens_part_LOS import get_kw_los
from nazgul.mount_doom.lens_system import LensSystem

from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.Modelling.lib_models import setup_lens,setup_sim_obs,get_kwargs_likelihood,get_lenses2model,get_lens_mask
from nazgul.Modelling.lib_models import is_someone_workin_on_it,set_workin_on_it,workin_on_it
from nazgul.Modelling.lib_models import save_kw,plot_model_plot
from nazgul.Modelling.lib_models import model_res_base,n_it_std,n_part_std,n_burn_std,n_run_std # default values

from nazgul.plot_image_pair import _limits,plot_image_pairs_pdf

lens_model_list   = ['EPL','LOS_MINIMAL']
source_model_list = ["SERSIC"]
res_dir_base      = model_res_base/"simNoShear/"

def get_kwargs_params(lens):
    # Params:
    # initial guess of non-linear parameters, we chose different starting parameters than the truth #
    tE = lens.gallens.thetaE.value

    # LOS added separately    
    kwargs_lens_init = [{'theta_E': tE + np.random.normal(0,.1,1)[0]*tE, 
                    'e1': 0, 'e2': 0, 
                    'gamma': 2., 
                    'center_x': 0., 'center_y': 0}]
    kwargs_source_init = [{'R_sersic': 0.03, 'n_sersic': 1., 'center_x': 0, 'center_y': 0}]
    
    # initial spread in parameter estimation #
    kwargs_lens_sigma = [{'theta_E': 0.3, 
                          'e1': 0.2, 'e2': 0.2, 'gamma': .2, 
                          'center_x': 0.1, 'center_y': 0.1}]
    kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': .5, 'center_x': .1, 'center_y': 0.1}]
    
    # hard bound lower limit in parameter space #
    kwargs_lower_lens = [{'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5, 'center_x': -10., 'center_y': -10}]
    kwargs_lower_source = [{'R_sersic': 0.001, 'n_sersic': .5, 'center_x': -10, 'center_y': -10}]
    # hard bound upper limit in parameter space #
    kwargs_upper_lens = [{'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5, 'center_x': 10., 'center_y': 10}]
    kwargs_upper_source = [{'R_sersic': 10, 'n_sersic': 5., 'center_x': 10, 'center_y': 10}]

    # add LOS params
    # First we fix to 0 kappa and omega (not gamma_od for now)
    # omega_LOS should not be fixed! the LOS shears in combination induce a small rotation
    # allowing for freedom in omega_LOS accounts for this and prevents bias in the shears
    
    gamma_prior = 0.5
    omega_prior = 0.5
    gamma_sigma = 0.1
    omega_sigma = 0.1

    # this is for the minimal model
    kwargs_fixed_los = {'kappa_od': 0.0, 'kappa_los': 0.0, 'omega_od': 0.0}

    kwargs_lens_init.append({'gamma1_od':0, 'gamma2_od': 0,
                             'gamma1_los':0, 'gamma2_los':0,
                             'omega_los': 0 })

    kwargs_lens_sigma.append({'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,
                              'gamma1_los': gamma_sigma, 'gamma2_los': gamma_sigma,
                              'omega_los': omega_sigma})

    kwargs_lower_lens.append({'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,
                              'gamma1_los': -gamma_prior, 'gamma2_los': -gamma_prior,
                              'omega_los': -omega_prior})

    kwargs_upper_lens.append({'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,
                              'gamma1_los': gamma_prior, 'gamma2_los': gamma_prior,
                              'omega_los': omega_prior})

    """
    # this is for the full LOS
    kwargs_fixed_los = {'kappa_od': 0, 'kappa_os': 0,'kappa_ds':0,
                        'omega_od': 0.0, 'omega_os': 0.0, 'omega_ds': 0.0}
    kwargs_lens_init.append({'gamma1_od': 0, 'gamma2_od': 0,
                             'gamma1_os': 0, 'gamma2_os': 0,
                             'gamma1_ds': 0, 'gamma2_ds': 0})
    kwargs_lens_sigma.append({'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,
                              'gamma1_os': gamma_sigma, 'gamma2_os': gamma_sigma,
                              'gamma1_ds': gamma_sigma, 'gamma2_ds': gamma_sigma})
    kwargs_lower_lens.append({'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,
                              'gamma1_os': -gamma_prior, 'gamma2_os': -gamma_prior,
                              'gamma1_ds': -gamma_prior, 'gamma2_ds': -gamma_prior})

    kwargs_upper_lens.append({'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,
                              'gamma1_os': gamma_prior, 'gamma2_os': gamma_prior,
                              'gamma1_ds': gamma_prior, 'gamma2_ds': gamma_prior})
    """
    
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, kwargs_fixed_los], kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}
    return kwargs_params


if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Simulate (without shear) and model the lens (with shear) - Study the Internal Shear")
    parser.add_argument('-nl','--n_lenses',type=int,dest="n_lenses",default=5,help=f"Number of lenses to model")
    parser.add_argument('-mtE','--min_thetaE',type=float,dest="min_thetaE",default=None,help=f"Min theta_E for the gal to be considered a lens")
    parser.add_argument('-snap','--snap',nargs="+",type=int,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")

    args       = parser.parse_args()
    n_lenses   = args.n_lenses
    min_thetaE = args.min_thetaE 
    snaps      = args.snaps #[25,26,27]
    sim        = args.sim
    subsim     = args.subsim
    simsuite   = args.simsuite

    lenses2skip = []
    
    kw_get_all_gallens = {"sim":sim,
                          "subsim":subsim,
                           "simsuite":simsuite,
                            "snaps":snaps}
    res_dir = res_dir_base
    gal_lenses  = get_lenses2model(res_dir=res_dir,
                                   reload=True,
                                   kw_get_all_gallens=kw_get_all_gallens,
                                   n_lenses=np.nan,
                                   min_thetaE=min_thetaE,
                                   skip_lenses=lenses2skip)

    sim_images = []
    obs_images = []
    nms_lenses = []
    extents    = []
    limits     = []
    err_fail   = []
    N_gallenses = len(gal_lenses)
    for i,gal_lens in enumerate(gal_lenses): 
        print("\n     Loading lens "+gal_lens.name+\
              "\n####################################################\n")
        lens = LensSystem.from_GalLens(gal_lens)
        try:
            lens = setup_lens(lens,res_dir=res_dir)
        except Exception as e:
            warnings.warn(f"DEBUG\nlens {lens} has failed due to:\n{e}\n")
            err_fail.append({"lens":{lens.name},"error":e})
            continue
        """plot_kappamap(lens.gallens.kappa_map, 
                      extent_kpc=lens.gallens.kw_extents["extent_kpc"],
                      savename=f"{lens.model_res_dir}/kappa_gal.png")"""
        
        multi_band_list = setup_sim_obs(lens)
        image_orig = lens.get_lensed_image(unconvolved=True)
        image_obs = multi_band_list[0][0]["image_data"]
        mask = get_lens_mask(lens,image_obs,plot_mask=False)
        limits.append([_limits(np.log10(image_obs))])
        nms_lenses.append(lens.name)
        obs_images.append(image_orig)
        sim_images.append(image_obs*mask)
        ext = lens.gallens.kw_extents["extent_arcsec"]
        extents.append(ext)
        del lens
        gc.collect()
        """
        fig,axis = plt.subplots(1,1)
        ax = axis[0]
        im0 =ax.imshow(image_orig,origin="lower",extent=ext,cmap="hot")
        ax.set_xlabel('RA ["]')
        ax.set_ylabel('DEC ["]')
        ax.set_title("Image")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical',label=label_clb)
        ax = axis[1]
        im0 =ax.imshow(image_obs*mask,origin="lower",extent=ext,cmap="hot")
        ax.set_xlabel('RA ["]')
        ax.set_ylabel('DEC ["]')
        ax.set_title("Image observed masked")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical',label=label_clb)
        fig.tight_layout()
        nm = f"{lens.model_res_dir}/kappa_gal.png"
        fig.savefig(nm)
        print(f"Saving {nm}")
        """
    del gal_lenses
    with open("tmp/del_fail.dll","wb") as f:
        dill.dump(err_fail,f)
    print("N fails:",len(err_fail),"\n% fails:",np.round((len(err_fail)/N_gallenses)*100,1))
    plot_image_pairs_pdf(images1 = sim_images,
                         images2 = obs_images,
                         names   = nms_lenses,
                         extents = extents,
                         limits  = limits,
                         output_pdf = "tmp/all_sim_lenses.pdf",
                         cmap1 = "hot",
                         cmap2 = "hot",
                         log_scale1 = True,
                         log_scale2 = True,
                         label1 = "Simulated images",
                         label2 = "Sim. Observed (+mask)")
