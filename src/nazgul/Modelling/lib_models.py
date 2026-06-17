# Copy from model_allLOS.py
# adapted to have all the required components for the specific models

import os
import warnings
import argparse
import numpy as np
import sys,json,dill
from pathlib import Path
from corner import corner
from copy import copy,deepcopy
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lenstronomy.Plots import chain_plot
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST

from python_tools.read_fits import load_fits
from python_tools.tools import mkdir,to_dimless
from python_tools.get_res import load_whatever

from nazgul.plot_PL import plot_all
from nazgul.Translator import std_sim,std_simsuite,std_subsim
from nazgul.masking import mask_SEAGLE,mask_max_dens,mask_bright_center,resize_mask
from nazgul.mount_doom.cracks_of_doom import LoadLens,get_extents
from nazgul.mount_doom.lens_system import LensSystem
from nazgul.plot_PL import plot_kappamap
from nazgul.stat_lenses import get_all_gallens


from nazgul.lens_part_LOS import get_kw_los
from nazgul.pathfinder import get_sim_dir,tmp_dir

lens_model_list_def   = ['EPL']
source_model_list_def = ["SERSIC"]
model_res_base      = tmp_dir/"models/"
#PSO
n_it_std   = 2000
n_part_std = 500
#MCMC
n_burn_std = 1000
n_run_std  = 10000

def _get_model_res_dir(lens,res_dir):
    res_dir = Path(f"{res_dir}/snap_{lens.gallens.Gal.snap}_{lens.name}")
    return res_dir
    
def get_link_lens_path(lens,res_dir):
    if not hasattr(lens,"model_res_dir"): 
        lens.model_res_dir = _get_model_res_dir(lens,res_dir=res_dir)
    return lens.model_res_dir/"link_gallens.pkl"

workin_on_it = "WOI.dll"
def set_workin_on_it(dir,wrk=True):
    woi_file = Path(dir)/workin_on_it
    with open(woi_file,"wb") as f:
        dill.dump({"workin_on_it":wrk},f)
    
def is_someone_workin_on_it(dir):
    woi_file = Path(dir)/workin_on_it
    if not woi_file.is_file():
        return False
    else:
        return load_whatever(woi_file)
    
def setup_lens(lens,res_dir,_plot=True,verbose=True):
    lens.model_res_dir = _get_model_res_dir(lens,res_dir=res_dir)
    lens.setup()
    lens.image_sim = lens.get_lensed_image(unconvolved=False)
    mkdir(lens.model_res_dir)
    
    if verbose:
        print(f"Saving modelling results in {lens.model_res_dir}") 
    # For conveniency, but likely not the best idea:
    print("TODO: This is valid ONLY when we are only modelling a galaxy")
    lens.kw_extents = get_extents(arcXkpc=lens.gallens.arcXkpc,
                                  _radec=lens.gallens._radec)
    lens.kappa_map = lens.gallens.kappa_map
    lens.Gal = lens.gallens.Gal
    lens.z_lens = lens.gallens.z_lens
    lens.z_source = lens.gallens.z_source
    lens.deltaPix = lens.gallens.deltaPix
    lens.pixel_num = lens.gallens.pixel_num

    
    #lens.kwargs_lens = lens.gallens.kwargs_lens
    if _plot:
        plot_all(lens,skip_caustic=True)

    # create link to lens
    src = lens.gallens.pkl_path
    dst = get_link_lens_path(lens,res_dir=res_dir)
    if not os.path.islink(dst):
        os.symlink(src,dst)
    return lens
    
    
def setup_sim_obs(lens,band_str="HST_F160W",pssf=3):
    if band_str=="HST_F160W":
        band     = HST(band='WFC3_F160W', psf_type="PIXEL")
        # TOOD: bug (?) in lenstronomy, psf_type is not passed to band
        band.obs["psf_type"] = "PIXEL"
        psf_path = Path(f"./ObsData/HST/WFC3/F160W/PSFSTD_WFC3IR_F160W.fits")
    else:
        raise RuntimeError("Pragma no cover: to implement other bands and PSFs")
    
    # the following contain 9 PSF depending on their position in the CCD
    # we will consider the central position -> i think it should be the second to last (not too important here )
    psf = load_fits(psf_path)[-2]
    if np.any(psf<0):
        warnings.warn("Some negative pixels in the PSF")
        i_psf0,j_psf0 = np.where(psf<0)
        if i_psf0.shape[0]*100/psf.ravel().shape[0]>30:
            raise ValueError("PSF has more than 30% negative pixels, something is not right")
        warnings.warn("Setting minimum value for negative PSF pixels")
        psf[psf<0] = np.min(psf[psf>0])/100
    # we can supersample it
    if pssf!=1:
        psf_ss  = zoom(psf,pssf,order=3)
        kwargs_psf = {"kernel_point_source":psf_ss,
                      "point_source_supersampling_factor":pssf}
    else:
        kwargs_psf = {"kernel_point_source":psf}
    
    multi_band_list = lens.sim_multi_band_list(band=band,
                                               kwargs_psf=kwargs_psf)
    return multi_band_list


def get_lens_mask(lens,image_obs,plot_mask=True):
    # masking inner and outer of thetaE -> nope, follow SEAGLE approach
    #image = kwargs_data["image_data"]
    mask_HD = mask_SEAGLE(lens)*mask_bright_center(lens,rad=lens.gallens.thetaE*.5) #mask_max_dens(lens)
    mask_LD = resize_mask(mask_HD,image_obs)
    if plot_mask:
        plt.close()
        plt.close("all")
        kw_extents = lens.kw_extents
        extent_arcsec = kw_extents["extent_arcsec"]
        kw_plot = {"cmap":"hot","extent":extent_arcsec,"origin":"lower"}
        fig,axes = plt.subplots(2,2, figsize=(10, 10))
        ax =axes[0][0]
        im0 = ax.imshow(np.log10(lens.image_sim),**kw_plot)
        ax.set_title("Image")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   
        
        ax =axes[0][1]
        ax.set_title("Masked Image")
        mask_nan = copy(mask_HD)
        mask_nan[np.where(mask_HD==0)] = np.nan
        im0 = ax.imshow(np.log10(mask_nan*lens.image_sim),**kw_plot)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   

        ax =axes[1][0]
        im0 = ax.imshow(np.log10(image_obs),**kw_plot)
        ax.set_title("Image (Realistic)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   
        
        ax =axes[1][1]
        ax.set_title("Masked Image (Realistic)")
        mask_nan = copy(mask_LD)
        mask_nan[np.where(mask_LD==0)] = np.nan
        im0 = ax.imshow(np.log10(mask_nan*image_obs),**kw_plot)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')   
        
        nm = f"{lens.model_res_dir}/{lens.name}_masked_im.png"
        print(f"Saving {nm}")
        plt.savefig(nm)
    return mask_LD
    
def get_kwargs_likelihood(lens,image_obs,plot_mask=True):
    mask = get_lens_mask(lens,image_obs,plot_mask=plot_mask)
    
    kwargs_likelihood = {'check_bounds': True, # punish out-of-bound soulutions
                     #'force_no_add_image': False,
                     'source_marg': False, # marginalization addition on the imaging likelihood based on the covariance of the inferred linear coefficients 
                     #'image_position_uncertainty': 0.004,
                     #'check_matched_source_position': True,
                     #'source_position_tolerance': 0.001,
                     'source_position_sigma': 0.01,
                     #'prior_lens': prior_lens
                      "image_likelihood_mask_list": [mask]  }
    return kwargs_likelihood
    
def get_kwargs_params_def(lens):
    raise RuntimeError("This function should be taken as a reference and re-implemented in the specific model")

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
 
    lens_params = [kwargs_lens_init, kwargs_lens_sigma, [{}, kwargs_fixed_los], kwargs_lower_lens, kwargs_upper_lens]
    source_params = [kwargs_source_init, kwargs_source_sigma, [{}], kwargs_lower_source, kwargs_upper_source]
    
    kwargs_params = {'lens_model': lens_params,
                    'source_model': source_params}
    return kwargs_params

def _get_lenses2model(kw_get_all_gallens={"snaps":[27]},n_lenses=None,min_thetaE=None,skip_lenses=[]):
    precomputed_lenses =  np.array(get_all_gallens(**kw_get_all_gallens))
    if min_thetaE is None and n_lenses is None:
        n_lenses = 5
        print(f"No boundary given - assuming n_lenses={n_lenses} - If you really want all the lenses, give n_lenses=np.nan")
    
    theta = []
    for lens  in precomputed_lenses:
        theta.append(lens.thetaE.value)
    theta = np.array(theta)
    if min_thetaE is None:
        if np.isnan(n_lenses):
            print("Assuming no boundary")
            lenses_selected = precomputed_lenses
        else:
            lenses_sort     = precomputed_lenses[theta.argsort()][::-1]
            lenses_selected = lenses_sort[:n_lenses]
    else:
        lenses_selected = precomputed_lenses[np.where(theta>min_thetaE)]
        if n_lenses:
            if not np.isnan(n_lenses):
                lenses_selected = lenses_selected[:n_lenses]
                
    if len(skip_lenses)!=0:
        lenses_accepted = []
        for l in lenses_selected:
            if l.name not in skip_lenses:
                lenses_accepted.append(l)
            else:
                print(f"Ignoring lens {l} because in the list of lenses to skip")
        lenses_selected = lenses_accepted

    if len(lenses_selected)==0:
        raise RuntimeError("The boundary given are too strict - no galaxy satisfy it")
    return lenses_selected

def get_lenses2model(res_dir,reload=True,**kw_get_lenses2model):
    cat_l2m = Path(res_dir)/"cat_lens2model.dll"
    update_cat  = True
    if cat_l2m.is_file():
        if reload:
            print(f"Loading previously computed catalogue of lenses to models {cat_l2m}") 
            kw_cat_lens = load_whatever(cat_l2m)
            if not verify_same_requirements(kw_cat_lens["kw_require"],kw_get_lenses2model):
                print(f"Catalogue {cat_l2m} exists, but doen't have the same requierements. Ignored and updated")
                update_cat = True
                lenses = _get_lenses2model(**kw_get_lenses2model)
            else:
                update_cat = False
                cat_lens = kw_cat_lens["lens_cat"]
                lenses_2unpack = [load_whatever(l) for l in cat_lens]
                lenses = [l.unpack() for l in lenses_2unpack]
        else:
            print(f"Catalogue {cat_l2m} exists, but ignored")
    else:
        print(f"Catalogue {cat_l2m} doesn't exists - creating it now")
        update_cat = True
        lenses = _get_lenses2model(**kw_get_lenses2model)
        
    if update_cat:
        lenses_cat = [lenses.pkl_path for l in lenses]
        kw_cat_lens = {"lens_cat":lenses_cat,
                       "kw_require":kw_get_all_gallens}
        with open(cat_l2m,"wb") as f:
            dill.dump(kw_cat_lens,f)
        print(f"Saving catalogue of lenses to models {cat_l2m}") 
    return lenses
    
def save_kw(kw,nm_kw,str_kw_type=""):
    with open(nm_kw,"wb") as f:
        dill.dump(kw,f)
    print(f"Saving {str_kw_type} kw: {nm_kw}")

def plot_model_plot(multi_band_list_out,kwargs_model,kwargs_result,kwargs_likelihood,res_dir,plot_point_sources=False):
    modelPlot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result, 
                          arrow_size=0.02, cmap_string="gist_heat",
                          image_likelihood_mask_list=kwargs_likelihood["image_likelihood_mask_list"])
    
    band_index_plot = 0
    
    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
    
    modelPlot.data_plot(ax=axes[0,0], band_index=band_index_plot)#,v_min=,v_max=band_i.vmax)
    modelPlot.model_plot(ax=axes[0,1], band_index=band_index_plot)#,v_min=band_i.vmin,v_max=band_i.vmax)
    modelPlot.normalized_residual_plot(ax=axes[0,2], band_index=band_index_plot)#,v_min=band_i.res_vmin,v_max=band_i.res_vmax)
    modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100, band_index=band_index_plot)#,v_min=band_i.vmin,v_max=band_i.vmax)
    modelPlot.convergence_plot(ax=axes[1, 1], band_index=band_index_plot,v_max=1)
    modelPlot.magnification_plot(ax=axes[1, 2], band_index=band_index_plot)
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    nm = f'{res_dir}/mass_model.pdf'
    plt.savefig(nm)
    plt.close(f)
    print(f"Saving {nm}")
    
    f, axes = plt.subplots(1,2, figsize=(8, 4), sharex=False, sharey=False)
    
    modelPlot.decomposition_plot(ax=axes[0], band_index=band_index_plot, text='Source light', source_add=True, unconvolved=True)#,v_min=band_i.vmin,v_max=band_i.vmax)
    modelPlot.decomposition_plot(ax=axes[1], band_index=band_index_plot, text='Source light convolved', source_add=True)#,v_min=band_i.vmin,v_max=band_i.vmax)
    
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    nm = f'{res_dir}/light_model.pdf'
    plt.savefig(nm)
    plt.close(f)
    print(f"Saving {nm}")
    
    kwargs_result_wo_tracer = deepcopy(kwargs_result)
    keys = copy(list(kwargs_result_wo_tracer.keys()))
    for k in keys:
        if "tracer" in str(k):
            del kwargs_result_wo_tracer[k]
    ### 
    logL,_prms   = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result_wo_tracer)
    
    n_data = modelPlot._imageModel.num_data_evaluate
    reduced_chi2  = -logL * 2 / n_data
    print(f'{np.round(reduced_chi2,2)} reduced X^2 of all evaluated imaging data combined\n')
    print("################################\n")
    
    #Normalised plot
    f, axes = plt.subplots(figsize=(10,7))
    modelPlot.normalized_residual_plot(ax=axes,v_min=-3, v_max=3,text=r"Norm. Resid $\chi^2_{red.}$="+str(np.round(reduced_chi2,2)))
    nm = f'{res_dir}/normalised_residuals.png'
    plt.savefig(nm)
    print(f"Saving {nm}")
    plt.close(f)
    
    #Caustics
    f, axes = plt.subplots(figsize=(10,7))
    modelPlot.source_plot(ax=axes, deltaPix_source=0.01, numPix=1000, with_caustics=True)
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    nm = f'{res_dir}/caustics.png'
    plt.savefig(nm)
    print(f"Saving {nm}")
    plt.close(f)
    if plot_point_sources:
        f, axes = plt.subplots(figsize=(10,7))
        modelPlot.decomposition_plot(ax=axes, text='Point source position', source_add=False, \
                        lens_light_add=False, point_source_add=True, v_min=-1, v_max=1)
        nm = f'{res_dir}/point_source_position.png'
        plt.savefig(nm)
        print(f"Saving {nm}")
        plt.close(f)


if __name__=="__main__":
    raise RuntimeError("Do not run this - kept only as a reference")
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Simulate and model the lens")
    parser.add_argument('-rt','--run_type',type=int,dest="run_type",default=0,help= f"""Type of run: 
        0 = standard, PSO_it = {n_it_std} PSO_prt = {n_part_std} MCMCb = {n_burn_std} MCMCr = {n_run_std}  
        1 = test run  PSO_it = 3      PSO_prt = 3      MCMCb = 1     MCMCr = 2 
       (PSO_it: PSO iterations, PSO_prt: PSO particles, MCMCr: MCMC run steps, MCMCb: MCMC burn in steps)\n""")
    parser.add_argument('-nl','--n_lenses',type=int,dest="n_lenses",default=5,help=f"Number of lenses to model")
    parser.add_argument('-mtE','--min_thetaE',type=float,dest="min_thetaE",default=None,help=f"Min theta_E for the gal to be considered a lens")
    parser.add_argument('-snap','--snap',nargs="+",type=int,dest="snaps",default=[],help=f"List of snaps to consider - default is all")
    parser.add_argument('-sim','--sim',type=str,dest="sim",default=std_sim,help=f"Simulation name")
    parser.add_argument('-ss','--simsuite',type=str,dest="simsuite",default=std_simsuite,help=f"Simulation suite name")
    parser.add_argument('-ssim','--subsim',type=str,dest="subsim",default=std_subsim,help=f"Sub-Simulation name")

    args       = parser.parse_args()
    run_type   = args.run_type
    n_lenses   = args.n_lenses
    min_thetaE = args.min_thetaE 
    snaps      = args.snaps #[25,26,27]
    sim        = args.sim
    subsim     = args.subsim
    simsuite   = args.simsuite
    if run_type==0:
        n_iterations = int(n_it_std) #number of iteration of the PSO run
        n_particles  = int(n_part_std) #number of particles in PSO run
        n_burn = int(n_burn_std) #MCMC burn in steps
        n_run  = int(n_run_std) #MCMC total steps 
    elif run_type ==1:
        print("Test Run")
        n_iterations = int(3) #number of iteration of the PSO run
        n_particles  = int(3) #number of particles in PSO run
        n_run  = int(2) #MCMC total steps 
        n_burn = int(1) #MCMC burn in steps
    else:
        raise RuntimeError("Give a valid run_type or implement it your own")

    lenses2skip = ["Sub_Gn22SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj1","Sub_Gn3SGn0_Npix200_PartAS_Prj2"]
    #["Sub_Gn3SGn0_Npix200_PartAS_Prj1","Sub_Gn22SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj0","Sub_Gn3SGn0_Npix200_PartAS_Prj2"]
    kw_get_all_gallens = {"sim":sim,
                          "subsim":subsim,
                           "simsuite":simsuite,
                            "snaps":snaps}
    res_dir = res_dir_base
    if run_type==1:
        res_dir = res_dir_base/"test"
    gal_lenses  = get_lenses2model(res_dir=res_dir,
                                   reload=True,
                                   kw_get_all_gallens=kw_get_all_gallens,
                                   n_lenses=n_lenses,
                                   min_thetaE=min_thetaE,
                                   skip_lenses=lenses2skip)
    for gal_lens in gal_lenses: 
        print("Loading lens "+gal_lens.name+"\n")
        print("Adding LOS effects")
        kw_los = get_kw_los()
        kw_add_lenses = {"lens_model_list":["LOS"],
                        "kwargs_lens":[kw_los]}
        lens = LensSystem.from_GalLens(gal_lens,kwargs_add_lenses=kw_add_lenses)

        lens = setup_lens(lens,res_dir=model_res_base) #change it with res_dir_base of the given model
        # verify that no-one is working on it
        if is_someone_workin_on_it(lens.model_res_dir):
            print(f"This lens is being worked on, skipping- if not, delete the {workin_on_it} file") 
            continue
        set_workin_on_it(lens.model_res_dir,wrk = True)
        
        plot_kappamap(lens.gallens.kappa_map, 
                      extent_kpc=lens.gallens.kw_extents["extent_kpc"],
                      savename=f"{res_dir}/kappa_gal.png")
        multi_band_list = setup_sim_obs(lens)
        image_obs = multi_band_list[0][0]["image_data"]
        
        # models
        kwargs_model = {'lens_model_list': lens_model_list,
                        'source_light_model_list': source_model_list}
        
        
        kwargs_likelihood = get_kwargs_likelihood(lens,image_obs=image_obs)
    
        kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
    
        # Params:
        
        kwargs_params = get_kwargs_params(lens)
    
        kwargs_constraints = {#'joint_source_with_point_source': [[0, 0]], 
        #    'joint_source_with_point_source': list [[i_point_source, k_source], [...], ...],
        #     joint position parameter between lens model and source light model
                               #   'num_point_source_list': [4],
                                  'solver_type':'NONE'# 'PROFILE_SHEAR',  # 'PROFILE', \
                              #'PROFILE_SHEAR', 'ELLIPSE', 'CENTER', 'NONE'
                                  }
    
        
        # actual fit:
        from lenstronomy.Workflow.fitting_sequence import FittingSequence
        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
        fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': n_particles, 'n_iterations':n_iterations}]
                          ,
                           ['MCMC', {'n_burn': n_burn, 'n_run': n_run, 'walkerRatio': 5, 'sigma_scale': .1}]
            ]
        kw_input = {"kwargs_data_joint":   kwargs_data_joint,
                    "kwargs_model":        kwargs_model, 
                    "kwargs_constraints":  kwargs_constraints, 
                    "kwargs_likelihood":   kwargs_likelihood, 
                    "kwargs_params":       kwargs_params,
                    "fitting_kwargs_list": fitting_kwargs_list,
                    "kw_add_lenses":       kw_add_lenses
                   }
        nm_input = f"{res_dir}/kw_input.dll"
        save_kw(kw_input,nm_input,"input")
        
        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()
        print("kwargs_result",kwargs_result)
        nm_res = f"{res_dir}/kw_res.dll"
        save_kw(kwargs_result,nm_res,"result output")
        
        
        # we need to extract the updated multi_band_list object since the coordinate shifts were updated in the kwargs_data portions of it
        multi_band_list_out = fitting_seq.multi_band_list
        nm_mblo = f"{res_dir}/multi_band_list_out.dll"
        save_kw(multi_band_list_out,nm_mblo,"output multiband list")


        plot_model_plot(multi_band_list_out,kwargs_model,kwargs_result,kwargs_likelihood,res_dir=res_dir)
        
        if run_type!=1:
            sampler_type, mc_sample, param_mcmc, mc_logL  = chain_list[-1]
            chnl_path = f'{lens.model_res_dir}/chain_list.dll'
            save_kw(chain_list,chnl_path,"chain list")

            corner(mc_sample,labels=param_mcmc,show_titles=True,plot_datapoints=False,hist_kwargs= {"density":True})
            nm = f'{lens.model_res_dir}/mcmc_post.pdf'
            plt.savefig(nm)
            print(f"Saving {nm}")
            plt.close()
            
        # test
        for i in range(len(chain_list)):
            chain_plot.plot_chain_list(chain_list, i)
        nm = f'{lens.model_res_dir}/chain_plot.pdf'
        plt.savefig(nm)
        print(f"Saving {nm}")
        plt.close()
        set_workin_on_it(lens.model_res_dir,wrk = False)
    