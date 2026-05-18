# "Opposite" of combined_modelling_results: instead of plotting all the lenses for 1 model, we plot all model results for 1 lens -run it for all lenses
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from copy import copy,deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from chainconsumer import Chain, ChainConsumer
from chainconsumer.plotting import plot_contour,plot_truths,plot_dist

from lenstronomy.Plots.model_plot import ModelPlot

from python_tools.tools import short_SciNot
from python_tools.get_res import load_whatever

from nazgul.lens_part_LOS import get_kw_los
from nazgul.mount_doom.cracks_of_doom import LoadLens


# thank you Nat!
green        = ['#a6dba0','#5aae61','#1b7837']
purple       = ['#c2a5cf', '#9970ab', '#762a83']
analogous    = ['#a0c3db', '#dbb7a0']
warm         = ['#fdcc8a', '#fc8d59', '#d7301f']
cool         = ['#41b6c4', '#2c7fb8', '#253494']

#plt.style.use('sanglier')
plt.rcParams.update({'font.size': 20})

from nazgul.combined_modelling_results import plot_result_line,_convert_shear2LOS,get_all_lens_models,name_models,get_res_dir,get_model_title
c = ChainConsumer()
import warnings
warnings.filterwarnings("ignore")
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot Combined results for all the models' run of a given lens (and run it for all lenses)")
    parser.add_argument('-ove','--overlay_ellipticity', dest="overlay_ellipticity", 
                        default=False, action="store_true",
                        help=f"If true, overlay ellipticity posterior")
    args                = parser.parse_args()
    overlay_ellipticity = args.overlay_ellipticity

    path_res = "./tmp/models/."
    # to which cifra significativa to round
    _rnd = 3
    kw_lenses_modelled = {}
    set_lenses = [] 
    for model in name_models:
        res_dir = get_res_dir(model)        
        lenses_modelled = get_all_lens_models(res_dir)
        kw_lenses_modelled[model]=lenses_modelled
        [set_lenses.append(l.name.replace("Sub_","")) for l in lenses_modelled]
    name_lenses = list(set(set_lenses))
    
    columns_ttl = ["Sim Image","Model","Norm. Resid.",r"P($\theta_E$|S.I.)"]
    if overlay_ellipticity:
        columns_ttl.append(r"P($\gamma_{LOS,1}$,$\gamma_{LOS,2}$|S.I.) + P(e$_1$,e$_2$|S.I.)")
    else:
        columns_ttl.append(r"P($\gamma_{LOS,1}$,$\gamma_{LOS,2}$|S.I.)")

    
    ncols  = len(columns_ttl)
    for i_lens,lens_name in enumerate(name_lenses):
        scl = 8
        fig, axes = plt.subplots(nrows, ncols, figsize=(scl*ncols,scl*nrows))
        for i_row,model in enumerate(name_models):
            lenses = kw_lenses_modelled[model]
            for lens in lenses:
                if lens.name.replace("Sub_","") == lens_name:
                    break
                del lens
            try:
                lens
            except:
                continue
            plot_result_line(model,lens,axes,i_row,nrows,columns_ttl,_rnd=_rnd,overlay_ellipticity=overlay_ellipticity)
            columns_ttl_i = columns_ttl[-1]
            if "noLOS" in model:
                columns_ttl_i = columns_ttl[-1].replace("LOS","Shear")                
            axes[i_row][-1].set_title(columns_ttl_i)
            ttl = get_model_title(model)
            if i_row>0:
                axes[i_row][2].set_title(ttl)
            else:
                plt.suptitle(ttl)
        plt.tight_layout()
        nm_mod_comb = f"{path_res}/{lens_name}_mod_comb.pdf"
        plt.savefig(nm_mod_comb)
        print(f"Saved {nm_mod_comb}")

"""
if model!="noLOS":
    gamma1_full = c.analysis.get_parameter_summary(chain=Chain(samples=full_chain, name='lenstronomy_mcmc_emcee'),column=r"gamma1_los_lens1")
    gamma2_full = c.analysis.get_parameter_summary(chain=Chain(samples=full_chain, name='lenstronomy_mcmc_emcee'),column=r'gamma2_los_lens1')

    nc_full_g1 =  np.isfinite(gamma1_full.array[0]) # checks if the lower bound is finite (CC returns NaN for both upper and lower bounds when param is unconstrained)
    nc_full_g2 =  np.isfinite(gamma2_full.array[0])
    covmat_full = Chain(samples=full_chain.loc[:, ['gamma1_los_lens1', 'gamma2_los_lens1']], name='lenstronomy_mcmc_emcee').get_covariance().matrix
    shear_mag_full = shear_magnitude(gamma1_full, gamma2_full)
    shear_stdev_full = shear_stdev(shear_mag_full, gamma1_full, gamma2_full, covmat_full)        
 
gamma_epl_full = c.analysis.get_parameter_summary(chain=Chain(samples=full_chain, name='lenstronomy_mcmc_emcee'),column=r'gamma_lens0')
"""
