import numpy as np
from glob import glob
from pathlib import Path
from copy import copy,deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.Plots.model_plot import ModelPlot

from python_tools.tools import short_SciNot
from python_tools.get_res import load_whatever

from nazgul.modelling_severals import res_dir_base
from nazgul.mount_doom.cracks_of_doom import LoadLens



def get_all_lens_models(res_dir=res_dir_base):
    pth_modlenses_res = glob(f"{res_dir}/snap*/kw_res.json")
    lenses = []
    for pth_res in pth_modlenses_res:
        try:
            # if res exists AND is loaded correctly
            load_whatever(pth_res)
            # then we consider the lens
            model_res_dir = Path(pth_res).parent
            lens_link = model_res_dir/"link_gallens.pkl"
            lens      = LoadLens(lens_link)
            lens.unpack() 
            lens.model_res_dir = model_res_dir
            lenses.append(lens)
        except Exception as e:
            print(f"Failed to load {pth_res} due to {e} - skipping.")
    return lenses
    
lenses_modelled = get_all_lens_models()
columns_ttl = ["Sim Image","Model","Norm. Resid.",r"P($\theta_E$|S.I.)",r"P($\gamma_{ext.}$|S.I.)"]

nrows  = len(lenses_modelled)
ncols  = len(columns_ttl) # n* of wanted columns

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))


for i_row,lens in enumerate(lenses_modelled):

    chnl_path = f'{lens.model_res_dir}/chain_list.dll'
    chain_list = load_whatever(chnl_path)
    sampler_type, mc_sample, param_mcmc, mc_logL  = chain_list[-1]
    param_mcmc = np.array(param_mcmc)
    mc_sample = np.array(mc_sample) 
    
    nm_mblo = f"{lens.model_res_dir}/multi_band_list_out.json"
    multi_band_list_out = load_whatever(nm_mblo)
    nm_input = f"{lens.model_res_dir}/kw_input.dll"            
    kw_input = load_whatever(nm_input)
    kwargs_data_joint= kw_input["kwargs_data_joint"] 
    kwargs_model = kw_input["kwargs_model"]
    kwargs_constraints  = kw_input["kwargs_constraints"] 
    kwargs_likelihood  = kw_input["kwargs_likelihood"]
    kwargs_params  = kw_input["kwargs_params"]
    fitting_kwargs_list  = kw_input["fitting_kwargs_list"]

    nm_res = f"{lens.model_res_dir}/kw_res.json"
    kwargs_result = load_whatever(nm_res)
    modelPlot = ModelPlot(multi_band_list_out, kwargs_model, kwargs_result, 
                          arrow_size=0.02, cmap_string="gist_heat",
                          image_likelihood_mask_list=kwargs_likelihood["image_likelihood_mask_list"])

    model_band = modelPlot._band_plot_list[0]
    kw_modelplot = {"vmin":model_band._v_min_default,
                    "vmax":model_band._v_max_default,
                    "extent":model_band._image_extent,
                    "origin":"lower",
                    "cmap":model_band._cmap}
    # Sim Image
    ax = axes[i_row][0]
    ax.set_ylabel(lens.name.replace("Sub_",""))
    if i_row ==0:
        ax.set_title(columns_ttl[0])
    ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])
    
    im0 = ax.imshow(np.log10(model_band._data),**kw_modelplot)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"flux$_{data}$")

    # Model
    ax = axes[i_row][1]
    if i_row ==0:
        ax.set_title(columns_ttl[1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    im0 = ax.imshow(np.log10(model_band._model),**kw_modelplot)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"flux$_{model}$")


    # Residual
    ax = axes[i_row][2]
    if i_row ==0:
        ax.set_title(columns_ttl[2])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    kwargs_result_wo_tracer = deepcopy(kwargs_result)
    keys = copy(list(kwargs_result_wo_tracer.keys()))
    for k in keys:
        if "tracer" in str(k):
            del kwargs_result_wo_tracer[k]

    logL,_prms   = modelPlot._imageModel.likelihood_data_given_model(source_marg=False, linear_prior=None, **kwargs_result_wo_tracer)

    n_data = modelPlot._imageModel.num_data_evaluate
    reduced_chi2  = -logL * 2 / n_data
    kw_modelplot_resid = deepcopy(kw_modelplot)
    kw_modelplot_resid["vmin"] = -3
    kw_modelplot_resid["vmax"] = 3
    kw_modelplot_resid["cmap"] = "bwr"
    im0 = ax.imshow(model_band._norm_residuals,**kw_modelplot_resid)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical',label=r"(f$_{model}$-f$_{data}$)/$\sigma$")
    
    x_txt = (model_band._image_extent[1]-model_band._image_extent[0])*3.5/5
    y_txt = (model_band._image_extent[3]-model_band._image_extent[2])*4/5
    ax.text(x_txt,y_txt,s=r"$\chi^2_{red.}$="+str(np.round(reduced_chi2,2)),color="k",backgroundcolor="w")
        
    # Posterior Theta_E
    ax = axes[i_row][3]
    if i_row ==0:
        ax.set_title(columns_ttl[3])
    
    i_thetaE = np.where(param_mcmc=="theta_E_lens0")
    thetaE = mc_sample.T[i_thetaE][0]
    lbl_tE_meas = r"$\theta_{E\;meas.}=$"+str(np.round(np.median(thetaE),3))+"+-"+str(np.round(np.std(thetaE),3))
    ax.hist(thetaE,bins=40,label=lbl_tE_meas)
    lbl_tE_true = r"$\theta_{E\;true}=$"+short_SciNot(lens.thetaE)
    ax.axvline(lens.thetaE.value,ls="--",color="k",label=lbl_tE_true)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(r"$\theta_E$")
    ax.legend()
    # Posterior gamma_ext
    ax = axes[i_row][4]
    if i_row ==0:
        ax.set_title(columns_ttl[4])
    i_gamma = np.where(param_mcmc=="gamma_ext_lens1")
    gamma = mc_sample.T[i_gamma][0]
    ax.hist(gamma,bins=40,label=r"$\gamma_{ext\;meas.}=$"+str(np.round(np.median(gamma),3))+"+-"+str(np.round(np.std(gamma),3)))
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(r"$\psi_{ext}$")
    ax.legend()

plt.tight_layout()
nm = f"{res_dir_base}/combined_result.pdf"
plt.savefig(nm)
print(f"Saved {nm}")
