# Try to generalise it
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


c = ChainConsumer()
# thank you Nat!
green        = ['#a6dba0','#5aae61','#1b7837']
purple       = ['#c2a5cf', '#9970ab', '#762a83']
analogous    = ['#a0c3db', '#dbb7a0']
warm         = ['#fdcc8a', '#fc8d59', '#d7301f']
cool         = ['#41b6c4', '#2c7fb8', '#253494']

#plt.style.use('sanglier')
plt.rcParams.update({'font.size': 20})

def shear_stdev(gamma, gamma1, gamma2, covmat):
    '''
    computes standard deviation on magnitude of gamma_LOS from covariance matrix
    '''
    first_term = ((gamma1.array[1]/gamma)**2.)*covmat[0,0]
    second_term = ((gamma2.array[1]/gamma)**2.)*covmat[1,1]
    third_term = ((2*gamma1.array[1]*gamma2.array[1])/gamma)*covmat[0,1]
    variance = first_term + second_term + third_term
    st_dev = np.sqrt(variance)
    return st_dev

def shear_magnitude(gamma1, gamma2):
    '''
    computes magnitude of gamma_LOS
    '''
    shear = np.sqrt(gamma1.array[1]**2. + gamma2.array[1]**2.)
    return shear

def lims_getter(g1, g2):
    '''
    gets x and y limits for the contour plot based on the size of the posterior, and ensures a square aspect ratio
    '''
    nsig = 5 # number of sigmas from the mean sets the plot lims
    g1_nsig = nsig*(g1.array[1] - g1.array[0]) # assuming Gaussian posterior i.e. upper and lower lims are the same
    g2_nsig = nsig*(g2.array[1] - g2.array[0])

    x_lower = g1.array[1] - g1_nsig
    x_upper = g1.array[1] + g1_nsig

    y_lower = g2.array[1] - g2_nsig
    y_upper = g2.array[1] + g2_nsig

    xdiff = x_upper - x_lower
    ydiff = y_upper - y_lower

    if max(xdiff, ydiff) == xdiff:
        y_lower_new = g2.array[1] - g1_nsig
        y_upper_new = g2.array[1] + g1_nsig
        return x_lower, x_upper, y_lower_new, y_upper_new
    else:
        x_lower_new = g1.array[1] - g2_nsig
        x_upper_new = g1.array[1] + g2_nsig
        return x_lower_new, x_upper_new, y_lower, y_upper

def get_all_lens_models(res_dir):
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


def _convert_shear2LOS(mc_sample,param_mcmc):
    """
    Only for convenience - rewrite gamma_ext, psi_ext as gamma_LOS_1,gamma_LOS_2
    """
    # needed to convert gamma_shear, psi_shear into gamma_shear1,gamma_shear2
    from lenstronomy.Util.param_util import shear_polar2cartesian,ellipticity2phi_q

    mc_sampleT =  np.array(mc_sample).T
    param_mcmc = list(param_mcmc)
    i_g1 = param_mcmc.index("gamma_ext_lens1")
    i_g2 = param_mcmc.index("psi_ext_lens1")
    gext,psiext = mc_sampleT[i_g1],mc_sampleT[i_g2]
    g1,g2 = shear_polar2cartesian(psiext,gext)
    mc_sampleT[i_g1] = g1
    mc_sampleT[i_g2] = g2

    param_mcmc[i_g1] = 'gamma1_los_lens1'
    param_mcmc[i_g2] = 'gamma2_los_lens1'
    
    return mc_sampleT.T, np.array(param_mcmc)

def _convert_polarshear2LOS(mc_sample,param_mcmc):
    """
    Only for convenience - rewrite gamma1, gamma2 as gamma_LOS_1,gamma_LOS_2
    """
    param_mcmc = list(param_mcmc)
    i_g1 = param_mcmc.index("gamma1_lens1")
    i_g2 = param_mcmc.index("gamma2_lens1")
    param_mcmc[i_g1] = 'gamma1_los_lens1'
    param_mcmc[i_g2] = 'gamma2_los_lens1'
    
    return mc_sample, np.array(param_mcmc)

def get_model_title(model):
    if model=="fitLOS":    
        ttl = "LOS not simulated, but modelled"
    elif model=="allLOS":    
        ttl = "LOS simulated and modelled"
    elif model=="fitLOS_fixedOD":
        ttl = r"LOS not simulated, but modelled (with fixed $\vec{\gamma_{od}}=0$)"
    elif model=="fitLOS_fixedOD_fixedOmgaLos":
        ttl = r"LOS not simulated, but modelled (with fixed $\vec{\gamma_{od}}=0$, $\omega_{\rm{LOS}}=0$)"
    elif model=="noLOS":
        ttl = r"LOS not simulated and external shear modelled"
    elif model=="noLOS_g12":
        ttl = r"LOS not simulated and external shear modelled (polar shear)"
    return ttl

def plot_los_outVsin(lenses,_rnd=3):
    model = "allLOS"
    fig,axes = plt.subplots(2,1,figsize=(12,16))
    #gamma_los_name =  r"$\gamma_{\rm{LOS}}$"
    str_glos   = r"$\gamma^{\rm{LOS}}$"
    str_glos_1 = r"$\gamma^{\rm{LOS}}_{1}$"
    str_glos_2 = r"$\gamma^{\rm{LOS}}_{2}$"
    
    str_glos_out_1 = r"$\gamma^{\rm{LOS}}_{1,\;\rm{out}}$"
    str_glos_out_2 = r"$\gamma^{\rm{LOS}}_{2,\;\rm{out}}$"
    str_glos_in_1 = r"$\gamma^{\rm{LOS}}_{1,\;\rm{in}}$"
    str_glos_in_2 = r"$\gamma^{\rm{LOS}}_{2,\;\rm{in}}$"

    str_glos_out = r"$\gamma^{\rm{LOS}}_{\rm{out}}$"
    str_glos_in = r"$\gamma^{\rm{LOS}}_{\rm{in}}$"

    g_los1_out_med,g_los2_out_med = [],[]
    g_los1_out_std,g_los2_out_std = [],[]
    g_los1_in,g_los2_in = [],[]

    g_los_out_med = []
    g_los_out_std = []
    g_los_in      = []
    for i,lens in enumerate(lenses):
        lens_name = lens.name.replace("Sub_","")
        nm_input = f"{lens.model_res_dir}/kw_input.dll"            
        kw_input = load_whatever(nm_input)

        chnl_path = f'{lens.model_res_dir}/chain_list.dll'
        chain_list = load_whatever(chnl_path)
        sampler_type, mc_sample, param_mcmc, mc_logL  = chain_list[-1]
        mc_sample  = np.array(mc_sample)
        param_mcmc = np.array(param_mcmc)
        i_gamma_los1 = np.where(param_mcmc=="gamma1_los_lens1")
        gamma_los1 = mc_sample.T[i_gamma_los1][0]
        i_gamma_los2 = np.where(param_mcmc=="gamma2_los_lens1")
        gamma_los2 = mc_sample.T[i_gamma_los2][0]
        
        gamma_los  = np.sqrt(gamma_los1**2 + gamma_los2**2)
        gamma_los_meas = np.median(gamma_los)
        gamma_los_std  = np.std(gamma_los)
        
        
        kw_add_lns = kw_input['kw_add_lenses']
        kw_los     = kw_add_lns["kwargs_lens"][kw_add_lns["lens_model_list"].index("LOS")]
        gamma_los1_true = kw_los["gamma1_od"] + kw_los["gamma1_os"] - kw_los["gamma1_ds"]
        gamma_los2_true = kw_los["gamma2_od"] + kw_los["gamma2_os"] - kw_los["gamma2_ds"]
        gamma_los_true =  np.sqrt(gamma_los1_true**2 + gamma_los2_true**2)
        
        g_los_out_med.append(gamma_los_meas)
        g_los_out_std.append(gamma_los_std)
        g_los_in.append(gamma_los_true)
        
        """
        gLOS_chain = pd.DataFrame( gamma_los , columns=gamma_los_name)
        plot_dist(ax, Chain(samples=gLOS_chain, name=gamma_los_name, shade=True, color='#2c7fb8', smooth=20, bins=10,
                                       shade_gradient = 0.4, linewidth=3.0), px=gamma_los_name,)
        lbl_glos_meas = r"$\gamma_{\rm{LOS}\;\rm{meas.}}=$"+str(np.round(np.median(gamma_los),_rnd))+"+-"+str(np.round(np.std(gamma_los),_rnd))
        ax.axvline(np.median(gamma_los),ls="-.",color="r",label=lbl_glos_meas)
        
        lbl_glos_true = r"$\gamma_{\rm{LOS}\;\rm{true}}=$"+str(np.round(gamma_los_true,_rnd))
        ax.axvline(gamma_los_true,ls="--",color="k",label=lbl_glos_true)
        ax.legend(loc="upper right")
        ax.scatter(i,gamma_los_true,marker="o",c="r",label=lbl_glos_true)
        ax.set_title(str_glos_meas+" vs "+str_glos_true)
        """
        g_los1_out_med.append(np.median(gamma_los1))
        g_los1_out_std.append(np.std(gamma_los1))
        g_los2_out_med.append(np.median(gamma_los2))
        g_los2_out_std.append(np.std(gamma_los2))
        g_los1_in.append(gamma_los1_true)
        g_los2_in.append(gamma_los2_true)

    if np.std(g_los1_in)>1e-10:
        axes = plot_los_outVsin_var(axes,
                             g_los1_in,g_los2_in,
                             g_los1_out_med,g_los1_out_std,
                             g_los2_out_med,g_los2_out_std,
                             str_glos_1,str_glos_2,
                             str_glos_in_1,str_glos_in_2,
                             str_glos_out_1,str_glos_out_2)
    else:
        print("NOTE: input glos is constant")
        axes = plot_los_outVsin_const(axes,
                             g_los1_in,g_los2_in,
                             g_los1_out_med,g_los1_out_std,
                             g_los2_out_med,g_los2_out_std,
                             str_glos_1,str_glos_2,
                             str_glos_in_1,str_glos_in_2,
                             str_glos_out_1,str_glos_out_2)

    
    #ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="grey")
    plt.suptitle(r"$\gamma_{\rm{in}}^{\rm{LOS}}$ vs $\gamma_{\rm{out}}^{\rm{LOS}}$")
    plt.tight_layout()
    plt.close()

    fig2,axes2 = plt.subplots(1,figsize=(9,6))
    if np.std(g_los_in)>1e-10:
        axes2.scatter(g_los_in,g_los_out_med)
        axes2.errorbar(g_los_in,g_los_out_med,yerr=g_los_out_std)
    else:
        print("NOTE: input glos is constant")
        ax = axes2
        x = np.arange(len(g_los_in))

        ax.set_ylabel(str_glos)
        ax.get_xaxis().set_ticks([])
        ax.set_xlabel("Lenses")
        ax.axhline(np.mean(g_los_in),ls="--",color="grey",label=str_glos_in)
        ax.scatter(x,g_los_out_med,marker="o",c="r",label=str_glos_out)
        ax.errorbar(x,g_los_out_med,yerr=g_los_out_std,c="r",fmt="ko",ecolor="k",elinewidth=.8)
        ax.legend()
    plt.suptitle(r"$\gamma_{\rm{in}}^{\rm{LOS}}$ vs $\gamma_{\rm{out}}^{\rm{LOS}}$")
    plt.tight_layout()
    plt.close()
    
    plt.tight_layout()
    return fig, fig2

def plot_los_outVsin_var(axes,g_los1_in,g_los2_in,
                             g_los1_out_med,g_los1_out_std,
                             g_los2_out_med,g_los2_out_std,
                             str_glos_1,str_glos_2,
                             str_glos_in_1,str_glos_in_2,
                             str_glos_out_1,str_glos_out_2):
    # sort them in growing order:
    i_g1 = copy(sorted(range(len(g_los1_in)), key=lambda k: g_los1_in[k]))
    i_g2 = copy(sorted(range(len(g_los2_in)), key=lambda k: g_los2_in[k]))

    g_los1_in      = np.array(g_los1_in)[i_g1]
    g_los1_out_med = np.array(g_los1_out_med)[i_g1]
    g_los1_out_std = np.array(g_los1_out_std)[i_g1]

    g_los2_in      = np.array(g_los2_in)[i_g2]
    g_los2_out_med = np.array(g_los2_out_med)[i_g2]
    g_los2_out_std = np.array(g_los2_out_std)[i_g2]
    
    ax = axes[0]
    ax.set_title(str_glos_1)
    ax.set_xlabel(str_glos_in_1)
    ax.set_ylabel(str_glos_out_1)
    ax.scatter(g_los1_in,g_los1_out_med,marker="o",c="k")
    ax.errorbar(g_los1_in,g_los1_out_med,yerr=g_los1_out_std,c="k",fmt="ko",ecolor="k",elinewidth=.8)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="grey")
    
    ax = axes[1]
    ax.set_title(str_glos_2)
    ax.set_xlabel(str_glos_in_2)
    ax.set_ylabel(str_glos_out_2)
    ax.scatter(g_los2_in,g_los2_out_med,marker="o",c="k")
    ax.errorbar(g_los2_in,g_los2_out_med,yerr=g_los2_out_std,c="k",fmt="ko",ecolor="k",elinewidth=.8)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="grey")
    return axes
    
def plot_los_outVsin_const(axes,g_los1_in,g_los2_in,
                             g_los1_out_med,g_los1_out_std,
                             g_los2_out_med,g_los2_out_std,
                             str_glos_1,str_glos_2,
                             str_glos_in_1,str_glos_in_2,
                             str_glos_out_1,str_glos_out_2): 
    
    x = np.arange(len(g_los1_in))

    ax = axes[0]
    
    #ax.set_title(str_glos_1)
    #ax.set_xlabel()
    ax.set_ylabel(str_glos_1)
    ax.get_xaxis().set_ticks([])
    ax.axhline(np.mean(g_los1_in),ls="--",color="grey",label=str_glos_in_1)
    ax.scatter(x,g_los1_out_med,marker="o",c="k",label=str_glos_out_1)
    ax.errorbar(x,g_los1_out_med,yerr=g_los1_out_std,c="k",fmt="ko",ecolor="k",elinewidth=.8)
    #ax.plot([0, 1], [0, 1], transform=ax.transAxes,ls="--",c="grey")
    ax.legend()
    ax = axes[1]
    #ax.set_title(str_glos_2)
    ax.set_ylabel(str_glos_2)
    ax.get_xaxis().set_ticks([])
    ax.set_xlabel("Lenses")
    ax.axhline(np.mean(g_los2_in),ls="--",color="grey",label=str_glos_in_2)
    ax.scatter(x,g_los2_out_med,marker="o",c="k",label=str_glos_out_2)
    ax.errorbar(x,g_los2_out_med,yerr=g_los2_out_std,c="k",fmt="ko",ecolor="k",elinewidth=.8)
    ax.legend()
    return axes
    
def plot_result_line(model,lens,axes,i_row,nrows,columns_ttl,_rnd=3,overlay_ellipticity=False):
    fig       = axes.flatten()[0].get_figure()
    lens_name = lens.name.replace("Sub_","")
    chnl_path = f'{lens.model_res_dir}/chain_list.dll'
    chain_list = load_whatever(chnl_path)
    sampler_type, mc_sample, param_mcmc, mc_logL  = chain_list[-1]
    param_mcmc = np.array(param_mcmc)
    if model == "noLOS":
        mc_sample,param_mcmc = _convert_shear2LOS(mc_sample,param_mcmc)
    elif model=="noLOS_g12":
        mc_sample,param_mcmc = _convert_polarshear2LOS(mc_sample,param_mcmc)

    full_chain = pd.DataFrame( np.array(mc_sample) , columns=param_mcmc)

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
    ax.set_ylabel(lens_name)
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
    if i_row==nrows-1:
        ax.set_xlabel(r"$\theta_E$")
    i_thetaE = np.where(param_mcmc=="theta_E_lens0")
    thetaE   = mc_sample.T[i_thetaE][0]
    plot_dist(ax, Chain(samples=full_chain, name=lens_name, shade=True, color='#2c7fb8', smooth=20, bins=10,
                                   shade_gradient = 0.4, linewidth=3.0), px="theta_E_lens0",)
    lbl_tE_meas = r"$\theta_{\rm{E}}=$"+str(np.round(np.median(thetaE),_rnd))+"+-"+str(np.round(np.std(thetaE),_rnd))+'"'
    ax.axvline(np.median(thetaE),ls="-.",color="k",label=lbl_tE_meas)
    lbl_tE_true = r"$\theta_{\rm{E, True}}=$"+short_SciNot(lens.thetaE.value)+'"'
    ax.axvline(lens.thetaE.value,ls="-",color="r",label=lbl_tE_true)
    ax.legend(loc="upper right")
    
    # Posterior gamma_ext
    ax = axes[i_row][4]
    if i_row ==0:
        ax.set_title(columns_ttl[4])
    plot_contour(ax, Chain(samples=full_chain, name=lens_name, shade=True, color=cool[1], shade_gradient = 0.8, linewidth=3.0), px="gamma1_los_lens1", py="gamma2_los_lens1")
    if overlay_ellipticity:
        plot_contour(ax, Chain(samples=full_chain, name=lens_name, shade=True, color=warm[1], shade_gradient = 0.8, linewidth=3.0), px="e1_lens0", py="e2_lens0")
        
    if overlay_ellipticity:
        column_g12e12 = ['gamma1_los_lens1', 'gamma2_los_lens1',"e1_lens0","e2_lens0"]
    else:
        column_g12e12 = ['gamma1_los_lens1', 'gamma2_los_lens1']
    g12e12 = full_chain.loc[:,column_g12e12]
    med_g12e12 = g12e12.median().array
    sig_g12e12 = g12e12.std().array
    for i in range(4):
        if i<2:
            col= warm[0]
        else:
            if not overlay_ellipticity:
                continue
            col = cool[1]
        _nm = column_g12e12[i].replace("_lens0","").replace("_lens1","")
        _nm = _nm.replace("gamma",r"$\gamma$")
        if "1" in _nm:
            _nm = _nm.replace("1","")
            _nm = _nm.replace("_los",r"$_{\rm{LOS}, 1}$")
        elif "2" in _nm:
            _nm = _nm.replace("2","")
            _nm = _nm.replace("_los",r"$_{\rm{LOS}, 2}$")
        
        if model=="noLOS":
           _nm = _nm.replace("LOS",r"Shear")
         
        
        lbl = _nm+f"={np.round(med_g12e12[i],_rnd)}+-{np.round(sig_g12e12[i],_rnd)}"
        col="k"
        if i%2==0:
            ax.axvline(med_g12e12[i],ls="-.",c=col,label=lbl)
        else:
            ax.axhline(med_g12e12[i],ls="-.",c=col,label=lbl)

    clmns = [r"$\gamma_{\rm{LOS},1}$",r"$\gamma_{\rm{LOS},2}$"]
    if "noLOS" in model:
        clmns = [r"$\gamma_{\rm{shear},1}$",r"$\gamma_{\rm{shear},2}$"]
        
    if i_row==nrows-1:
        if overlay_ellipticity:
            ax.set_xlabel(clmns[0]+r"/e$_1$")
        else:
            ax.set_xlabel(clmns[0])
    if nrows==1:
        ax.set_xlabel(clmns[0])
    if overlay_ellipticity:
        ax.set_ylabel(clmns[1]+r"/e$_2$")
    else:
        ax.set_ylabel(clmns[1])

    if model=="allLOS":
        kw_add_lns = kw_input['kw_add_lenses']
        kw_los     = kw_add_lns["kwargs_lens"][kw_add_lns["lens_model_list"].index("LOS")]
        gamma_los1_true = kw_los["gamma1_od"] + kw_los["gamma1_os"] - kw_los["gamma1_ds"]
        gamma_los2_true = kw_los["gamma2_od"] + kw_los["gamma2_os"] - kw_los["gamma2_ds"]
    elif "fitLOS" in model or "noLOS" in model:
        gamma_los1_true,gamma_los2_true = 0,0
    ax.axvline(gamma_los1_true,ls="-",label="Truth "+clmns[0]+f"= {np.round(gamma_los1_true,_rnd)}",c="r")
    ax.axhline(gamma_los2_true,ls="-",label="Truth "+clmns[1]+f"= {np.round(gamma_los2_true,_rnd)}",c="r")
        
    ax.legend(loc="upper left")
    #ax.legend()
    return axes
import warnings
warnings.filterwarnings("ignore")

name_models = ["noLOS","noLOS_g12","fitLOS","allLOS","fitLOS_fixedOD","fitLOS_fixedOD_fixedOmgaLos"]
def get_res_dir(model):
    if model=="noLOS":
        from nazgul.model_ext_shear import res_dir_base as res_dir
    elif model=="noLOS_g12":
        from nazgul.model_ext_shear_g12 import res_dir_base as res_dir
    elif model=="fitLOS":
        from nazgul.model_fitLOS import res_dir_base as res_dir
    elif model=="allLOS":
        from nazgul.model_allLOS import res_dir_base as res_dir
    elif model=="fitLOS_fixedOD":
        from nazgul.model_fitLOS_fixedOD import res_dir_base as res_dir
    elif model=="fitLOS_fixedOD_fixedOmgaLos":
        from nazgul.model_fitLOS_fixedOD_fixedOmegaLos import res_dir_base as res_dir
    else:
        if model in name_models:
            print("To implement") 
        raise RuntimeError(f"model {model} not known")
    return res_dir
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog=sys.argv[0],description="Plot Combined results for all the lens model of given run")
    parser.add_argument('-m','--model',type=str,
                        dest="model",
                        help=f"Name of type of model - accepted: {name_models}")
    
    parser.add_argument('-ove','--overlay_ellipticity', dest="overlay_ellipticity", 
                        default=False, action="store_true",
                        help=f"If true, overlay ellipticity posterior")
    args     = parser.parse_args()
    model    = args.model
    overlay_ellipticity = args.overlay_ellipticity

    # to which cifra significativa to round
    _rnd = 3
    res_dir = get_res_dir(model)
    nm_combined = f"{res_dir}/combined_result.pdf"    
        
    lenses_modelled = get_all_lens_models(res_dir)
    columns_ttl = ["Sim Image","Model","Norm. Resid.",r"P($\theta_E$|S.I.)"]
    if overlay_ellipticity:
        columns_ttl.append(r"P($\gamma_{\rm{LOS},1}$,$\gamma_{\rm{LOS},2}$|S.I.) + P(e$_1$,e$_2$|S.I.)")
    else:
        columns_ttl.append(r"P($\gamma_{\rm{LOS},1}$,$\gamma_{\rm{LOS},2}$|S.I.)")

    if "noLOS" in model:
        columns_ttl[-1] = columns_ttl[-1].replace("LOS","Shear")
    nrows  = len(lenses_modelled)
    ncols  = len(columns_ttl) # n* of wanted columns

    scl = 8
    print("DEBUG - skipping result line")
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(scl*ncols,scl*nrows))
    for i_row,lens in enumerate(lenses_modelled):
        plot_result_line(model,lens,axes,i_row,nrows,columns_ttl,\
                         _rnd=_rnd,overlay_ellipticity=overlay_ellipticity)
    plt.suptitle(get_model_title(model))
    plt.tight_layout()
    plt.savefig(nm_combined)
    print(f"Saved {nm_combined}")
    plt.close()
    """
    for i_row,lens in enumerate(lenses_modelled):
        nm_single = f"{lens.model_res_dir}/single_result.pdf"    

        fig, axes = plt.subplots(1, ncols, figsize=(scl*ncols,scl*1))
        plot_result_line(model,lens,np.array([axes]),0,1,columns_ttl,\
                         _rnd=_rnd,overlay_ellipticity=overlay_ellipticity)
        plt.suptitle(get_model_title(model))
        plt.tight_layout()
        plt.savefig(nm_single)
        print(f"Saved {nm_single}")
        plt.close()

    if model=="allLOS":
        fig,fig2 = plot_los_outVsin(lenses_modelled,_rnd=_rnd)
        _nm = "comp_glos_12_in_vs_out.pdf"
        nm = f"{res_dir}/{_nm}"
        fig.savefig(nm)
        print(f"Saved {nm}")
        _nm = "comp_glos_in_vs_out.pdf"
        nm = f"{res_dir}/{_nm}"
        fig2.savefig(nm)
        print(f"Saved {nm}")
        plt.close()
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
