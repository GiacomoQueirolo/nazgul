"""
Study the statistic of isodensity contours of available lenses
"""
#import glob
import numpy as np
import matplotlib.pyplot as plt

from python_tools.tools import to_dimless

from nazgul.pathfinder import std_sim
from nazgul.stat_lenses import get_all_gallens_paths
from nazgul.mount_doom.cracks_of_doom import LoadLens
from nazgul.mount_doom.lens_system import LensSystem

from nazgul.isodens import fit_isodens,plot_isodens

if __name__=="__main__":
    gal_lenses_path = get_all_gallens_paths()
    gamma_distr = [] 
    f_ellipt_distr = []
    DPA_distr = []
    f_bxdk_distr = [] # boxydiskyness, b4
    drift_x,drift_y = [],[]
    fig,axis = plt.subplots(6,2,figsize=(10,23))
    for i,gal_lens_path in enumerate(gal_lenses_path):        
        lens = LoadLens(gal_lens_path)
        if lens is False:
            continue
        try:
            kw_res  = fit_isodens(lens)
        except Exception as e:
            print(f"Lens {gal_lens_path} failed:\n{e}")
            print("skipping")
            continue
        logr    = kw_res["loglogfit"]["fitx"]
        r       = 10**np.array(logr) #kpc
        theta   = r*to_dimless(gal_lens.arcXkpc) #arcsec
        th_nrm  = theta/to_dimless(gal_lens.thetaE) #dimless
        i_thetaE = np.argmin(np.abs(th_nrm-1))
        isolist = kw_res["isodens"]["isolist"]
    
        y             = np.log10(isolist.intens[1:])
        gamma_der     = -np.gradient(y,logr)
        gamma_fit_fix = -kw_res["loglogfit"]["popt_log"][1]
        gamma_distr.append(gamma_fit_fix)

        eps    = isolist.eps[1:]
        eps_i  = eps[i_thetaE]
        f_eps  = eps/eps_i
        pa     = isolist.pa[1:]/np.pi*180. # deg
        # correct for pointing angle around ~0
        if np.any(np.abs(np.diff(pa))>135):
            pa[np.where(pa>135)] -=180
        pa_i = pa[i_thetaE]
        Dpa  = pa-pa_i

        geom  = kw_res["isodens"]["geom"]
        dx    = (isolist.x0[1:]-geom.x0) #pix
        dxarc = dx*to_dimless(gal_lens.deltaPix) #arcsec
        x0    = dxarc/to_dimless(gal_lens.thetaE) # dimless
        
        dy    = (isolist.y0[1:]-geom.y0) #pix
        y0    = to_dimless(dy*gal_lens.deltaPix/gal_lens.thetaE) #dimless

        bxdk   = isolist.b4[1:] # boxy-diskyness
        f_bxdk = bxdk/bxdk[i_thetaE]
        
        f_ellipt_distr.append(np.median(f_eps))
        DPA_distr.append(np.median(Dpa))
        f_bxdk_distr.append(np.median(f_bxdk))
        drift_x.append(np.median(x0))
        drift_y.append(np.median(y0))
        
        ax = axis[0][0]
        ax.plot(logr,gamma_der,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'log10(Semimajor axis [kpc])')
            ax.set_ylabel(r"$\gamma$ []")
            ax.set_title(r"Fit $\gamma$(r)")
        ax = axis[1][0]
        ax.plot(th_nrm,f_eps,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
            ax.set_ylabel(r"$\epsilon/\epsilon(\theta_E)$ []")
            ax.set_title(r"Ellipticity $\epsilon$ normalised for $\epsilon(\theta_E$)")
        ax = axis[2][0]
        ax.plot(th_nrm,Dpa,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
            ax.set_ylabel(r"P.A.-P.A.($\theta_E$) [$^o$]")
            ax.set_title(r"Pointing Angle P.A. rescaled by P.A.($\theta_E$)")
        
        ax = axis[3][0]
        ax.plot(th_nrm,x0,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
            ax.set_ylabel(r'(X0-Xcnt)/$\theta_E$ []')
            ax.set_title(r"Drift$_x$ = (Center$_x$-Cnt[0]$_x$)/$\theta_E$)")
        ax = axis[4][0]
        ax.plot(th_nrm,y0,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
            ax.set_ylabel(r'(Y0-Ycnt)/$\theta_E$ []')
            ax.set_title(r"Drift$_y$ = (Center$_y$-Cnt[0]$_y$)/$\theta_E$)")
        
        ax = axis[5][0]
        ax.plot(th_nrm,f_bxdk,alpha=.3,color="grey",ls="-")
        if i==0:
            ax.set_xlabel(r'Semimajor axis in arcsec /$\theta_E$ []')
            ax.set_ylabel(r"b4/b4($\theta_E$) []")
            ax.set_title(r"Boxy-diskiness b4 normalised for b4($\theta_E$)")

    
    # define limits to ignore outliers
    
    n_bins = 30
    ax = axis[0][1]
    ax.hist(gamma_distr,bins=n_bins)
    med_gamma = np.median(gamma_distr)
    ax.axvline(med_gamma,ls="--",c="r",label=r"median($\gamma$)="+str(np.round(med_gamma,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"$\gamma$(r)")
    ax.set_title(r"Distr. fit $\gamma$")
    ax.legend()

    ax = axis[1][1]
    ax.hist(f_ellipt_distr,bins=n_bins)
    med_ell = np.median(f_ellipt_distr)
    ax.axvline(med_ell,ls="--",c="r",label=r"median($\epsilon/\epsilon(\theta_E)$)="+str(np.round(med_ell,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"$\epsilon/\epsilon(\theta_E)$")
    ax.set_title(r"Distr. ellipticity $/\epsilon(\theta_E)$ ")
    ax.legend()

    ax = axis[2][1]
    ax.hist(DPA_distr,bins=n_bins)
    med_dpa = np.median(DPA_distr)
    ax.axvline(med_dpa,ls="--",c="r",label=r"median(D P.A.)="+str(np.round(med_dpa,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"P.A. - P.A.($\theta_E$)")
    ax.set_title(r"Distr. Pointing Angle - PA$(\theta_E)$")
    ax.legend()

    ax = axis[3][1]
    ax.hist(drift_x,bins=n_bins)
    med_dx = np.median(drift_x)
    ax.axvline(med_dx,ls="--",c="r",label=r"median(Drift x)="+str(np.round(med_dx,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"Drift x")
    ax.set_title(r"Distr. Drift x")
    ax.legend()

    ax = axis[4][1]
    ax.hist(drift_y,bins=n_bins)
    med_dy = np.median(drift_y)
    ax.axvline(med_dy,ls="--",c="r",label=r"median(Drift y)="+str(np.round(med_dy,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"Drift y")
    ax.set_title(r"Distr. Drift y")
    ax.legend()

    ax = axis[5][1]

    ax.hist(f_bxdk_distr,bins=n_bins)
    med_fb4 = np.median(f_bxdk_distr)
    ax.axvline(med_fb4,ls="--",c="r",label=r"median(b4/b4$(\theta_E)$)="+str(np.round(med_fb4,2))+" ["+str(len(gamma_distr))+" lenses]")
    ax.set_xlabel(r"b4/b4$(\theta_E)$")
    ax.set_title(r"Distr. Boxy-Diskyness parameter b4 /b4$(\theta_E)$")
    
    ax.legend()
    plt.tight_layout()
    nm = "tmp/distr_isoparams.png"
    print(f"Saving {nm}")
    plt.savefig(nm)
    