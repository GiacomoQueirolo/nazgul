# to run to test generate_particle_lens
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from mpl_toolkits.axes_grid1 import make_axes_locatable

import stpsf
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.ObservationConfig.JWST import JWST

from nazgul.plot_PL import plot_all
from nazgul.particle_galaxy import PartGal
from python_tools.tools import to_dimless
from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
from nazgul.generate_particle_lens import wrapper_get_rnd_lens,get_extents,LoadLens,LensPart
from nazgul.generate_particle_lens import pixel_num,kw_prior_z_source_minimal

if __name__ == "__main__":

    #print("Loading specific gal for debugging")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18//Gn5SGn0.pkl")
    profiler = Profiler()
    profiler.start()

    #mod_LP = wrapper_get_rnd_lens(reload=False)
    Gal    = PartGal(5,0,
                 z=None,snap="20",    # redshift or snap
                 M=None,Centre=None,
                 reload=False)
    mod_LP = LensPart(Galaxy=Gal,kwlens_part=kwlens_part_AS,
                       kw_prior_z_source=kw_prior_z_source_minimal, 
                       pixel_num=pixel_num,reload=False,
                      savedir_sim="test_sim_lens_AMR")
    """
    mod_LP = LoadLens("/pbs/home/g/gqueirolo/EAGLE/sim_lens/RefL0025N0752/snap19_G12.0/test_sim_lens_AMR/Gn12SGn0_Npix200_PartAS.pkl")
    #sim_lens/RefL0025N0752/snap16_G8.0//test_sim_lens_AMR/G8SGn0_Npix200_PartAS.pkl")
    """
    mod_LP.run()
    profiler.stop()
    print(profiler.output_text(color=True,show_all=False))
    plot_all(mod_LP,skip_caustic=True)

    profiler.start()

    band_HST = HST(band='WFC3_F160W', psf_type="GAUSSIAN")
    #band = HST(band='WFC3_F160W', psf_type="PIXEL") #-> if pixel, we need to give kernel_point_source (and point_source_supersampling_factor etc)
    #band.obs["psf_type"] = "PIXEL"
    #del band.obs["seeing"]
    #band.obs["kernel_point_source"] = []
    SimObs_HST = mod_LP.get_SimObs(band_HST,kwargs_source_model=None)

    image_hst = mod_LP.sim_image(SimObs_HST)
    band_JWST = JWST(band='F444W', psf_type="GAUSSIAN")
    

    
    SimObs_jwst = mod_LP.get_SimObs(band_JWST,kwargs_source_model=None)
    image_jwst  = mod_LP.sim_image(SimObs_jwst)


    # more realistic PSF for JWST
    nrc = stpsf.NIRCam()
    nrc.filter =  'F444W'
    
    pssf = 4
    psf = nrc.calc_psf(oversample=pssf)
    
    psf_data = psf[2].data
    band_JWST = JWST(band=nrc.filter, psf_type="PIXEL")
    """kwb = band_JWST.kwargs_single_band()
    kwb["kernel_point_source"]               = psf_data
    kwb["point_source_supersampling_factor"] = pssf
    pixel_num   =  int(to_dimless(2*mod_LP.radius)/kwb["pixel_scale"])
    SimObs_JWST = SimAPI(numpix=pixel_num, 
                    kwargs_single_band=kwb,
                    kwargs_model=mod_LP.kwargs_source_model)
    """
    kwargs_psf_JWST = {"kernel_point_source":psf_data,
                  "point_source_supersampling_factor":pssf}
    SimObs_JWST = mod_LP.get_SimObs(band_JWST,kwargs_psf=kwargs_psf_JWST,
                    kwargs_source_model=mod_LP.kwargs_source_model)
    image_jwst_real = mod_LP.sim_image(SimObs_JWST)


    plt.close()
    fig, axis = plt.subplots(1,4,figsize=(19,7))
    kw_extents = get_extents(mod_LP.arcXkpc,mod_LP)
    extent_arcsec = kw_extents["extent_arcsec"]
    ax  = axis[0]
    im0 = ax.matshow(mod_LP.image_sim,origin='lower',extent=extent_arcsec,cmap="hot")
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.set_title(r"Original sim. Image")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    ax  = axis[1]
    im0 = ax.matshow(image_hst,origin='lower',extent=extent_arcsec,cmap="hot")
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.set_title(r"HST sim. Image ")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    ax  = axis[2]
    im0 = ax.matshow(image_jwst,origin='lower',extent=extent_arcsec,cmap="hot")
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.set_title(r"JWST sim. Image ")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    ax  = axis[3]
    im0 = ax.matshow(image_jwst_real,origin='lower',extent=extent_arcsec,cmap="hot")
    ax.set_xlabel("RA ['']")
    ax.set_ylabel("DEC ['']")
    ax.set_title(r"JWST sim. Image (real PSF) ")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    plt.suptitle(mod_LP.Gal.Name)
    nm = "tmp/comp_SimIm.png"
    plt.tight_layout()
    plt.savefig(nm)
    plt.close()
    print(f"Saving {nm}")
    profiler.stop()
