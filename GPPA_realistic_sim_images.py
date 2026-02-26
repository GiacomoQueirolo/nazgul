# copied and adapted from test_generate_particle_lens.py
from python_tools.get_res import LoadClass
from python_tools.read_fits import load_fits, load_fitshead,get_transf_matrix
from python_tools.conversion import get_pixscale
from python_tools.tools import to_dimless
from generate_particle_lens import LensPart,LoadLens
from generate_particle_lens import kwlens_part_AS,z_source_max,pixel_num
from project_gal import ProjectionError
from pyinstrument import Profiler

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from lenstronomy.ImSim.Numerics.convolution import PixelKernelConvolution
import lenstronomy.Util.image_util as image_util
from mpl_toolkits.axes_grid1 import make_axes_locatable

from generate_particle_lens import  wrapper_get_rnd_lens

def create_realistic_image(lens,
                           exp_time=1500,  #sec. (~<exp_time> for J1433 opt. HST obs )
                           bckg_rms=0.006, #counts (~ from f814w for J1433) 
                           psf_fits_path = "tmp/psf_0005_f814w_corr_interp.fits",
                           err_psf_fits_path = "tmp/e.psf_0001_f814w_corr.fits",
                           supersampling_factor=5,
                          do_plot=True):
    psf_supersampled      = load_fits(psf_fits_path)
    err_psf               = load_fits(err_psf_fits_path)
    head_psf_supersampled = load_fitshead(psf_fits_path)

    
    # PIXEL RESOLUTION #
    ####################
    #Important: the following is NOT corrected by the supersampling factor - good here,but to keep in mind
    HST_deltapix      = get_pixscale(psf_fits_path) # 0.04 #''/pix # F814W 
    if lens.deltaPix.value/HST_deltapix >1:
        raise RuntimeError("We got a problem - simulated resolution lower than aimed resolution, we have to reconsider the simulation")
        # we have to consider the simapi for these images
    sim_img_resampled = zoom(lens.image_sim,  lens.deltaPix.value/HST_deltapix)

    # PSF CONV #
    ############
    # we use archival PSF for convolution

    # first supersample it:
    sim_img_supersampled = zoom(sim_img_resampled, supersampling_factor) 

    # convolve it:
    PKC_ss = PixelKernelConvolution(psf_supersampled)
    sim_img_ss_conv =  PKC_ss.convolution2d(sim_img_supersampled)

    # downsample it back:
    sim_img_conv = zoom(sim_img_ss_conv,1/supersampling_factor) 

    # NOISE #
    #########
    # follow lenstronomy method, already used in GPPA in function update_kwargs_data_joint
    if lens.exp_time is None:
        lens.exp_time = exp_time
    if lens.bckg_rms is None:
        lens.bckg_rms = bckg_rms
    # but this all depends on the values of the image - are those flux, counts, counts/sec or what?
    
    poisson = image_util.add_poisson(sim_img_conv,    exp_time=lens.exp_time)
    bkg     = image_util.add_background(sim_img_conv, sigma_bkd=lens.bckg_rms)
    
    sim_img_conv_noised = sim_img_conv + poisson + bkg

    #other realisation of the bkg for the error map
    bkg_noise = image_util.add_background(sim_img_conv, sigma_bkd=lens.bckg_rms)
    noise_map = np.hypot(poisson,bkg_noise)
    if do_plot:
        from generate_particle_lens import get_extents
        kw_extents = get_extents(lens.arcXkpc,lens)
        extent_arcsec = kw_extents["extent_arcsec"]
        kw_plot = {"cmap":"gist_heat","origin":"lower","extent":extent_arcsec}
        fig, axss = plt.subplots(2,3, figsize=(15, 10))
        
        axs = axss[0]
        ims = []
        plt.suptitle("Lens name: "+lens.name)
        axs[0].set_title("Simulated image")
        ims.append(axs[0].imshow(np.log10(lens.image_sim),**kw_plot))
        axs[1].set_title("+HST pixel scale")
        ims.append(axs[1].imshow(np.log10(sim_img_resampled),**kw_plot))
        axs[2].set_title("+PSF convolution")
        ims.append(axs[2].imshow(np.log10(sim_img_conv),**kw_plot))
        axs = axss[1]
        axs[0].set_title("+Noise")
        ims.append(axs[0].imshow(np.log10(sim_img_conv_noised),**kw_plot))
        axs[1].set_title("Error map")
        ims.append(axs[1].imshow(np.log10(noise_map),**kw_plot))
        axs[2].set_title("PSF")
        ims.append(axs[2].imshow(np.log10(psf_supersampled),**kw_plot))
        i=0
        for axs in axss:
            for ax in axs:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(ims[i], cax=cax, orientation='vertical')
                i+=1
    
        nm = "tmp/image"+lens.name+".pdf"
        print(f"Saving {nm}")
        plt.tight_layout()
        plt.savefig(nm)
        plt.close("all")
    # following Lenstronomy naming convention
    kw_data = {"kernel_point_source":psf_supersampled,
               "point_source_supersampling_factor":supersampling_factor,
               "psf_error_map":err_psf, #non-supersampled
               "image_data":sim_img_conv_noised,
               "noise_map":noise_map,
               "exp_time":exp_time,
               "bckg_rms":bckg_rms,
               "deltaPix":HST_deltapix}
    return kw_data
    

def lnstr_kw_psf(kw_data):
    kwargs_psf = {'psf_type': 'PIXEL',
          'kernel_point_source':kw_data["kernel_point_source"],
          'point_source_supersampling_factor': kw_data["point_source_supersampling_factor"],
          'psf_variance_map':kw_data["psf_error_map"]**2}
    return kwargs_psf

from python_tools.conversion import pixscale2transfmat,_xy2radec
def lnstr_kw_data(kw_data):
    trsnf_matrx = pixscale2transfmat(kw_data["deltaPix"])
    pix0 = 0,0 # pix0
    CV0  = 0,0 # critical value -> set it to 0, as this is the 0 point
    #Critical pixel: we could set it to 0 or the center of the image (stnd.)
    CP0  = (len(kw_data["image_data"][0])-1)/2.,(len(kw_data["image_data"][1])-1)/2.
    ra0,dec0 =_xy2radec(*pix0,*CV0,*CP0,trsnf_matrx) 
    kwargs_data = {'background_rms': kw_data["bckg_rms"],
                  "exposure_time":   kw_data["exp_time"],
                  "ra_at_xy_0":      ra0,
                  "dec_at_xy_0":     dec0,
                  "transform_pix2angle": trsnf_matrx,
                  "image_data":  kw_data["image_data"]}
    # equivalent of doing data_configure_simple(len(kw_data["image_data"][0]),\
    # kw_data["deltaPix"],exposure_time=kw_data["exp_time"],background_rms=kw_data["bckg_rms"])
    return kwargs_data

    
if __name__ == "__main__":
    #print("Loading specific gal for debugging")
    #Gal = LoadClass("/pbs/home/g/gqueirolo/EAGLE/data/RefL0025N0752//Gals/snap_18//Gn5SGn0.pkl")
    profiler = Profiler()
    profiler.start()
    mod_LP = wrapper_get_rnd_lens()
    profiler.stop()
    print(profiler.output_text(color=True,show_all=False))
    plot_all(mod_LP,skip_caustic=True)
    kw_data = create_realistic_image(mod_LP)
    
    