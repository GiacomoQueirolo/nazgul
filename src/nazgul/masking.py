from scipy.ndimage import gaussian_filter
from python_tools.tools import to_dimless
from python_tools.image_manipulation import mask_in, mask_out

def masking_thetaE(lens):
    # we want to mask everything apart the thetaE 
    tE  = to_dimless(lens.thetaE) #arcsec
    # rad = 2thetaE
    # let us mask r<thetaE/2 and r> thetaE*3/2
    r_mask_in = .5*tE/to_dimless(lens.deltaPix)     #pixel 
    r_mask_out = 3*tE/(2*to_dimless(lens.deltaPix)) #pixel
    # by construction recentered around densest point
    cx,cy = lens.pixel_num/2.,lens.pixel_num/2.
    mask = np.ones_like(lens.image_sim)
    mask = mask_in(cx,cy,r_mask_in,mask)
    mask = mask_out(cx,cy,r_mask_out,mask)
    return mask


def mask_SEAGLE(lens,image=None,threshold_scale=3,fwhm=.05,sig_clip=3):
    """Mask the image following SEAGLE approach:
        SEAGLE_I, Section 3.5:
        "convolving the noisy lensed images with a
        Gaussian with a FWHM of 0.25 arcsec to reduce the noise
        and smear the images to a slightly larger footprint. We then
        set a surface brightness threshold for the mask being a fac-
        tor of typically 2.5–5 below the original noise. Pixels above
        the threshold are set to one and all others to zero."
    """
    if not image:
        image = lens.image_sim
    fwhm_pix = to_dimless(fwhm)/to_dimless(lens.deltaPix)
    filt_img = gaussian_filter(image,sigma=fwhm_pix)
    # compute the threshold from the noise
    sgc = sigma_clip(image,sigma=sig_clip)        
    msk_sky = np.invert(sgc.mask)
    threshold = threshold_scale*np.median(msk_sky*image)
    
    mask     = np.ones_like(image)
    mask[np.where(filt_img<threshold)] = 0
    return mask

def mask_center(lens,image=None,rad=0.15):
    """
    mask the centre of the lens
    """
    if not image:
        image = lens.image_sim
    # by construction recentered around densest point
    cx,cy = lens.pixel_num/2.,lens.pixel_num/2.
    
    mask = np.ones_like(image)
    r_mask_in = to_dimless(rad/lens.deltaPix)     #pixel 
    mask = mask_in(cx,cy,r_mask_in,mask)
    return mask