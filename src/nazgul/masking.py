import numpy as np
from astropy.stats import sigma_clip
from scipy.ndimage import zoom,gaussian_filter

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
    if image is None:
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

def mask_max_dens(lens,image=None,rad=0.15):
    """
    mask the densest coord of the lens (~center)
    """
    if not image:
        image = lens.image_sim
    # by construction recentered around densest point
    cx,cy = lens.pixel_num/2.,lens.pixel_num/2.
    # for some reason this might be a bit off -> take the maximum density
    y,x    = np.where(lens.kappa_map==lens.kappa_map.max())
    #RA,DEC = lens.get_RADEC()
    #ra,dec = RA[0][x],DEC[:,0][y]
    
    mask = np.ones_like(image)
    r_mask_in = to_dimless(rad/lens.deltaPix)     #pixel 
    #mask = mask_in(cx,cy,r_mask_in,mask)
    mask = mask_in(x,y,r_mask_in,mask)
    return mask
    
def mask_center(lens,image=None,rad=0.15):
    """
    mask the centre of the lens
    """
    if image is None:
        image = lens.image_sim
    # by construction recentered around densest point
    cx,cy = lens.pixel_num/2.,lens.pixel_num/2.
    
    mask = np.ones_like(image)
    r_mask_in = to_dimless(rad/lens.deltaPix)     #pixel 
    mask = mask_in(cx,cy,r_mask_in,mask)
    return mask

def invert_mask(mask):
    inv_bool_mask = ~np.bool(mask)
    dtp_input     = np.array(mask).dtype
    inv_mask      = np.array(inv_bool_mask,dtype=dtp_input)
    return inv_mask
    
def resize_mask(mask,target_image):
    # assuming same edge size
    for im in mask,target_image:
        if im.shape[0]!=im.shape[1]:
            raise RuntimeError("The image and masks needs to be square")
            
    scaling = target_image.shape[0]/mask.shape[0]
    mask_rescaled = zoom(mask,scaling,order=3)
    mask_rescaled[np.where(mask_rescaled>=0.5)] = 1
    mask_rescaled[np.where(mask_rescaled<0.5)]  = 0
    return mask_rescaled
    
def mask_bright_center(lens,image=None,rad_pix=10):
    """
    re-centre the mask around the brightest pixel
    """
    if image is None:
        image = lens.image_sim
    mask_cent = mask_center(lens,image=image,rad=5*lens.deltaPix)
    # find brightest pixel within the mask
    masked_part = invert_mask(mask_cent)*image
    """
    # the following is not necessary since we are using the sim image
    # to define the mask (ie no noise)
    fwhm = 0.7
    fwhm_pix = to_dimless(fwhm)/to_dimless(lens.deltaPix)
    filt_masked = gaussian_filter(masked_part,sigma=fwhm_pix)
    ymax,xmax   = np.where(filt_masked==np.max(filt_masked))
    # consider the first one
    xmax = xmax[0]
    ymax = ymax[0]
    """
    ymax,xmax   = np.where(masked_part==np.max(masked_part))
    # consider the first one
    xmax = xmax[0]
    ymax = ymax[0]
    # mask around that one
    mask = np.ones_like(image)
    r_mask_in = rad_pix# to_dimless(rad/lens.deltaPix)     #pixel 
    mask = mask_in(xmax,ymax,r_mask_in,mask)
    return mask