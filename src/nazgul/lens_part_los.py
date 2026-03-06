"""
Add LOS effects
"""
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from lenstronomy.LensModel.LineOfSight.LOSModels.los import LOS

from python_tools.conversion import find_index

from generate_particle_lens import LensPart



kw_los_path = "/pbs/home/g/gqueirolo/analosis/analosis/results/datasets/golden_sample_input_kwargs.csv"
def get_kw_los(kw_los_path=kw_los_path,index=0):
    kw   = pd.read_csv(kw_los_path)
    los_cols = ['kappa_os', 'gamma1_os', 'gamma2_os', 'omega_os',
    'kappa_od', 'gamma1_od', 'gamma2_od', 'omega_od',
    'kappa_ds', 'gamma1_ds', 'gamma2_ds', 'omega_ds',
    'kappa_los', 'gamma1_los', 'gamma2_los', 'omega_los']
    los  = kw.loc[:, los_cols]
    list_los = los.to_dict('records')
    kw_los = list_los[index]
    return kw_los

# decorator:
def bounds_error(func):
    """Raise error if given coordinates are outside the bounds of the lenspart area"""
    def func_bounded(self,x,y,*args):
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        extents = self.kw_extents["extent_arcsec"]
        if np.any(x<extents[0]) or np.any(x>extents[1])  \
            or np.any(y<extents[2]) or np.any(y>extents[3]):
            raise RuntimeError("Input coordinates are outside allowed range")
        return func(self,x,y,*args)
    return func_bounded

class LensPartLOS(LensPart):
    """
    Child of LensPart
    Add LOS effects
    """
    @property
    def name(self):
        # define name and path of savefile
        # overwritten to allow for a different path
        # -> issue as it will 2x the space required  (not terrirble actually)
        #TODO find better solution
        return f"{self.Gal_name}_Npix{self.pixel_num}_Part{self.PartLens_name}_LOS"

    @property
    def los(self):
        #initialise LOS class 
        # used for the distort_vector function
        return LOS()
        
    
    @bounds_error
    def get_xy_indexes(self,x,y):
        ra,dec = self.get_RADEC()
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))    
        
        index_x    = find_index(x.ravel(),ra[0])
        index_y    = find_index(y.ravel(),dec[:,0])
        xy_indexes = np.stack([index_y,index_x],-1).T
        return xy_indexes
        
    @bounds_error
    def interp_map(self,x,y,map):
        """
        Interpolate a given map at the given coordinates
        """
        xy_indexes = self.get_xy_indexes(x,y)
        int_map    = map_coordinates(map,xy_indexes, order=3,mode="nearest")
        return int_map
        
    def alpha_LOS_APost(self,kwargs_los,alpha_map=None):
        """
        compute alpha_map with LOS effects ''a posteriori''
        by interpolating the alpha_map 
        """
        # if not given as input, reads the computed deflection map
        if alpha_map is None:
            # if not already, compute alpha_map
            alpha_map = self.alpha_map
            
        ra,dec = self.get_RADEC()
        alpha_map_ra  = alpha_map[0]
        alpha_map_dec = alpha_map[1]
        
        # follows the alpha function of lenstronomy.LensModel.LineOfSight.single_plane_los.SinglePlaneLOS
        
        # Angular position where the ray hits the deflector's plane
        ra_d,dec_d = self.los.distort_vector(
                ra,
                dec,
                kappa=kwargs_los["kappa_od"],
                omega=kwargs_los["omega_od"],
                gamma1=kwargs_los["gamma1_od"],
                gamma2=kwargs_los["gamma2_od"],
            )
        
        radec_d_indexes = get_xy_indexes(ra_d,dec_d) 

        # alpha(theta*A_od)
        alpha_ra_flat  = map_coordinates(alpha_map_ra,  radec_d_indexes, order=3,mode="nearest")
        alpha_dec_flat = map_coordinates(alpha_map_dec, radec_d_indexes, order=3,mode="nearest")
        alpha_ra       = alpha_ra_flat.reshape(alpha_map[0].shape)
        alpha_dec      = alpha_dec_flat.reshape(alpha_map[1].shape)

        # Correction due to the background convergence, shear and rotation
        # A_ds*alpha(theta*A_od)
        alpha_ra, alpha_dec = self.los.distort_vector(
            alpha_ra,
            alpha_dec,
            kappa=kwargs_los["kappa_ds"],
            omega=kwargs_los["omega_ds"],
            gamma1=kwargs_los["gamma1_ds"],
            gamma2=kwargs_los["gamma2_ds"],
        )

        # Perturbed position in the absence of the main lens
        # theta*A_os
        theta_ra_os, theta_dec_os = self.los.distort_vector(
            ra,
            dec,
            kappa=kwargs_los["kappa_os"],
            omega=kwargs_los["omega_os"],
            gamma1=kwargs_los["gamma1_os"],
            gamma2=kwargs_los["gamma2_os"],
        )
    
        # Complete displacement
        alpha_ra     += ra - theta_ra_os
        alpha_dec    += dec - theta_dec_os
        alpha_map_LOS = (alpha_ra,alpha_dec)
        return alpha_map_LOS

    def hessian_LOS_APost(self,kwargs_los,hessian=None):
        """
        compute hessian maps with LOS effects ''a posteriori''
        by interpolating the hessian maps 
        """
        # if not given as input, reads the computed deflection map
        if hessian is None:
            # if not already, compute alpha_map
            hessian = self.hessian
        ra,dec = self.get_RADEC()
        f_xx, f_xy, f_yx, f_yy = hessian
        # follows the hessian function of lenstronomy.LensModel.LineOfSight.single_plane_los.SinglePlaneLOS

        # Angular position where the ray hits the deflector's plane
        ra_d, dec_d = self.los.distort_vector(
            ra,
            dec,
            kappa=kwargs_los["kappa_od"],
            omega=kwargs_los["omega_od"],
            gamma1=kwargs_los["gamma1_od"],
            gamma2=kwargs_los["gamma2_od"],
        )

        # Hessian matrix of the main lens only
        radec_d_indexes = get_xy_indexes(ra_d,dec_d) 

        f_xx_flat = map_coordinates(f_xx, radec_d_indexes, order=3,mode="nearest")
        f_xy_flat = map_coordinates(f_xy, radec_d_indexes, order=3,mode="nearest")
        f_yx_flat = map_coordinates(f_yx, radec_d_indexes, order=3,mode="nearest")
        f_yy_flat = map_coordinates(f_yy, radec_d_indexes, order=3,mode="nearest")
        f_xx      = f_xx_flat.reshape(f_xx[0].shape)
        f_xy      = f_xy_flat.reshape(f_xx[0].shape)
        f_yx      = f_yx_flat.reshape(f_xx[0].shape)
        f_yy      = f_yy_flat.reshape(f_xx[0].shape)

        # Multiply on the left by (1 - Gamma_ds)
        f_xx, f_xy, f_yx, f_yy = self.los.left_multiply(
            f_xx,
            f_xy,
            f_yx,
            f_yy,
            kappa=kwargs_los["kappa_ds"],
            omega=kwargs_los["omega_ds"],
            gamma1=kwargs_los["gamma1_ds"],
            gamma2=kwargs_los["gamma2_ds"],
        )

        # Multiply on the right by (1 - Gamma_od)
        f_xx, f_xy, f_yx, f_yy = self.los.right_multiply(
            f_xx,
            f_xy,
            f_yx,
            f_yy,
            kappa=kwargs_los["kappa_od"],
            omega=kwargs_los["omega_od"],
            gamma1=kwargs_los["gamma1_od"],
            gamma2=kwargs_los["gamma2_od"],
        )

        # LOS contribution in the absence of the main lens
        f_xx += kwargs_los["kappa_os"] + kwargs_los["gamma1_os"]
        f_xy += kwargs_los["gamma2_os"] - kwargs_los["omega_os"]
        f_yx += kwargs_los["gamma2_os"] + kwargs_los["omega_os"]
        f_yy += kwargs_los["kappa_os"] - kwargs_los["gamma1_os"]

        return f_xx, f_xy, f_yx, f_yy
    
    
    def image_LOS_APost(self,kwargs_los,alpha_map=None):
        """
        Produce image with LOS distortion
        """
        alpha_map_LOS = self.alpha_map_LOS(kwargs_los,alpha_map)
        image_LOS     = self.get_lensed_image(alpha_map=alpha_map_LOS)
        return image_LOS
