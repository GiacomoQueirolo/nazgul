"""
copied from lens_part_los
Idea: have a much more general approach
be a superstructure, which treats the galaxy as a lens model
where we can add or not more analytical lens profiles w. their kwargs
as well as los 
-> basically a new SinglePlane
"""

import numpy as np

from generate_particle_lens import LensPart
from project_gal import ProjectionError
from lens_part_los import LensPartLOS,bounds_error

# mass
from particle_galaxy import Gal2MXYZ,Gal2kwMXYZ
from project_gal import project_kw_parts


__author__ = ["giacomo_queirolo"]

from lenstronomy.LensModel.single_plane import SinglePlane
from lenstronomy.LensModel.LineOfSight.single_plane_los import SinglePlaneLOS


__all__ = ["SinglePlaneLensPart"]

# ugly but so far best way:
from lenstronomy.LensModel.profile_list_base import _SUPPORTED_MODELS
LOS_models = [lm for lm in _SUPPORTED_MODELS if lm[:3]=="LOS" ]

raise RuntimeError("This is wrongggg! the alpha e.g. is not correctly implemented for LOS")
class SinglePlaneLensPart():
    """This class is based on the 'SinglePlaneLOS' class, modified to handle lens-galaxies obtained from hydrodynamical particles
    -> not inheritance per se, rather a wrapper
    Overwritten functions:
    - init
    - potential
    - alpha
    - hessian
    - density
    - mass_3d
    - mass_2d
    """

    def __init__(
        self,
        lens_model_list=[], #does NOT contain GALAXY
        kwargs_lenspart=None,
        index_los=None,
        profile_kwargs_list=None,
        lens_redshift_list=None,
        z_source_convention=None,
        alpha_scaling=1,
        use_jax=False
    ):
        """
        Instance of SinglePlaneLensPart() based on the SinglePlaneLOS(), except:
        - argument kwargs_lenspart to initialise the particle galaxy class
        
        See SinglePlaneLOS() and SinglePlane() docstring for other documentation.
        """
        if lens_redshift_list is None:
            lens_redshift_list = [None] * len(lens_model_list)
        if profile_kwargs_list is None:
            profile_kwargs_list = [{} for _ in range(len(lens_model_list))]

        """
        Issue: now the LOS is defined both in the SinglePlane and the LensPart
        but in principle it could be implemented in only one of the two by mistake 
        -> ensure it's not the case

        similarly: the 
        """
        self.has_los = False
        if index_los is None:
            self.SP_instance = SinglePlane(lens_model_list=lens_model_list,
            profile_kwargs_list=profile_kwargs_list,
            lens_redshift_list=lens_redshift_list,
            z_source_convention=z_source_convention,
            alpha_scaling=alpha_scaling,
            use_jax=use_jax)
        else:
            self.has_los = True
            self.SP_instance = SinglePlaneLOS(lens_model_list=lens_model_list,
            index_los=index_los,
            profile_kwargs_list=profile_kwargs_list,
            lens_redshift_list=lens_redshift_list,
            z_source_convention=z_source_convention)
            
        # NB: It is important to run that init first, in order to create a
        # list_func for the entire model, before splitting it between a main
        # lens and the LOS corrections

        # Define a separate class for the LensPart
        # Depending if it has LOS effects or not 
        #->LensPartLOS is anyway a wrapper, it only augment LensPart, deals 
        if kwargs_lenspart is not None:
            #TODO: find a better way...
            if  "kw_add_lenses" in kwargs_lenspart.keys():
                lml = kwargs_lenspart["kw_add_lenses"]["lens_model_list"]
                if lml in LOS_models:
                    self.has_los = True
            self._lenspart = LensPartLOS(**kwargs_lenspart)
            try:
                self._lenspart.run()
            except ProjectionError as PE:
                print("The selected galaxy fails to be a lens")
            #Note: we assume the FoR to be centered on the _lenspart centre (i.e. Max Density coord)
        else:
            self._lenspart = None
    
    def __getattr__(self,name):
        return self.SP_instance.__getattribute__(name)

    def potential(self, x, y, kwargs, k=None):
        """Lensing potential"""
        potential_prof = self.SP_instance.potential(x,y,kwargs,k=k)
        potential_part = self._lenspart.interp_map(x,y,self._lenspart.psi)
        return potential_part + potential_prof

    def alpha(self, x, y, kwargs, k=None):
        """Displacement angle"""
        alpha_prof = self.SP_instance.alpha(x,y,kwargs,k=k)
        if self._lenspart is None:
            return alpha_prof
        if self.has_los:
            #Maybe the following is not the best, as it interpolate twice?
            _,kwargs_los = self.SP_instance.split_lens_los(kwargs)
            map_alpha_part_x,map_alpha_part_y = self.alpha_LOS_APost(kwargs_los)
        else:
            map_alpha_part_x,map_alpha_part_y = self._lenspart.alpha_map
        alpha_part_x = self._lenspart.interp_map(x,y,map_alpha_part_x)
        alpha_part_y = self._lenspart.interp_map(x,y,map_alpha_part_y)
        alpha_part   = np.array([alpha_part_x,alpha_part_y])
        alpha_part   = alpha_part.reshape(np.shape(alpha_prof))
        return np.sum([alpha_prof,alpha_part],axis=0)

    def hessian(self,x,y,kwargs,k=None):
        """hessian matrix"""
        hessian_prof = self.SP_instance.hessian(x,y,kwargs,k=k)
        if self._lenspart is None:
            return hessian_prof
        if self.has_los:
            #Maybe the following is not the best, as it interpolate twice?
            _,kwargs_los = self.SP_instance.split_lens_los(kwargs)
            f_xx,f_xy,f_yx,f_yy = self.hessian_LOS_APost(kwargs_los)        
        else:
            f_xx,f_xy,f_yx,f_yy = self._lenspart.hessian
        f_xx    = self._lenspart.interp_map(x,y,f_xx)    
        f_xy    = self._lenspart.interp_map(x,y,f_xy)    
        f_yx    = self._lenspart.interp_map(x,y,f_yx)    
        f_yy    = self._lenspart.interp_map(x,y,f_yy)
        hessian_part = f_xx,f_xy,f_yx,f_yy
        return np.sum([hessian_prof,hessian_part],axis=0)
            

    """
    Given that we overwrote alpha and the potential, it should call that, so this should be actually fine as it is 
    @bounds_error
    def fermat_potential(
        self, x_image, y_image, kwargs_lens, x_source=None, y_source=None, k=None
    ):
        #Calculates the Fermat Potential adding the LensPart to the analytical one

        
        fermat_potential_prof = self.SP_instance.fermat_potential(x_image,y_image,kwargs_lens,\
                                                         x_source=x_source,y_source=y_source,k=k)
    """
    

    def mass_3d(self, r, kwargs, bool_list=None):
        """Computes the mass within a 3d sphere of radius r *for the main lens only*

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass (in angular units, modulo epsilon_crit)
        """

        print("Note: The computation of the 3d mass ignores the LOS corrections.")

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)
        mass_3d = self._main_lens.mass_3d(r=r, kwargs=kwargs_main, bool_list=bool_list)
        if self._lenspart is None:
            return mass_3d
        m,x,y,z  = Gal2MXYZ(self._lenspart.Gal)
        arcXkpc  = self._lenspart.arcXkpc
        x_,y_,z_ = x*arcXkpc, y*arcXkpc, z*arcXkpc
        R        = np.linalg.norm([x.value,y.value,z.value])*x.unit
        R_arcsec = R*arcXkpc
        m_       = m[np.where(r<R_arcsec.value)]
        m3d      = np.sum(m_)
        m3d_kpc2 = m3d/self._lenspart.SigCrit
        m3d_arc2 = m3d_kpc2*(arcXkpc**2)
        mass_3d += m3d_arc2.value
        print("To test and verify")
        return mass_3d

    def mass_2d(self, r, kwargs, bool_list=None):
        """Computes the mass enclosed a projected (2d) radius r *for the main lens only*

        The mass definition is such that:

        .. math::
            \\alpha = mass_2d / r / \\pi

        with alpha is the deflection angle

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the lens model classes
        :param bool_list: list of bools that are part of the output
        :return: projected mass (in angular units, modulo epsilon_crit)
        """

        print("Note: The computation of the 2d mass ignores the LOS corrections.")

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)
        mass_2d = self._main_lens.mass_2d(r=r, kwargs=kwargs_main, bool_list=bool_list)
        if self._lenspart is None:
            return mass_2d
        kw_parts      = Gal2kwMXYZ(self.Gal) 
        kw_parts_proj = project_kw_parts(kw_parts=kw_parts,proj_index=self._lenspart.proj_index)
        x = kw_parts_proj["Xs"]
        y = kw_parts_proj["Ys"]
        m = kw_parts_proj["Ms"]

        arcXkpc  = self._lenspart.arcXkpc
        x_,y_    = x*arcXkpc, y*arcXkpc
        R        = np.hypot([x.value,y.value])*x.unit
        R_arcsec = R*arcXkpc
        m_       = m[np.where(r<R_arcsec.value)]
        m2d      = np.sum(m_)
        m2d_kpc2 = m2d/self._lenspart.SigCrit
        m2d_arc2 = m2d_kpc2*(arcXkpc**2)
        print("To test and verify")
        mass_2d += m2d_arc2.value
        return mass_2d

    def density(self, r, kwargs, bool_list=None):
        """3d mass density at radius r *for the main lens only* The integral in the LOS
        projection of this quantity results in the convergence quantity.

        :param r: radius (in angular units)
        :param kwargs: list of keyword arguments of lens model parameters matching the
            lens model classes
        :param bool_list: list of bools that are part of the output
        :return: mass density at radius r (in angular units, modulo epsilon_crit)
        """

        print("Note: The computation of the density ignores the LOS corrections.")

        kwargs_main, kwargs_los = self.split_lens_los(kwargs)
        density = self._main_lens.density(r=r, kwargs=kwargs_main, bool_list=bool_list)
        if self._lenspart is None:
            return density
        m,x,y,z = Gal2MXYZ(self._lenspart.Gal)
        arcXkpc =self._lenspart.arcXkpc
        x_,y_,z_ = x*arcXkpc, y*arcXkpc, z*arcXkpc
        R = np.linalg.norm([x.value,y.value,z.value])*x.unit
        R_arcsec = R*arcXkpc
        m_ = m[np.where(r<R_arcsec.value)]
        m3d = np.sum(m_)
        m3d_kpc2 = m3d/self._lenspart.SigCrit
        m3d_arc2 = m3d_kpc2*(arcXkpc**2)
        dens     = m3d_arc2.value/(np.pi*r*r)
        print("To test and verify")
        density += dens
        return density

