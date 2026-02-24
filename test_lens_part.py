"""
Compare fit LOS effects with "a posteriori" implementation obtained from interpolation of alpha map
"""
import dill
import numpy as np

from lens_part import SinglePlaneLensPart

from particle_galaxy import PartGal
from python_tools.image_manipulation import plot_comp_two_images
from particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
from generate_particle_lens import get_extents
from generate_particle_lens import pixel_num,kw_prior_z_source_minimal
from lens_part_los import get_kw_los
    
if __name__ == "__main__":
    reload = True

    Gal    = PartGal(5,0,
                 z=None,snap="20",    # redshift or snap
                 M=None,Centre=None,
                 reload=reload)

    kw_add_lenses = {"lens_model_list":["LOS"],
                    "kwargs_lens":[]}
    # load first parameters from analosis, golden sample
    kw_los = get_kw_los()
    kw_add_lenses["kwargs_lens"] = [kw_los]

    # compute the LOS effect exactly (analytically)
    kwargs_lenspart = {"Galaxy":Gal,
                      "kwlens_part":kwlens_part_AS,
                      "kw_prior_z_source":kw_prior_z_source_minimal,
                      "pixel_num":pixel_num,
                      "reload":reload,
                      "savedir_sim":"test_sim_lens_LOS"}
    
    kwargs_lenspart_LOS = {"Galaxy":Gal,
                      "kwlens_part":kwlens_part_AS,
                      "kw_prior_z_source":kw_prior_z_source_minimal,
                      "pixel_num":pixel_num,
                      "kw_add_lenses":kw_add_lenses,
                      "reload":reload,
                      "savedir_sim":"test_sim_lens_LOS"}
    # no actual lens:
    nolens = SinglePlaneLensPart()
    print(nolens.alpha(0,0,{}))
    
    # only a SIS lens
    lensModelSIS = SinglePlaneLensPart(lens_model_list=["GAUSSIAN_POTENTIAL"])
    kwargs_lens = [
            {
                "amp": 1.0,
                "sigma_x": 2.0,
                "sigma_y": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ]
    output1, output2 = lensModelSIS.alpha(x=1.0, y=1.0, kwargs=kwargs_lens)
    assert output1 == -0.19470019576785122 / (8 * np.pi)
    assert output2 == -0.19470019576785122 / (8 * np.pi)

    # only our galaxy 
    lens_part = SinglePlaneLensPart(kwargs_lenspart=kwargs_lenspart)
    print(lens_part.alpha(0,0,{}))
    # +LOS:
    lens_part = SinglePlaneLensPart(kwargs_lenspart=kwargs_lenspart_LOS)
    print(lens_part.alpha(0,0,{}))
    # +Gaussian:
    lens_part = SinglePlaneLensPart(kwargs_lenspart=kwargs_lenspart_LOS,
                                    lens_model_list=["GAUSSIAN_POTENTIAL"])
    print(lens_part.alpha(0,0,kwargs=kwargs_lens))