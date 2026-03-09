from nazgul.particle_lenses import default_kwlens_part_AS  as kwlens_part_AS
from nazgul.mount_doom.generate_particle_lens_sub import SubLensPart
from nazgul.mount_doom.cracks_of_doom import pixel_num,min_thetaE
from nazgul.mount_doom.cracks_of_doom import source_model_list
from nazgul.mount_doom.cracks_of_doom import kwargs_band_sim,kw_prior_z_source_stnd
from nazgul.pathfinder import get_lens_highdir

import nazgul.mount_doom.cracks_of_doom as cod

verbose = True
from nazgul.particle_galaxy import PartGal


Gal    = PartGal(5,0,
             z=None,snap="20",    # redshift or snap
             M=None,Centre=None,
             reload=True)
lens = SubLensPart(Gal,
                 kwlens_part=kwlens_part_AS, # if PM or AS, and if so size of the core
                 pixel_num=pixel_num, # sim prms 
                 kw_prior_z_source = kw_prior_z_source_stnd, # could likelihood of z_source
                 min_thetaE = min_thetaE,
                 subdir="./",
                 reload=True # reload previous lens
                 ) 

lens.run()