# Script to sistematically create all available GL 
import numpy as np
from nazgul.mount_doom.generate_particle_lens_dom import wrapper_get_all_lens

all_lenses = wrapper_get_all_lens(kw_galpart={"min_z":.4,
                                       "max_z":0.51,
                                       "min_mass":1e12},
                           reload=False)

print("Stat: \n")
print("Lenses: "+str(len(all_lenses)))
print("Success!")
