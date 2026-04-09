# Script to sistematically create all available GL 
import numpy as np
from nazgul.mount_doom.generate_gal_lens import wrapper_get_all_lens

from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
all_lenses = wrapper_get_all_lens(kw_galpart={"min_z":.09,
                                       "max_z":0.101,
                                       "min_mass":1e12},
                           reload=False,
                            _test=True)

profiler.stop()
print(profiler.output_text(color=True,show_all=False))

print("Stat: \n")
print("Lenses: "+str(len(all_lenses)))
print("Success!")
