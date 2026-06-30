# Script to sistematically create all available GL 
import numpy as np
import warnings
from nazgul.mount_doom.generate_gal_lens import wrapper_get_all_lens
run_profiler = False
if run_profiler:
    warnings.warn("Running profiler might increase memory use and lead to a OOM kill")
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()

# Following SEAGLE_I
print("Following SEAGLE_I but now applied to COLIBRE")
all_lenses = wrapper_get_all_lens(kw_galpart={
    'simsuite':'COLIBRE',
    "sim":"L0025N0752",
    'subsim':'THERMAL_AGN_m5',
    "min_z":0.199,#0.099,#0.49,
    "max_z":0.201,#0.301,#0.51,
    "kw_criteria":{"min_vel_disp":120,
                    "min_hmr":1,
                    "min_mass_stars":1.76e10*.6777}
    },                
    reload=True,
    _test=False,
    _list_of_skippable_gals=[])#_list)

if run_profiler:
    profiler.stop()
    print(profiler.output_text(color=True,show_all=False))

print("Stat: \n")
print("Lenses: "+str(len(all_lenses)))
print("Success!")
