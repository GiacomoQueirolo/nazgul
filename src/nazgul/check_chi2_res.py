import numpy as np
from glob import glob
from Modelling.lib_models import load_kwargs_input,load_kwargs_result,get_model_plot, get_red_chi2
from pathlib import Path

res_dir = "results/models/simNoShear/"
chi2 = []
n_pix_used = []
for g in glob(f"{res_dir}/snap_*/kw_res.dll"):
    model_res_dir = Path(g).parent
    kwargs_result  = load_kwargs_result(model_res_dir)
    try:
        model_plot     = get_model_plot(model_res_dir,kwargs_result=kwargs_result)
        chi2.append(get_red_chi2(model_plot,verbose=False))
        # add a check for the correlation between chi2 and image size 
        kw_input = load_kwargs_input(model_res_dir)
        mask = kw_input["kwargs_likelihood"]["image_likelihood_mask_list"][0]
        n_pix_used.append(mask.sum())
        
    except Exception as e:
        print(f"Failed {e}:\n{model_res_dir}")

print(np.array(chi2))

import matplotlib.pyplot as plt

plt.scatter(chi2,n_pix_used,c="k")
plt.xlabel(r"$\chi^2$")
plt.ylabel(r"N pixel (not masked)")
print("Cropped")
plt.xlim(0,200)
plt.ylim(0,5000)

nm = f"{res_dir}/chi2VsNpix.png"
plt.savefig(nm)
print(f"Saved {nm}")