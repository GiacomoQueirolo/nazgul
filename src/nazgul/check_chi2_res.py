import numpy as np
from glob import glob
from los_res_study import get_model_plot,load_kwargs_result
from Modelling.lib_models  import get_red_chi2
from pathlib import Path

chi2 = []
for g in glob("results/models/simNoShear/snap_27_*/kw_res.dll"):
    model_res_dir = Path(g).parent
    kwargs_result  = load_kwargs_result(model_res_dir)
    try:
        model_plot     = get_model_plot(model_res_dir,kwargs_result=kwargs_result)
        chi2.append(get_red_chi2(model_plot,verbose=False))
    except Exception as e:
        print(f"Failed {e}:\n{model_res_dir}")

print(np.array(chi2))
