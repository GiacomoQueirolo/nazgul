import numpy as np
import matplotlib.pyplot as plt
# Fitting ellipses - fast and easy 
# to have a first parameter estimates

def fit_ellipse_moments(image, threshold=None):
    img = np.asarray(image, dtype=float)

    # Optional: restrict to significant region
    if threshold is not None:
        mask = img > threshold
    else:
        mask = np.ones_like(img, dtype=bool)

    y, x = np.indices(img.shape)

    w = img * mask
    w_sum = w.sum()

    # --- centroid ---
    x0 = (x * w).sum() / w_sum
    y0 = (y * w).sum() / w_sum

    # --- centered coordinates ---
    dx = x - x0
    dy = y - y0

    # --- second moments ---
    Ixx = (w * dx * dx).sum() / w_sum
    Iyy = (w * dy * dy).sum() / w_sum
    Ixy = (w * dx * dy).sum() / w_sum

    # --- covariance matrix ---
    cov = np.array([[Ixx, Ixy],
                    [Ixy, Iyy]])

    # --- eigen decomposition ---
    eigvals, eigvecs = np.linalg.eigh(cov)

    # sort largest → smallest
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # --- ellipse parameters ---
    a = np.sqrt(eigvals[0])   # semi-major
    b = np.sqrt(eigvals[1])   # semi-minor

    # angle (radians)
    pa = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    return {
        "x0": x0,
        "y0": y0,
        "a": a,
        "b": b,
        "pa": pa
    }
    
def fit_ellipse_isocontour(image, level):
    mask = image >= level
    return fit_ellipse_moments(image * mask)
    
    

def plot_ellipse(ax, params, color="r"):
    t = np.linspace(0, 2*np.pi, 200)

    a, b = params["a"], params["b"]
    pa = params["pa"]
    x0, y0 = params["x0"], params["y0"]

    x = a * np.cos(t)
    y = b * np.sin(t)

    # rotate
    xr = x*np.cos(pa) - y*np.sin(pa)
    yr = x*np.sin(pa) + y*np.cos(pa)

    ax.plot(xr + x0, yr + y0, color=color, lw=2)

def plot_ellipse_isocontours(map,ellipses,nm="tmp/fit_ellipses.png"):
    fig, ax = plt.subplots()
    ax.imshow(map, origin="lower")
    for ell in ellipses:
        plot_ellipse(ax, ell)
    plt.savefig(nm)
    print(f"Saved {nm}")
    
# show params
    
def _get_prm(ellipses,_get_prm):
    prm = np.array([_get_prm(ell) for ell in ellipses])
    return prm

def get_eps(ellipses):
    def _get_eps(params):
        return 1-(params["b"]/params["a"])
    return _get_prm(ellipses,_get_eps)
    
def get_pa(ellipses):
    def _get_pa(ell):
        return ell["pa"]
    return _get_prm(ellipses,_get_pa)

def get_x0(ellipses):
    def _get_x(ell):
        return ell["x0"]
    return _get_prm(ellipses,_get_x)

def get_y0(ellipses):
    def _get_y(ell):
        return ell["y0"]
    return _get_prm(ellipses,_get_y)
def get_rad(ellipses):
    def _get_rad(ell):
        return np.sqrt(ell["a"]**2 + ell["b"]**2)
    return _get_prm(ellipses, _get_rad)
def get_sma(ellipses):
    def _get_a(ell):
        return ell["a"]
    return _get_prm(ellipses, _get_a)

prm_nms = "eps","x0","y0","pa","sma"

def get_prm(ellipses,prm_nm):
    if prm_nm=="eps":
        return get_eps(ellipses)
    elif prm_nm=="pa":
        return get_pa(ellipses)
    elif prm_nm=="x0":
            return get_x0(ellipses)
    elif prm_nm=="y0":
            return get_y0(ellipses)
    elif prm_nm=="rad":
            return get_rad(ellipses)
    elif prm_nm=="sma":
            return get_sma(ellipses)
    else:
        raise RuntimeError(f"Prm {prm_nm} not implemented")

def get_kw_prms(ellipses):
    kw_prms= {}
    for prm in prm_nms:
        kw_prms[prm] = get_prm(ellipses,prm)
    return kw_prms

# useful for isodensity fit:
def _get_initial_kwfit(kw_prms,sma_in=5):
    sma = kw_prms["sma"]
    # define accurate initial parameters
    d_sma = np.abs(sma-sma_in)
    indx = np.where(d_sma<1)

    kw_init = {}
    for prm_nm in prm_nms:
        kw_init[prm_nm] = np.nanmean(kw_prms[prm_nm][indx])
    kw_init["sma"] = sma_in
    return kw_init

def get_initial_kwfit(map,sma_in=5,levels=None,nlevels=60,plot=False):
    if levels is None:
        levels=np.linspace(1e-7,.99,nlevels)
    ellipses = [fit_ellipse_isocontour(map, l * map.max()) for l in levels]
    if plot:
        plot_ellipse_isocontours(ellipses)
    kw_prms = get_kw_prms(ellipses)
    kw_init = _get_initial_kwfit(kw_prms,sma_in=sma_in)
    return kw_init
    
if __name__=="__main__":
    raise NotImplementedError("Example of use - read, do not run")
    
    levels = np.linspace(1e-7,.99,60)
    
    ellipses = [fit_ellipse_isocontour(map, l * map.max()) for l in levels]
    plot_ellipse_isocontours(ellipses)

    prms = []
    for p in prm_nms:
        prms.append(get_prm(ellipses,p))

    for i,prm in enumerate(prms):
        axes[i].scatter(np.log10(sma),prm)
        axes[i].set_ylabel(prm_nms[i])
        axes[i].set_xlabel("log10(sma)")
    
    nm ="tmp/param_ellipses.png"
    plt.savefig(nm)
    print(f"Saved {nm}")
    
