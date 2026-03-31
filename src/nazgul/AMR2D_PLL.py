# updated from AMR2D - now parallelised and much faster and lighter for large particle numbers
# heavily helped by chatgpt

import numpy as np
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import numba
from time  import time
from numba import njit, prange

@njit
def split_indices(x, y, pts, x0, x1, y0, y1):
    xm = 0.5*(x0 + x1)
    ym = 0.5*(y0 + y1)
    
    # temporary lists
    c0 = []
    c1 = []
    c2 = []
    c3 = []
    
    for i in pts:
        xi = x[i]
        yi = y[i]

        if xi < xm:
            if yi < ym:
                c0.append(i)    # SW
            else:
                c2.append(i)    # NW
        else:
            if yi < ym:
                c1.append(i)    # SE
            else:
                c3.append(i)    # NE

    return c0, c1, c2, c3

def test_split_indices(x, y, m):
    """unit test function for split_indices"""
    pts = np.arange(len(x))

    c0,c1,c2,c3 = split_indices(x, y, pts, x.min(), x.max(), y.min(), y.max())
    
    total  = len(c0) + len(c1) + len(c2) + len(c3)
    unique = len(set(np.concatenate((c0,c1,c2,c3))))

    print("Total:", total)
    print("Unique:", unique)
    print("Expected:", len(pts))
    assert total == unique
    assert len(pts)==total
    


@njit
def compute_mass(pts, m):
    acc = 0.0
    for i in pts:
        acc += m[i]
    return acc

@njit
def needs_refinement(npts, area, density, pmax, min_area, dens_thresh):
    if npts > pmax and area > min_area:
        return True
    #if density > dens_thresh and size > min_area:
    #    return True
    return False

def build_AMR(x, y, m, 
              x0, x1, y0, y1, 
              max_particles=200, 
              min_area=0.01,
              dens_thresh=0.0):

    # stack of pending cells: (x0,x1,y0,y1,pts)
    stack = [(x0, x1, y0, y1, np.arange(x.size))]

    cells = []

    while stack:
        x0, x1, y0, y1, pts = stack.pop()

        sizex = x1 - x0
        sizey = y1 - y0
        area  = sizex * sizey
        mass  = compute_mass(pts, m)
        density = mass / area
        
        if needs_refinement(len(pts), area, density,
                            max_particles, min_area, dens_thresh):

            c0,c1,c2,c3 = split_indices(x, y, pts, x0, x1, y0, y1)
            for i, child in enumerate((c0, c1, c2, c3)):
                #if len(child) > 0 and len(child) < len(pts):
                if len(child) > 0 and (x1 - x0)*(y1 - y0) > min_area:
                    xm = 0.5*(x0+x1)
                    ym = 0.5*(y0+y1)
            
                    if i == 0:      # SW
                        stack.append((x0, xm, y0, ym, np.array(child)))
                    elif i == 1:    # SE
                        stack.append((xm, x1, y0, ym, np.array(child)))
                    elif i == 2:    # NW
                        stack.append((x0, xm, ym, y1, np.array(child)))
                    else:           # NE
                        stack.append((xm, x1, ym, y1, np.array(child)))
        else:
            # store cell
            # ignore particle ID -> keep it to verify uniqueness of binning
            cells.append([x0, x1, y0, y1, pts, mass, density])
            #cells.append([x0, x1, y0, y1, mass, density])

    return cells    
    
def validate_no_duplicates(cells, N,verbose=False):
     seen = np.zeros(N, dtype=np.int32)
     for (_,_,_,_,pts,_,_) in cells:
         seen[pts] += 1
 
     if np.all(seen==1):
         if verbose:
             print("OK: No duplicates, each particle appears once.")
     else:
         assert("ERROR: Some particles appear multiple times or not at all.")
    
def AMR_density_PLL(x, y, m, max_particles=300, min_area=None,dens_thresh=None,Sigma_crit=None):
    # numba strips units; if present, store them and add them a posteriori
    units = False
    try:
        units = True
        space_unit = x.unit
        x = x.value
        y = y.value
        min_area = min_area.value
    except:
        space_unit = 1
    try:
        units = True
        mass_unit = m.unit
        m = m.value
    except:
        mass_unit = 1 
    
    # Domain
    x0, x1 = np.min(x), np.max(x)
    y0, y1 = np.min(y), np.max(y)
    if min_area is None:
        domain = (x1-x0)*(y1-y0)
        min_area = (domain_size / 300)
    if dens_thresh is None:
        if Sigma_crit is None:
            raise RuntimeError("Provide either density threshold dens_thresh or critical density Sigma_crit")
        dens_thresh = 0.1*Sigma_crit
    cells = build_AMR(x, y, m,
                      x0,x1,y0,y1,
                      max_particles=max_particles,
                      min_area=min_area,dens_thresh=dens_thresh)
    if units:
        ucells = []
        for c in cells:
            uc = c
            if space_unit!=1:
                uc[:4]*=space_unit      #coords
                uc[-1]/=space_unit**2  #density
            if mass_unit!=1:
                uc[-2]*=mass_unit
                uc[-1]*=mass_unit      #mass and density
            ucells.append(uc)
        cells = ucells
    validate_no_duplicates(cells,len(m))
    # if all particle are accounted for, we don't need particle ID
    cells = [c[:4] + c[5:] for c in cells]
    # cells: x0,x1,y0,y1,m,density
    return cells


def plot_AMR_cells(kw_2Ddens):
    fig, ax = plt.subplots(figsize=(8,8))
    a,b= [],[]
    
    xc,yc = kw_2Ddens["MD_coords"] #kpc
    cells = kw_2Ddens["AMR_cells"]
    # to speed up the code I need to vectorise it -
    # but then I need to ingore the units

    x0,x1,y0,y1,mass,dns  = np.array([[cc.value for cc in c] for c in cells]).T
    x0_unit,x1_unit,y0_unit,y1_unit,mass_unit,dns_unit  = [c.unit for c in cells[0]]
    # verify that the units are consistent
    assert x0_unit==x1_unit
    assert x0_unit==y0_unit
    assert x0_unit==y1_unit
    assert x0_unit==xc.unit
    assert x0_unit==yc.unit
    
    length_unit = x0_unit
    x0 *=length_unit
    x1 *=length_unit
    y0 *=length_unit
    y1 *=length_unit
    mass *=mass_unit
    dns  *=dns_unit
    
    vmax,vmin = np.max(dns.value),np.min(dns.value)
    cmap = plt.get_cmap("hot")
    norm = Normalize(vmin=vmin, vmax=vmax)
    def add_patch_pll(i):
        ax.add_patch(patches.Rectangle((x0[i].value,y0[i].value),x1[i].value-x0[i].value,y1[i].value-y0[i].value,fill=True,linewidth=0.5,facecolor=cmap(norm(dns[i].value))))
        
    with Pool(cpu_count()) as pool:
        _ = pool.map(add_patch_pll, np.arange(len(cells)))

    #_ = [ax.add_patch(patches.Rectangle((x0[i].value,y0[i].value),x1[i].value-x0[i].value,y1[i].value-y0[i].value,fill=True,linewidth=0.5,facecolor=cmap(norm(dns[i].value)))) for i in range(len(cells))]

    

    patches_list = [
        Rectangle(
            (x0[i].value, y0[i].value),
            x1[i].value - x0[i].value,
            y1[i].value - y0[i].value)
        for i in range(len(cells))
    ]

    colors = cmap(norm([d.value for d in dns]))

    pc = PatchCollection(
        patches_list,
        facecolor=colors,
        linewidth=0.5,
        fill=True
    )

    ax.add_collection(pc)
    ax.set_xlim(np.min(x0.value),np.max(x0.value))
    ax.set_ylim(np.min(y0.value),np.max(y0.value))
    ax.set_aspect("equal")
    ax.set_xlabel("x ["+str(x0.unit)+"]")
    ax.set_ylabel("y ["+str(y0.unit)+"]")
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(f"Density [{dns_unit}]")
    return fig,ax
    
def _test_AMR():
    N = int(1e6)
    M = int(N/10)
    t0 = time()
    
    x,y = np.random.normal(0,2,(2,N))
    x0,y0 = np.random.normal(0,.4,(2,M))
    x0 +=3
    y0 +=-2
    x = np.hstack([x,x0])
    y = np.hstack([y,y0])
    m   = np.abs(np.random.normal(100,3,(len(x))))
    
    x0,x1 = np.min(x),np.max(x)
    y0,y1 = np.min(y),np.max(y)
    
    cells = AMR_density_pll(x, y, m,dens_thresh=0) #, x0, x1, y0, y1)
    print(len(cells))
    t1 = time()
    print(t1-t0)
    fig,ax=plot_AMR_cells(cells)
    nm_test = "tmp/AMR_pll_test.pdf"
    plt.savefig(nm_test)
    print(f"Saving {nm_test}")
    