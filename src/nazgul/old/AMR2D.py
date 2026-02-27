# My implementation of 2D AMR (adaptive mesh refinement)
# heavily helped by chatgpt

import numpy as np
from scipy.ndimage import zoom

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

class Cell:
    def __init__(self, x0, x1, y0, y1, particles_idx):
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.children = []
        self.particles_idx = particles_idx  # indices of particles inside
        self.mass = 0.0
        self.area = (x1 - x0) * (y1 - y0)
        self.density = None

    @property
    def is_leaf(self):
        return len(self.children) == 0


def compute_cell_mass(cell, m):
    """Sum particle masses in the cell."""
    cell.mass = np.sum(m[cell.particles_idx])
    cell.density = cell.mass / cell.area

def needs_refinement(cell, max_particles=200, min_size=0.01):
    """Return True if this cell should be subdivided."""
    if len(cell.particles_idx) > max_particles and (cell.x1-cell.x0) > min_size:
        return True
    return False
    
def subdivide(cell, x, y):
    xm = 0.5*(cell.x0 + cell.x1)
    ym = 0.5*(cell.y0 + cell.y1)

    # prepare the 4 child bounding boxes
    boxes = [
        (cell.x0, xm, cell.y0, ym),  # SW
        (xm, cell.x1, cell.y0, ym),  # SE
        (cell.x0, xm, ym, cell.y1),  # NW
        (xm, cell.x1, ym, cell.y1)   # NE
    ]

    children = []
    pts = cell.particles_idx

    # assign parent’s particles to children
    for (x0, x1, y0, y1) in boxes:
        mask = (x[pts] >= x0) & (x[pts] < x1) & (y[pts] >= y0) & (y[pts] < y1)
        child_idx = pts[mask]

        # avoid creating children identical to parent
        if child_idx.size == pts.size:
            continue

        children.append(Cell(x0, x1, y0, y1, child_idx))

    cell.children = children

def build_amr(cell, x, y, m, max_particles=200, min_size=0.01):
    compute_cell_mass(cell, m)

    if not needs_refinement(cell, max_particles, min_size):
        return

    subdivide(cell, x, y)

    for child in cell.children:
        build_amr(child, x, y, m, max_particles, min_size)
        
def get_leaves(cell):
    if cell.is_leaf:
        return [cell]
    leaves = []
    for c in cell.children:
        leaves.extend(get_leaves(c))
    return leaves

# Actual AMR constructor
def AMR_density(x, y, m, max_particles=200, min_size=0.01):
    # root domain
    x0, x1 = np.min(x), np.max(x)
    y0, y1 = np.min(y), np.max(y)

    root = Cell(x0, x1, y0, y1, np.arange(len(x)))

    build_amr(root, x, y, m, max_particles, min_size)

    leaves = get_leaves(root)

    return leaves



from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_amr_cells(cells):
    fig, ax = plt.subplots(figsize=(8,8))
    a,b= [],[]
    dns = [c.density for c in cells]
    vmax,vmin = np.max(dns),np.min(dns)
    cmap = plt.get_cmap("hot")
    norm = Normalize(vmin=vmin, vmax=vmax)
    for cell in cells:
        w = cell.x1 - cell.x0
        h = cell.y1 - cell.y0
        a.append(cell.x0)
        b.append(cell.y0)
        color = cmap(norm(cell.density))
        rect = patches.Rectangle((cell.x0, cell.y0), w, h,
                                 fill=True, linewidth=0.5,facecolor=color)
        ax.add_patch(rect)
    ax.set_xlim(np.min(a),np.max(a))
    ax.set_ylim(np.min(b),np.max(b))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Density")
    return fig,ax
    
def _test_AMR():
    x,y = np.random.normal(0,2,(2,10000))
    x0,y0 = np.random.normal(0,.4,(2,1000))
    x0 +=3
    y0 +=-2
    x = np.hstack([x,x0])
    y = np.hstack([y,y0])

    m   = np.abs(np.random.normal(100,3,(len(x))))

    cells = AMR_density(x,y,m,max_particles=10,min_size=0.0001)

    fig,ax = plot_amr_cells(cells)
    density = [c.density for c in cells]
    c_mode = cells[np.argmax(density,axis=0)]
    MD =(c_mode.x0+c_mode.x1)/2.,(c_mode.y0+c_mode.y1)/2. 
    print("MD:",MD)
    ax.axvline(MD[0])
    ax.axhline(MD[1])
    plt.savefig("tmp/test_AMR.png")