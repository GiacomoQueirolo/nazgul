"""
plot_image_pairs.py
--------------------
Plot pairs of images (img1, img2) for n systems into a memory-efficient
multi-page PDF, with 2 columns x rows_per_page rows per page.

Layout per page:
    | name_A  img1_A  img2_A | name_B  img1_B  img2_B |
    | name_C  img1_C  img2_C | name_D  img1_D  img2_D |
    ...

Usage:
    from plot_image_pairs import plot_image_pairs_pdf
    plot_image_pairs_pdf(images1, images2, names, output_pdf="out.pdf")
"""

import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _limits(img):

    # copy from lenstronomy/Plots/model_band_plot.py
    lo = max(np.nanmin(img), -5)
    hi = min(np.nanmax(img), 10)

    return lo, hi

def plot_image_pairs_pdf(
    images1,
    images2,
    names,
    extents=None,
    output_pdf="image_pairs.pdf",
    rows_per_page=5,
    img_size_inch=3.0,
    cmap1="gist_heat",
    cmap2="bwr",
    label1="Image 1",
    label2="Image 2",
    label_cl1 = "flux",
    label_cl2 = "flux",
    #vmin1=None, vmax1=None,
    #vmin2=None, vmax2=None,
    log_scale1=True,
    log_scale2=False,
    limits=None,
    dpi=100,
    verbose=True,
):
    """
    Parameters
    ----------
    images1, images2 : list of 2D np.ndarray
        The two images per system. Must be the same length as names.
    names : list of str
        System names, one per entry.
    output_pdf : str
        Output PDF path.
    rows_per_page : int
        Number of system rows per page. Each page has 2 columns of systems,
        so rows_per_page=5 → 10 systems per page.
    img_size_inch : float
        Size of each image panel in inches.
    cmap1, cmap2 : str
        Colormaps for image 1 and image 2.
    label1, label2 : str
        Column header labels.
    vmin1, vmax1, vmin2, vmax2 : float or None
        Global colour limits. If None, computed per-image.
    log_scale1, log_scale2 : bool
        If True, apply log10 to the image before plotting.
    dpi : int
        Resolution of saved figures.
    verbose : bool
        Print progress.
    """
    assert len(images1) == len(images2) == len(names), \
        "images1, images2, names must all have the same length."
    
    n = len(names)
    assert n>0, "No images given"
    systems_per_page = rows_per_page * 2   # 2 columns of systems


    # Layout constants
    # Each system block: [name label | img1 | cbar1 | img2 | cbar2]
    # 2 system blocks side by side per row
    # Per system: 5 sub-columns (label + img + cbar + img + cbar)
    # Total grid cols: 5 * 2 = 10
    n_grid_cols  = 10
    name_col_w   = 0.6   # relative width of the name label column
    img_col_w    = 1.0   # relative width of each image column
    cbar_col_w   = 0.06  # relative width of each colorbar column
 
    col_widths   = [name_col_w, img_col_w, cbar_col_w, img_col_w, cbar_col_w,   # system A
                    name_col_w, img_col_w, cbar_col_w, img_col_w, cbar_col_w]    # system B
 
    fig_width    = 2 * (name_col_w + 2 * img_col_w + 2 * cbar_col_w) * img_size_inch
    fig_height   = rows_per_page * img_size_inch + 0.6  # +0.6 for header
 
    def _prep_img(img, log_scale):
        if log_scale:
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.log10(np.where(img > 0, img, np.nan))
        else:
            out = img.astype(float)
        return out
 
 
    page_indices = range(0, n, systems_per_page)
    n_pages = len(range(0, n, systems_per_page))
 
    with PdfPages(output_pdf) as pdf:
        for page_num, start in enumerate(page_indices):
            batch = list(range(start, min(start + systems_per_page, n)))
            n_batch = len(batch)
 
            # Actual rows needed for this page (may be < rows_per_page on last page)
            n_rows = (n_batch + 1) // 2
 
            fig_h = n_rows * img_size_inch + 0.6
            fig   = plt.figure(figsize=(fig_width, fig_h), dpi=dpi)
 
            # One extra row at top for column headers
            gs = gridspec.GridSpec(
                n_rows + 1, n_grid_cols,
                figure=fig,
                width_ratios=col_widths,
                height_ratios=[0.1] + [1.0] * n_rows,
                hspace=0.14,
                wspace=0.05,
            )
 
            # ── column headers ────────────────────────────────────────────
            for col_block in range(2):
                base = col_block * 5
                # img1 header (spans img1 + its cbar column)
                ax_h1 = fig.add_subplot(gs[0, base + 1:base + 3])
                ax_h1.set_axis_off()
                ax_h1.text(0.5, 0.5, label1, ha="center", va="center",
                           fontsize=8, fontweight="bold",
                           transform=ax_h1.transAxes)
                # img2 header (spans img2 + its cbar column)
                ax_h2 = fig.add_subplot(gs[0, base + 3:base + 5])
                ax_h2.set_axis_off()
                ax_h2.text(0.5, 0.5, label2, ha="center", va="center",
                           fontsize=8, fontweight="bold",
                           transform=ax_h2.transAxes)
 
            # ── system rows ───────────────────────────────────────────────
            for slot, idx in enumerate(batch):
                row      = slot // 2          # grid row (0-based, after header)
                col_block = slot % 2           # left (0) or right (1) column block
                gs_row   = row + 1            # +1 for header row
                base_col = col_block * 5
 
                img1 = _prep_img(images1[idx], log_scale1)
                img2 = _prep_img(images2[idx], log_scale2)
                if limits is not None:
                    lo1, hi1 = _limits(img1)
                    #lo2, hi2 = _limits(img2, vmin2, vmax2)
                else:
                    lo1,hi1 = limits[idx]
                lo2,hi2 = lo1,hi1
                
                # name label
                ax_nm = fig.add_subplot(gs[gs_row, base_col])
                ax_nm.set_axis_off()
                ax_nm.text(0.5, 0.5, names[idx],
                           ha="center", va="center",
                           fontsize=7, rotation=90,
                           transform=ax_nm.transAxes,
                           wrap=True)
 
                # image 1 + its colorbar
                ax1 = fig.add_subplot(gs[gs_row, base_col + 1])
                im1 = ax1.imshow(img1, origin="lower", cmap=cmap1,
                                 extent=extents[idx],
                                 vmin=lo1, vmax=hi1, 
                                 aspect="equal")

                #ax1.set_axis_off()
                #cax1 = fig.add_subplot(gs[gs_row, base_col + 2])
                #fig.colorbar(im1, cax=cax1,label=label_cl1)
                #cax1.tick_params(labelsize=5)
 
                # image 2 + its colorbar
                ax2 = fig.add_subplot(gs[gs_row, base_col + 3])
                im2 = ax2.imshow(img2, origin="lower", cmap=cmap2,
                                 extent=extents[idx],
                                 vmin=lo2, vmax=hi2, 
                                 aspect="equal")
                ax2.get_yaxis().set_visible(False) # it is shared w. im 1

                #ax2.set_axis_off()
                cax2 = fig.add_subplot(gs[gs_row, base_col + 4])
                fig.colorbar(im2, cax=cax2,label=label_cl2)
                cax2.tick_params(labelsize=5)
 
            # fill empty slot on last page if odd number of systems
            if n_batch % 2 == 1:
                row      = (n_batch - 1) // 2
                gs_row   = row + 1
                for c in range(5):
                    ax_empty = fig.add_subplot(gs[gs_row, 5 + c])
                    ax_empty.set_axis_off()
 
            """fig.suptitle(
                f"Page {page_num + 1}/{n_pages}  —  systems {start + 1}–{min(start + systems_per_page, n)} of {n}",
                fontsize=8, y=1.0
            )"""
 
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            del fig
            gc.collect()
 
            if verbose:
                print(f"Saved page {page_num + 1}/{n_pages} "
                      f"(systems {start + 1}–{min(start + systems_per_page, n)})")
 
    if verbose:
        print(f"Done → {output_pdf}")

# ── demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    N   = 47   # deliberately odd and non-multiple of page size

    def fake_galaxy(seed):
        rng = np.random.default_rng(seed)
        x   = np.linspace(-3, 3, 80)
        X, Y = np.meshgrid(x, x)
        img = np.exp(-(X**2 + Y**2) / 0.5) * rng.uniform(0.5, 2.0)
        img += rng.exponential(0.01, img.shape)
        return img

    images1 = [fake_galaxy(i)        for i in range(N)]
    images2 = [fake_galaxy(i) - fake_galaxy(i + N) for i in range(N)]
    names   = [f"Gn{i+1}SGn0"        for i in range(N)]

    plot_image_pairs_pdf(
        images1, images2, names,
        output_pdf="./image_pairs_demo.pdf",
        rows_per_page=5,
        img_size_inch=3.0,
        label1="Sim Image",
        label2="Norm. Residual",
        label_cl1 = "flux",
        label_cl2 = "flux",
        log_scale1=True,
        log_scale2=False,
        cmap1="gist_heat",
        cmap2="bwr",
        #vmin2=-3, vmax2=3,
        verbose=True,
    )