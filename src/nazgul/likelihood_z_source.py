# Defining likelihood of z_source 
# used to sample the z_source given z_lens
# based on lenspop results
# https://github.com/tcollett/LensPop
import numpy as np
from pathlib import Path

from python_tools.tools import Read_Column_File

LensPop_dir = Path("./LensPop/LensPop")
file0       = LensPop_dir/"lenses_LSSTa.txt"
if LensPop_dir.is_dir() is False or file0.is_file() is False :
    LensPop_dir.mkdir()
    print("Downloading the LensPop catalogue for LSST")
    import requests
    git_path = "https://raw.githubusercontent.com/tcollett/LensPop/master"
    for lett in "a","b","c":
        file_name = "lenses_LSST"+lett+".txt"
        raw_file  = requests.get(git_path+"/"+file_name)
        with open(LensPop_dir/file_name,"w") as f:
            f.write(raw_file)
    
print("This data has been taken from tcollett/LensPop.git, assuming the LSST telescope")

zla = Read_Column_File(LensPop_dir/"lenses_LSSTa.txt")[0]
zsa = Read_Column_File(LensPop_dir/"lenses_LSSTa.txt")[1]
zlb = Read_Column_File(LensPop_dir/"lenses_LSSTb.txt")[0]
zsb = Read_Column_File(LensPop_dir/"lenses_LSSTb.txt")[1]
zlc = Read_Column_File(LensPop_dir/"lenses_LSSTc.txt")[0]
zsc = Read_Column_File(LensPop_dir/"lenses_LSSTc.txt")[1]


stat_zl = np.hstack([zla,zlb,zlc])
stat_zs = np.hstack([zsa,zsb,zsc])

# the simplest way would be to return a likelihood based on
# all z_sources
def logL_z_source_all(z_source,z_lens=None,*args):
    # ignore z_lens
    lkl,bins = np.histogram(stat_zs,bins=40,density=1)
    zs       = (bins[1:]+bins[:-1])/2
    return np.interp(z_source,zs,lkl)

# slighly more complex (not sure how correct)
# given the z_lens, select a range of zl around, 
# and only fit for z_s of these lenses
def logL_z_source_zl(z_source,z_lens,dzl=0.2,*args):
    stat_zs_cut = stat_zs[np.where(np.abs(stat_zl-z_lens)<dzl)]
    lkl,bins = np.histogram(stat_zs_cut,bins=40,density=1)
    zs       = (bins[1:]+bins[:-1])/2
    return np.interp(z_source,zs,lkl)


kw_prior_z_source_all = {"f_lkl_z_source":logL_z_source_all,
                          "prms_lkl_z_source":[]}
kw_prior_z_source_zl = {"f_lkl_z_source":logL_z_source_zl,
                          "prms_lkl_z_source":[]}


"""
kw_prior_z_source_easy = kw_prior_z_source_all
kw_prior_z_source_mid  = kw_prior_z_source_zl
# Plots
plt.title("LSST expected lenses population")
plt.xlabel("z")
plt.hist(stat_zl,bins=30,color="b",alpha=.4,density=1,label=r"z$_l$")
plt.hist(stat_zs,bins=30,color="r",alpha=.4,density=1,label=r"z$_s$")

z_rng = np.linspace(0,max(stat_zs),1000)
plt.plot(z_rng,logL_z_source_easy(z_rng),c="k",ls="--",label="lkl fit")
plt.legend()
plt.savefig("tmp/zsl.png")

plt.close()
plt.title("LSST expected lenses population")
plt.xlabel("z")
z_lens = .5
plt.axvline(z_lens,c="r",ls="-.")
dzl = 0.2
plt.axes().axvspan(z_lens-dzl,z_lens+dzl,alpha=.4,color="k",label=r"z$_l$ range considered")
stat_zl_cut = stat_zl[np.where(np.abs(stat_zl-z_lens)<dzl)]
plt.hist(stat_zl_cut,bins=10,color="b",alpha=.4,density=1,label=r"corresponding z$_l$")
stat_zs_cut = stat_zs[np.where(np.abs(stat_zl-z_lens)<dzl)]
plt.hist(stat_zs_cut,bins=40,density=1,alpha=.4,color="r",label=r"corresponding z$_s$")
plt.plot(z_rng,logL_z_source_mid(z_rng,z_lens=z_lens),color="k",ls="--",label="lkl fit")
plt.legend()
plt.savefig("tmp/zsl2.png")
"""