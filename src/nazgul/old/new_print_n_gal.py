# copy from print_n_gal.py
# adapted to plot N of galaxies at different redshifts
# then output a table of indexes to use to get them, with 
# -  mass
# - coordinates
# - redshift

# ude get_gal_indexes to have a better statistic
import time 
import pickle
import numpy as np
import matplotlib.pyplot as plt

from get_gal_indexes import get_gals
from fnct import Galaxy,sim_path

myData_Gals = get_gals(plot=False,save_pkl=False)

# Plot a bit of info
"""
plt.scatter(myData_Gals["z"],np.log(myData_Gals["N_gal"]))
plt.xlabel("z")
plt.ylabel("log(N_gal)")
plt.savefig("high_Ngalvsz.png")
plt.close()
plt.scatter(myData_Gals["z"],np.log(myData_Gals["Mtot_gal"]))
plt.ylabel("log(Sum(M_gal(zi)))")
plt.xlabel("z")
plt.savefig("high_Mgalvsz.png")
print("Tot n gal considered",np.sum(myData_Gals["N_gal"]))
"""
CntX,CntY,CntZ = myData_Gals["CMx"],myData_Gals["CMy"],myData_Gals["CMz"]
Gn,Sgn = myData_Gals["Gn"],myData_Gals["SGn"]
Z =myData_Gals["z"]
print("N gals:",len(Z))

n_stars = []
n_dm = []
n_gas = []
n_bh = []

m_stars = []
m_dm = []
m_gas = []
m_bh = []
pkl_name = sim_path+"/kwres_new_print_n_gal.pkl"
names  = "stars","gas","dm","bh"

try:
    with open(pkl_name,"rb") as f:
        kw_res = pickle.load(f)
    for n in names:
        if n=="stars":
            m_stars,n_stars = kw_res[n]
        elif n=="gas":
            m_gas,n_gas = kw_res[n]
        elif n=="dm":
            m_dm,n_dm = kw_res[n]
        elif n=="bh":
            m_bh,n_bh = kw_res[n]
except FileNotFoundError:
    

    t0 = time.time()
    
    for i,(z,cntx,cnty,cntz,gn,sgn)  in enumerate(zip(Z,CntX,CntY,CntZ,Gn,Sgn)):
        
        dt = time.time()-t0
        prc = i*100/len(Z)
        eta = (len(Z)-i)/(i+1/dt)
        print("Time [s]:",dt,"\nN:",i,"\nPerc:",prc,"\nETA[min]:",eta/60.,"\n")
        centre = np.array([cntx,cnty,cntz])
        gl = Galaxy(Gn=gn,SGn=sgn,CntX=cntx,CntY=cnty,CntZ=cntz,z=z)
        n_stars.append(len(gl.stars["mass"]))   
        m_stars.append(np.sum(gl.stars["mass"]))
        
        n_dm.append(len(gl.dm["mass"]))   
        m_dm.append(np.sum(gl.dm["mass"]))
        
        n_gas.append(len(gl.gas["mass"]))   
        m_gas.append(np.sum(gl.gas["mass"]))
    
        n_bh.append(len(gl.bh["mass"]))   
        m_bh.append(np.sum(gl.bh["mass"]))
    masses = [m_stars,m_gas,m_dm,m_bh]
    numb   = [n_stars,n_gas,n_dm,n_bh]

    kw_res = {}
    for n,m,nb in zip(names,masses,numb):
        kw_res[n] = [m,nb]
    with open(pkl_name,"wb") as f:
        pickle.dump(kw_res,f)

print("Average log(N) stars part.",np.mean(np.log(n_stars)))
print("Average log(N) gas part.",np.mean(np.log(n_gas)))
print("Average log(N) dm part.",np.mean(np.log(n_dm)))
print("Average log(N) bh part.",np.mean(np.log(n_bh)))

print("Average log(M) stars part.",np.mean(np.log(m_stars)))
print("Average log(M) gas part.",np.mean(np.log(m_gas)))
print("Average log(M) dm part.",np.mean(np.log(m_dm)))
print("Average log(M) bh part.",np.mean(np.log(m_bh)))

for m,n,name in zip(masses,numb,names):
    # ignore gals w/o this specific particle
    # -> this is the case for stars,gas and bh, not dm
    m = np.array(m)
    n = np.array(n)
    m = m[m>0]
    n = n[n>0]
    
    hist,bins,_ =plt.hist(m)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.close()
    plt.hist(m,bins=logbins)
    plt.xscale("log")
    plt.title("Masses "+name+" (ignoring 'empty' galaxies)")
    #as in, wo considering the ones without this particle
    plt.xlabel(r"M [M$_{\odot}$]")
    plt.savefig(sim_path+"/masses_"+name+".pdf")
    plt.close()
    
    hist,bins,_ =plt.hist(n)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.close()
    plt.hist(n,bins=logbins)
    plt.xscale("log")    
    plt.title("Number of particles for "+name)
    plt.savefig(sim_path+"/number_"+name+".pdf")
    plt.close()

    