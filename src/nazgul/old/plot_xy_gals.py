import numpy as np
import matplotlib.pyplot as plt
from get_gal_indexes import get_gals
myData_Gals = get_gals(plot=False,save_pkl=False)
CntX,CntY,CntZ = myData_Gals["CMx"],myData_Gals["CMy"],myData_Gals["CMz"]
plt.scatter(CntX,CntY,c=np.log(myData_Gals["M"]),alpha=.2,cmap="coolwarm_r",marker=".")
plt.colorbar(label=r"log(Mass/M$_\odot$)")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("plot_xy_gals.pdf")
