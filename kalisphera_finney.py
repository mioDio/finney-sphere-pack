import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spam.kalisphera as kali
import spam
import spam.datasets
import math

flag = "finney"

df = pd.read_csv("data/FinneyPack4021.txt",
                 delim_whitespace=True,
                 names=["idx", "x", "y", "z"])

spheres = list(zip(np.array(df["x"]),
                   np.array(df["y"]),
                   np.array(df["z"])))


########### Example ############
################################

if flag == "example":
    pix_size = 30e-6
    boxSize, centres, radii = spam.datasets.loadDEMboxsizeCentreRadius()
    
    print(f"Box size: {boxSize}")
    print(f"Centres shape: {centres.shape}")
    print(f"Radii shape: {radii.shape}")
    
    N = int(math.ceil(np.max(boxSize[:]) / pix_size))
    print(f"N: {N}")
    print(f"Min centre: {min(centres.ravel())}")

########### Finney ##############
#################################

if flag == "finney":
    boxSize = 40 # mm
    # bbox = [[-20, 20],
    #         [-20, 20],
    #         [-20, 20]]
    
    D = 2.  # mm, diameter of each sphere
    radius = D / 2.
    N = 256
    
    pix_size = boxSize / N
    centres = np.array(spheres) + boxSize * 0.5
    radii = np.ones(centres.shape[0]) * radius

    print(f"Pix size: {pix_size} mm")
    print(f"First centre: {spheres[0]} mm")
    print(f"Radii in mm: {radii[0]}")

print(f"Pix size: {pix_size}")
# convert mm to voxels
centres = np.array(centres) / pix_size
radii = radii / pix_size
print(f"Centre in pix: {centres[0]}")
print(f"Radii in pix: {radii[0]}")

Box = np.zeros((N, N, N), dtype="<f8")
kali.makeSphere(Box, centres, radii)

Box[np.where(Box > 1.0)] = 1.0
Box[np.where(Box < 0.0)] = 0.0

np.save("Finney_N" + str(N) + ".npy", Box)

plt.figure()
plt.imshow(Box[Box.shape[0] // 2],
           vmin=0,
           vmax=1,
           cmap='Greys_r')
# plt.title("Kalisphera rescaled [0.25, 0.75]")
plt.show()
