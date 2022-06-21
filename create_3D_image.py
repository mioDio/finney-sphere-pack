import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/FinneyPack4021.txt",
                 delim_whitespace=True,
                 names=["idx", "x", "y", "z"])

spheres_all = list(zip(np.array(df["x"]),
                       np.array(df["y"]),
                       np.array(df["z"])))

bbox = [[-20, 20],
        [-20, 20],
        [-20, 20]]
D = 1.  # diameter of each sphere
N = 50

xr = np.linspace(bbox[0][0], bbox[0][1], N)
yr = np.linspace(bbox[1][0], bbox[1][1], N)
zr = np.linspace(bbox[2][0], bbox[2][1], N)
print(xr.max())
print(xr.min())

resx = (abs(bbox[0][0]) + abs(bbox[0][1])) / N
print(f"Resolution: {resx}")
print(f"Pixels per sphere: {1./resx}")

def inside_sphere(coords, sphere):
    xdiff = coords[0] - sphere[0]
    ydiff = coords[1] - sphere[1]
    zdiff = coords[2] - sphere[2]
    dist = np.sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff)
    if dist <= 0.5:
        return True
    elif dist > 0.5:
        return False

print(np.array(spheres_all).shape)

def filter_spheres(x, y, z):
    s_arr = np.array(spheres_all)
    diffx = abs(s_arr[:, 0] - x)
    diffy = abs(s_arr[:, 1] - y)
    diffz = abs(s_arr[:, 2] - z)
    close_x = diffx <= 1.
    close_y = diffy <= 1.
    close_z = diffz <= 1.
    closexy = np.logical_or(close_x, close_y)
    close = np.logical_or(closexy, close_z)
    # print(close.shape)
    # print(np.sum(close))
    spheres = s_arr[close, :]
    return spheres
    
img = np.zeros((N, N, N),
               dtype=np.uint8)

for i, x in enumerate(xr):
    for j, y in enumerate(yr):
        for k, z in enumerate(zr):
            spheres = filter_spheres(x, y, z)
            if len(spheres) == 0:
                break
            for s in spheres[:]:
                if inside_sphere([x, y, z], s):
                    img[i, j, k] = 1
                    break

np.save("data/img_" + str(N) + ".npy", img)   
plt.imshow(img[:, :, int(N/2)])
plt.show()

