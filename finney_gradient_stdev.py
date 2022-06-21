import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import SimpleITK as sitk
from scipy.ndimage import generic_filter
import scipy.ndimage

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

arr3d = np.load("Finney_N500.npy")

adjust_range = False
gaussian_blur = False
gradient_kernel = "directional"

# arr3d = arr3d[256:768,256:768,256:768]
print(arr3d.shape)
arr3d[arr3d > 0] = 1

# Since we'd like [0, 1] greyvalues to be a normalised basis,
# we will move the pore value from 0 to 0.25 and the solid value
# from 1 to 0.75.
# This leaves a bit of data range in [0, 1] for the application of noise
if adjust_range == True:
    arr3d = arr3d * 0.5
    arr3d = arr3d + 0.25
if gaussian_blur == True:
    arr3d = scipy.ndimage.filters.gaussian_filter(arr3d,
                                                  sigma=0.8)
# arr3d = np.random.normal(arr3d,
#                          scale=0.1)

if gradient_kernel == "prewitt":
    norm = 18.
    gx = np.zeros((3,3,3), dtype=np.float32)
    gx[:, :, 0] = 1/norm
    gx[:, :, 2] = -1/norm
    gy = np.zeros((3,3,3), dtype=np.float32)
    gy[:, 0, :] = 1/norm
    gy[:, 2, :] = -1/norm
    gz = np.zeros((3,3,3), dtype=np.float32)
    gz[0, :, :] = 1/norm
    gz[2, :, :] = -1/norm
elif gradient_kernel == "directional":
    # directional
    norm = 2.
    gx = np.zeros((3,3,3), dtype=np.float32)
    gx[1, 1, 0] = 1/norm
    gx[1, 1, 2] = -1/norm
    gy = np.zeros((3,3,3), dtype=np.float32)
    gy[1, 0, 1] = 1/norm
    gy[1, 2, 1] = -1/norm
    gz = np.zeros((3,3,3), dtype=np.float32)
    gz[0, 1, 1] = 1/norm
    gz[2, 1, 1] = -1/norm
    
# print(gx)
# print(gy)
# print(gz)

img_gx = convolve(arr3d, gx)
img_gy = convolve(arr3d, gy)
img_gz = convolve(arr3d, gz)

gmag = np.sqrt(img_gx * img_gx + img_gy * img_gy + img_gz * img_gz)

img = sitk.GetImageFromArray(arr3d)


std_filter = sitk.NoiseImageFilter()
std_filter.SetRadius(1)
std_img = std_filter.Execute(img)
std_sitk = sitk.GetArrayFromImage(std_img)

H, xedges, yedges = np.histogram2d(gmag.ravel(),
                                   std_sitk.ravel(),
                                   bins=np.linspace(0, 1., 100))
H = H.T

fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=[8, 10])
vmin=1e-7
vmax=1.
ax1.imshow(H / np.sum(H),
           interpolation='none',
           origin="lower",
           extent=[xedges[0],
                   xedges[-1],
                   yedges[0],
                   yedges[-1]],
           aspect='equal',
           cmap='viridis',
           norm=LogNorm(vmin=vmin,vmax=vmax))

# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# cbar = fig.colorbar(cm.ScalarMappable(norm = LogNorm(vmin=vmin,
#                                                      vmax=vmax),
#                                       cmap='viridis'),
#                     ax=ax1,
#                     cax=cax)

# cbar.set_label("frac. of voxels", rotation=270)
# cbar.ax.get_yaxis().labelpad = 15
# ax1.plot(xedges, yedges, 'k--', label="1:1")
# ax1.plot(xedges, 2./3 * yedges, 'r--', label="2/3")
# ax1.plot(xedges, 1./2. * yedges, 'g--', label="1/2")

xmax=1.
ymax=0.5
ax1.set_xlim(0, xmax)
ax1.set_ylim(0, ymax)
ax1.set_aspect(xmax/ymax)
ax1.set_xlabel(r"$\left|G\right|$")
ax1.set_ylabel(r"$\sigma$")
ax1.xaxis.label.set_size(16)
ax1.yaxis.label.set_size(16)
ax1.legend(loc=4)
ax2.imshow(arr3d[arr3d.shape[0] // 2],cmap='Greys_r')

# std_diff = std_sitk - std_arr

# plt.figure()
# plt.imshow(grad_arr[grad_arr.shape[0] // 2],
#            # vmin=0,
#            # vmax=1,
#            cmap='Greys_r')
# plt.imshow(gdiff[gdiff.shape[0] // 2],
#            # vmin=0,
#            # vmax=1,
#            cmap='Greys_r')
# plt.imshow(std_sitk[std_sitk.shape[0] // 2],
#            # vmin=0,
#            # vmax=1,
#            cmap='Greys_r')

# plt.title("Kalisphera rescaled [0.25, 0.75]")
plt.show()



