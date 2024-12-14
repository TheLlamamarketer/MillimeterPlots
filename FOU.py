import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


# Load the image in grayscale
img = cv2.imread('G1.png', cv2.IMREAD_UNCHANGED)
print(img.shape, img.dtype, img.min(), img.max())

red_chan = img[:,:,2]
cmap_yell = plt.cm.colors.LinearSegmentedColormap.from_list('custom_yellow', ['black', 'yellow'], N=256)


green_chan = img[:,:,1]
arrow_mask = (green_chan > 50) & (red_chan > 50)

green_working = green_chan.astype(float)
green_working[arrow_mask] = np.nan

slices = []
range_slices = (0, 1280)

slices = []
for i in range(range_slices[0], range_slices[1]):
    vertical_slice = green_working[:, i]
    normalized_slice = vertical_slice / 255.0 
    slices.append(normalized_slice)

average_slice = np.nanmean(slices, axis=0)
std_slice = np.nanstd(slices, axis=0)

plt.plot(average_slice, label='Normalized Intensity')
plt.fill_between(np.arange(len(average_slice)),
                 average_slice - std_slice,
                 average_slice + std_slice,
                 alpha=0.3, label='Std Dev')
plt.xlabel('Pixel Row')
plt.ylabel('Intensity')
plt.title('Intensity Profile with Error Bounds')
plt.legend()
plt.show()

# plot the image with the slice being shown as a line
#plt.imshow(green_channel, cmap='Greens')
cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_green', ['black', 'green', 'lime'], N=256)
plt.imshow(red_chan, cmap=cmap_yell)
plt.imshow(green_working, cmap=cmap)
plt.gca().set_axis_off()
plt.gca().set_facecolor('none')
plt.gcf().patch.set_alpha(0.0) 

#plt.fill_betweenx(np.arange(0, green_chan.shape[0]), range_slices[0], range_slices[1], color='orange', alpha=0.3)
plt.show()
