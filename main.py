import dicom
import os
import numpy
import copy
import numpy as np
import matplotlib.pyplot as plt

# PARAMS
# eps - "czułość" algorytmu, większa wartośc - większy obszar zostanie obięty
# PathDicom - ścieżka do katalogu z dicomami
# NUM = indeks interesującego nas przekroju
eps = 25
PathDicom = "./dicoms/"
NUM = 50

lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# read the file and store image data
for filenameDCM in lstFilesDCM:
    ds = dicom.read_file(filenameDCM)
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array


def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out

def region_growing(img, start_point, seed):
    print("processing...")
    list = []
    outimg = np.zeros_like(img)
    max = -2000
    for row in range(len(outimg)):
        for col in range(len(outimg[0])):
            outimg[row, col] = img[row, col]
            if img[row,col] > max:
                max = img[row,col]

    list.append((start_point[0], start_point[1]))
    processed = []
    factor = eps
    while(len(list) > 0):
        pix = list[0]
        outimg[pix[0], pix[1]] = max
        for coord in get8n(pix[0], pix[1], img.shape):
            if outimg[coord[0], coord[1]] != max and img[coord[0], coord[1]] > seed - factor and img[coord[0], coord[1]] < seed + factor:
                outimg[coord[0], coord[1]] = max
                list.append(coord)
        list.pop(0)

    print("Region growing: done. ")
    print("Starting post processing...")
    outimg2 = copy.deepcopy(outimg)
    for row in range(len(outimg)):
        for col in range(len(outimg[0])):
            counter = 0
            for coord in get8n(row, col, img.shape):
                if(outimg[coord[0], coord[1]] == max):
                    counter = counter + 1
            if counter > 2:
                outimg2[row, col] = max

    print("all: done")
    return outimg2


def get_pixels_hu(scans, referenceDM):
    # image = np.stack([s.pixel_array for s in scans])
    image = scans.pixel_array
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = referenceDM.RescaleIntercept
    slope = referenceDM.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

ds2 = dicom.read_file(lstFilesDCM[NUM])
image = get_pixels_hu(ds2, RefDs)

ax = plt.gca()
fig = plt.gcf()
implot = ax.imshow(image, cmap='gray')
plt.axis('off')
def onclick(event):
    if event.xdata != None and event.ydata != None:
        x = int(event.xdata)
        y = int(event.ydata)
        seed = image[y,x]
        print('Seed: ' + str(x) + ', ' + str(y), seed)
        img2 = region_growing(image, (y,x), seed)
        implot = ax.imshow(img2, cmap='gray')
        plt.show()

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
