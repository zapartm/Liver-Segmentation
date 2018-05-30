import dicom
import os
import numpy
import cv2
import numpy as np
from matplotlib import pyplot, cm

PathDicom = "./prez/"
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

# loop through all the DICOM files
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

cols = len(ArrayDicom);
#
# pyplot.figure(dpi=300)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 2]))
# pyplot.show()


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

# def region_growing(img, start_point, seed):
#     list = []
#     outimg = np.zeros_like(img)
#     list.append((start_point[0], start_point[1]))
#     processed = []
#     factor = 4
#     counter = 0
#     while(len(list) > 0):
#         pix = list[0]
#         outimg[pix[0], pix[1]] = 255
#         for coord in get8n(pix[0], pix[1], img.shape):
#             if outimg[coord[0], coord[1]] != 255 and img[coord[0], coord[1]] > seed - factor and img[coord[0], coord[1]] < seed + factor:
#                 outimg[coord[0], coord[1]] = 255
#                 list.append(coord)
#         list.pop(0)
#         if counter == 100:
#             counter = 0
#             cv2.imshow("progress",outimg)
#             cv2.waitKey(1)
#         counter += 1
#     return outimg

def region_growing(img, start_point, seed):
    list = []
    outimg = np.zeros_like(img)
    for row in range(len(outimg)):
        for col in range(len(outimg[0])):
            outimg[row, col] = img[row, col]

    list.append((start_point[0], start_point[1]))
    processed = []
    factor = 2
    counter = 0
    while(len(list) > 0):
        pix = list[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            if outimg[coord[0], coord[1]] != 255 and img[coord[0], coord[1]] > seed - factor and img[coord[0], coord[1]] < seed + factor:
                outimg[coord[0], coord[1]] = 255
                list.append(coord)
        list.pop(0)
        if counter == 100:
            counter = 0
            cv2.imshow("progress",outimg)
            cv2.waitKey(1)
        counter += 1
    return outimg


def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyWindow('Input')
        seed = image[y,x]
        print('Seed: ' + str(x) + ', ' + str(y), seed)
        img2 = region_growing(image, (y,x), seed)
        cv2.destroyWindow("progress")
        cv2.imshow('proccessing image', img2)
        cv2.waitKey()

image = cv2.imread('3.png', 0)
# image = ArrayDicom[:, :, 50]

cv2.namedWindow('Input')
cv2.setMouseCallback('Input', on_mouse, 0, )
cv2.imshow('Input', image)
cv2.waitKey()
