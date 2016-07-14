#!/usr/bin/env python

import numpy as np
import cv2
import sys

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def insertLastPoints(img, points):
	"""Insert corners and mids of edges of image to point list."""
	size = img.shape
	x = img.shape[1]-1
	y = img.shape[0]-1
	x_half = int(x/2)
	y_half = int(y/2)

	points.append((0, 0))
	points.append((x_half, 0))
	points.append((x, 0))
	points.append((x, y_half))
	points.append((x, y))
	points.append((x_half, y))
	points.append((0, y))
	points.append((0, y_half))

def getIndices(rect, points):
	"""returns indices of delaunay triangles"""
	subdiv = cv2.Subdiv2D(rect)
	for p in points:
		subdiv.insert(p)
	triangle_list = subdiv.getTriangleList()
	
	indicesTri = []
	ind = [None] * 3
	pt = [None] * 3
	for triangle in triangle_list:
		pt[0] = (int(triangle[0])), int(triangle[1])
		pt[1] = (int(triangle[2])), int(triangle[3])
		pt[2] = (int(triangle[4])), int(triangle[5])
		if rect[0] <= (pt[0])[0] <= rect[2] and rect[1] <= (pt[0])[1] <= rect[3] and \
			rect[0] <= (pt[1])[0] <= rect[2] and rect[1] <= (pt[1])[1] <= rect[3] and \
			rect[0] <= (pt[2])[0] <= rect[2] and rect[1] <= (pt[2])[1] <= rect[3]:
			for i in range(0,3):
				for j in xrange(0, len(points)):
					if (((pt[i])[0] == (points[j])[0]) and ((pt[i])[1] ==  (points[j])[1])):
						ind[i] = j
			indicesTri.append(list(ind))
	return indicesTri

def delaunayMorphing(img1, img2, points_img1, points_img2, alpha = 0.5, steps = 2):
	"""Returns list of morphed images."""

	assert 0 <= alpha <= 1, "Alpha not between 0 and 1."
	assert len(points_img1) == len(points_img2), "Point list have different size."

	# Convert Mat to float data type
	img1 = np.float32(img1)
	img2 = np.float32(img2)

	insertLastPoints(img1, points_img1)
	insertLastPoints(img2, points_img2)

	points = [];

	# Compute weighted average point coordinates
	for i in xrange(0, len(points_img1)):
		x = int(( 1 - alpha ) * points_img1[i][0] + alpha * points_img2[i][0])
		y = int(( 1 - alpha ) * points_img1[i][1] + alpha * points_img2[i][1])
		points.append((x,y))

	rect = (0, 0, max(img1.shape[1], img2.shape[1]), max(img1.shape[0],img2.shape[0]))
	indicesTri = getIndices(rect, points)
	
	images = []
	for a in np.linspace(0.0, 1.0, num = steps):
		# Allocate space for final output
		imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

		for ind in indicesTri:
			x = ind[0]
			y = ind[1]
			z = ind[2]

			t1 = [points_img1[x], points_img1[y], points_img1[z]]
			t2 = [points_img2[x], points_img2[y], points_img2[z]]
			t = [ points[x], points[y], points[z] ]

			# Morph one triangle at a time.
			morphTriangle(img1, img2, imgMorph, t1, t2, t, a)
		images.append(np.copy(np.uint8(imgMorph)))
	return images
