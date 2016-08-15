import numpy as np
import cv2

def applyAffineTransform(src, srcTri, dstTri, size):
	"""
	Apply affine transform calculated using srcTri and dstTri to src and output an image of size.
	"""
	# Given a pair of triangles, find the affine transform.
	warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )

	# Apply the Affine Transform just found to the src image
	dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
	return dst


def morphTriangle(img1, img2, img, t1, t2, t, alpha):
	"""
	Warps and alpha blends triangular regions from img1 and img2 to img.
	"""
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


def getIndices(rect, points):
	"""
	Returns indices of delaunay triangles.
	"""
	subdiv = cv2.Subdiv2D(rect)
	for p in points:
		subdiv.insert(p)
	triangleList = subdiv.getTriangleList()
	
	indicesTri = []
	ind = [None] * 3
	pt = [None] * 3
	for triangle in triangleList:
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


def getCorners(img, img2, pointsImg1, pointsImg2):
	"""Adds the corners and middle point of edges to pointlists.
	Finds the user selectet points which are nearest to the four corners and the 
	four middle points of the edges of the image. Computes the delta between 
	them and their coresponding points. Adds corner points and middle points of
	edges to the point lists and offsets them using the computet delta values.
	Returns the global max and minima used for cropping
	"""
	
	xMax = min(img.shape[1], img2.shape[1]) - 1
	yMax = min(img.shape[0], img2.shape[0]) - 1
	xMean = int(xMax / 2)
	yMean = int(yMax / 2)

	# left middle
	pMinMean, iMinMean = min(((val, idx) for (idx, val) in enumerate(pointsImg1)), key=lambda p: (p[0])[0] + abs(yMean - (p[0])[1]))
	deltaYHalf = int((pMinMean[1] - (pointsImg2[iMinMean])[1])/2)
	deltaXHalf = int((pMinMean[0] - (pointsImg2[iMinMean])[0])/2)
	pointsImg1.append((0 + abs(deltaXHalf) + deltaXHalf, pMinMean[1] + deltaYHalf))
	pointsImg2.append((0 + abs(deltaXHalf) - deltaXHalf, pMinMean[1] - deltaYHalf))
	globalXMin = abs(deltaXHalf)
	
	# right middle
	pMaxMean, iMaxMean = min(((val, idx) for (idx, val) in enumerate(pointsImg1)), key=lambda p: (xMax -(p[0])[0]) + abs(yMean - (p[0])[1]))
	deltaYHalf = int((pMaxMean[1] - (pointsImg2[iMaxMean])[1])/2)
	deltaXHalf = int((pMaxMean[0] - (pointsImg2[iMaxMean])[0])/2)
	pointsImg1.append((xMax - abs(deltaXHalf) + deltaXHalf, pMaxMean[1] + deltaYHalf))
	pointsImg2.append((xMax - abs(deltaYHalf) - deltaXHalf, pMaxMean[1] - deltaYHalf))
	globalXMax = xMax - abs(deltaXHalf)
	
	# top middle
	pMeanMin, iMeanMin = min(((val, idx) for (idx, val) in enumerate(pointsImg1)), key=lambda p: abs(xMean - (p[0])[0]) +  (p[0])[1])
	deltaXHalf = int((pMeanMin[0] - (pointsImg2[iMeanMin])[0])/2)
	deltaYHalf = int((pMeanMin[1] - (pointsImg2[iMeanMin])[1])/2)
	pointsImg1.append((pMeanMin[0] + deltaXHalf, 0 + abs(deltaYHalf) + deltaYHalf))
	pointsImg2.append((pMeanMin[0] - deltaXHalf, 0 + abs(deltaYHalf) - deltaYHalf))
	globalYMin = abs(deltaYHalf)

	# bottom middle
	pMeanMax, iMeanMax = min(((val, idx) for (idx, val) in enumerate(pointsImg1)), key=lambda p: abs(xMean - (p[0])[0]) + (yMax - (p[0])[1]))
	deltaXHalf = int((pMeanMax[0] - (pointsImg2[iMeanMax])[0])/2)
	deltaYHalf = int((pMeanMax[1] - (pointsImg2[iMeanMax])[1])/2)
	pointsImg1.append((pMeanMax[0] + deltaXHalf, yMax - abs(deltaYHalf) + deltaYHalf))
	pointsImg2.append((pMeanMax[0] - deltaXHalf, yMax - abs(deltaYHalf) - deltaYHalf))
	globalYMax = yMax - abs(deltaYHalf)
	
	# bottom left 
	pMinMax, iMinMax = max(((val, idx) for (idx, val) in enumerate(pointsImg1)), key=lambda p: (xMax - (p[0])[0]) + (p[0])[1])
	deltaYHalf = int((pMinMax[1] - (pointsImg2[iMinMax])[1])/2)
	deltaXHalf = int((pMinMax[0] - (pointsImg2[iMinMax])[0])/2)
	pointsImg1.append((0 + abs(deltaXHalf) + deltaXHalf, yMax - abs(deltaYHalf) + deltaYHalf))
	pointsImg2.append((0 + abs(deltaXHalf) - deltaXHalf, yMax - abs(deltaYHalf) - deltaYHalf))
	globalXMin = abs(deltaXHalf) if abs(deltaXHalf) > globalXMin	else globalXMin
	globalYMax = yMax - abs(deltaYHalf) if (yMax - abs(deltaYHalf)) < globalYMax	else globalYMax

	# bottom right 
	pMaxMax, iMaxMax = max(((val, idx) for (idx, val) in enumerate(pointsImg1)), key=lambda p: (p[0])[0] + (p[0])[1])
	deltaYHalf = int((pMaxMax[1] - (pointsImg2[iMaxMax])[1])/2)
	deltaXHalf = int((pMaxMax[0] - (pointsImg2[iMaxMax])[0])/2)
	pointsImg1.append((xMax - abs(deltaXHalf) + deltaXHalf, yMax - abs(deltaYHalf) + deltaYHalf))
	pointsImg2.append((xMax - abs(deltaXHalf) - deltaXHalf, yMax - abs(deltaYHalf) - deltaYHalf))
	globalXMax = xMax - abs(deltaXHalf) if (xMax - abs(deltaXHalf)) < globalXMax	else globalXMax
	globalYMax = yMax - abs(deltaYHalf) if (yMax - abs(deltaYHalf)) < globalYMax	else globalYMax

	# top right 
	pMaxMin, iMaxMin = max(((val, idx) for (idx, val) in enumerate(pointsImg1)), key=lambda p: (p[0])[0] + (yMax - (p[0])[1]))
	deltaYHalf = int((pMaxMin[1] - (pointsImg2[iMaxMin])[1])/2)
	deltaXHalf = int((pMaxMin[0] - (pointsImg2[iMaxMin])[0])/2)
	pointsImg1.append((xMax - abs(deltaXHalf) + deltaXHalf, 0 + abs(deltaYHalf) + deltaYHalf))
	pointsImg2.append((xMax - abs(deltaXHalf) - deltaXHalf, 0 + abs(deltaYHalf) - deltaYHalf))
	globalXMax = xMax - abs(deltaXHalf) if (xMax - abs(deltaXHalf)) < globalXMax	else globalXMax
	globalYMin = abs(deltaYHalf) if abs(deltaYHalf) > globalYMin	else globalYMin

	# top left
	pMinMin, iMinMin = min(((val, idx) for (idx, val) in enumerate(pointsImg1)), key=lambda p: (p[0])[0] + (p[0])[1])
	deltaYHalf = int((pMinMin[1] - (pointsImg2[iMinMin])[1])/2)
	deltaXHalf = int((pMinMin[0] - (pointsImg2[iMinMin])[0])/2)
	pointsImg1.append((0 + abs(deltaXHalf) + deltaXHalf, 0 + abs(deltaYHalf) + deltaYHalf))
	pointsImg2.append((0 + abs(deltaXHalf) - deltaXHalf, 0 + abs(deltaYHalf) - deltaYHalf))
	globalXMin = abs(deltaXHalf) if abs(deltaXHalf) > globalXMin	else globalXMin
	globalYMin = abs(deltaYHalf) if abs(deltaYHalf) > globalYMin	else globalYMin

	return globalXMin, globalXMax, globalYMin, globalYMax


def morph(img1, img2, pointsImg1, pointsImg2, alpha = 0.5, steps = 2):
	"""Returns list of morphed images."""

	assert 0 <= alpha <= 1, "Alpha not between 0 and 1."
	assert len(pointsImg1) == len(pointsImg2), "Point lists have different size."

	# Convert Mat to float data type
	img1 = np.float32(img1)
	img2 = np.float32(img2)

	# Add the corner points and middle point of edges to the point lists
	globalXMin, globalXMax, globalYMin, globalYMax = getCorners(img1, img2, pointsImg1, pointsImg2)

	# Compute weighted average point coordinates
	points = [];
	for i in xrange(0, len(pointsImg1)):
		x = int(( 1 - alpha ) * pointsImg1[i][0] + alpha * pointsImg2[i][0])
		y = int(( 1 - alpha ) * pointsImg1[i][1] + alpha * pointsImg2[i][1])
		points.append((x,y))

	rect = (0, 0, max(img1.shape[1], img2.shape[1]), max(img1.shape[0],img2.shape[0]))
	indicesTri = getIndices(rect, points)
	
	images = []
	for a in np.linspace(0.0, 1.0, num = steps):
		# Allocate space for final output
		imgMorph = np.zeros((max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1]), max(img1.shape[2], img2.shape[2])), dtype=img1.dtype)

		for ind in indicesTri:
			x = ind[0]
			y = ind[1]
			z = ind[2]

			t1 = [pointsImg1[x], pointsImg1[y], pointsImg1[z]]
			t2 = [pointsImg2[x], pointsImg2[y], pointsImg2[z]]
			t = [ points[x], points[y], points[z] ]

			# Morph one triangle at a time.
			morphTriangle(img1, img2, imgMorph, t1, t2, t, a)
		# add cropped images to list
		images.append(np.copy(np.uint8(imgMorph[globalYMin:globalYMax, globalXMin:globalXMax, : ])))
	return images
