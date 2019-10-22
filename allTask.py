import copy as cp
import imutils
import cv2
import numpy as np
import sys
import re
import os
import math


# Checks if the path exists
def check_path(path):
    if not os.path.exists(path):
        print('ERROR! The given path does not exist.')
        sys.exit(0)


# Finds corners : harris and shitomasi 2 Task Find corner
'''
Function : cv2.cornerHarris(image,blocksize,ksize,k)
Parameters are as follows :
1. image : the source image in which we wish to find the corners (grayscale)
2. blocksize : size of the neighborhood in which we compare the gradient
3. ksize : aperture parameter for the Sobel() Operator (used for finding Ix & Iy)
4. k : Harris detector free parameter (used in computing R)
'''


def harris_corners(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    # You can play with these parameters to get different outputs
    corners_img = cv2.cornerHarris(gray_img, 3, 3, 0.04)

    image[corners_img > 0.001 * corners_img.max()] = [255, 255, 0]

    return image


'''
Function: cv2.goodFeaturesToTrack(image,maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
image – Input 8-bit or floating-point 32-bit (grayscale image).
maxCorners – You can specify the maximum no. of corners to be detected. (Strongest ones are returned if detected more than max.)
qualityLevel – Minimum accepted quality of image corners.
minDistance – Minimum possible Euclidean distance between the returned corners.
corners – Output vector of detected corners.
mask – Optional region of interest. 
blockSize – Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. 
useHarrisDetector – Set this to True if you want to use Harris Detector with this function.
k – Free parameter of the Harris detector (used in computing R)
'''


def shi_tomasi(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # You can play with these parameters to get different outputs
    corners_img = cv2.goodFeaturesToTrack(gray_img, 1200, 0.01, 10)
    # corners_img = np.int0(corners_img)

    blank_img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    for corners in corners_img:
        x, y = corners.ravel()
        cv2.circle(image, (x, y), 3, [255, 255, 0], -1)
        cv2.circle(blank_img, (x, y), 2, [255, 255, 0], -1)

    return image, blank_img


def find_corners(img):
    img_dup = cp.copy(img)
    img_dup1 = cp.copy(img)

    harris = harris_corners(img)
    shitomasi, silhouette = shi_tomasi(img_dup)

    # Display different corner detection methods side by side

    out1 = np.concatenate((harris, shitomasi), axis=1)
    out2 = np.concatenate((img_dup1, silhouette), axis=1)

    out3 = np.concatenate((out1, out2), axis=0)
    # cv2.imshow('Left: Harris, Right: Shi-Tomasi',out1)
    # cv2.imshow('Important points',out2)
    cv2.imshow('Corners', out3)
    return harris, shitomasi, silhouette, out3


# 3 Task Function
def houghCircles(img):
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if ((re.compile("3*")).match(cv2.__version__)):
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                                   1, 20, param1=70, param2=50, minRadius=0, maxRadius=0)
    else:
        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT,
                                   1, 20, param1=70, param2=50, minRadius=0, maxRadius=0)
    if circles is not None:
        if circles.any() == True :
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    return cimg


def eulerToCoordinateTransform(line):
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    return [(x1, y1), (x2, y2)]


def getIntersection(line_1, line_2):
    line1 = eulerToCoordinateTransform(line_1)
    line2 = eulerToCoordinateTransform(line_2)

    s1 = np.array(line1[0])
    e1 = np.array(line1[1])

    s2 = np.array(line2[0])
    e2 = np.array(line2[1])

    a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
    b1 = s1[1] - (a1 * s1[0])

    a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
    b2 = s2[1] - (a2 * s2[0])

    if abs(a1 - a2) < sys.float_info.epsilon:
        return False

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1
    return (x, y)


def houghLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1.75, np.pi / 180, 150)
    if lines is not None:
        if lines.any() == True :

            for line in lines:
                for rho, theta in line:
                    line_transform = eulerToCoordinateTransform(line)
                    cv2.line(img, line_transform[0], line_transform[1], (0, 0, 255), 2)

            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    if getIntersection(lines[i], lines[j]):
                        center = getIntersection(lines[i], lines[j])
                        if (math.isnan(center[0]) != True) or (math.isnan(center[0]) != True):
                            cv2.circle(img, (int(center[0]), int(center[1])), 2, (0, 255, 0), 3)
    return img

def main():
    imagepath = input("please.. typing image name : ")
    temp = imagepath[:imagepath.index(".")]
    check_path(imagepath)
    ### construct the argument parser and parse the arguments

    # loading image
    image = cv2.imread(imagepath)

    # Compute the ratio of the old height to the new height, clone it,
    # and resize it easier for compute and viewing
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    ### convert the image to grayscale, blur it, and find edges in the image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blurring to remove high frequency noise helping in
    # Contour Detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detection
    edged = cv2.Canny(gray, 75, 200)

    print("STEP 1: Edge Detection")
    # cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.imwrite('Edged-' + temp + '.jpg', edged)
    # finding the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ## What are Contours ?
    ## Contours can be explained simply as a curve joining all the continuous
    ## points (along the boundary), having same color or intensity.
    ## The contours are a useful tool for shape analysis and object detection
    ## and recognition.

    # Handling due to different version of OpenCV
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Taking only the top 5 contours by Area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    ### Heuristic & Assumption

    # A document scanner simply scans in a piece of paper.
    # A piece of paper is assumed to be a rectangle.
    # And a rectangle has four edges.
    # Therefore use a heuristic like : we’ll assume that the largest
    # contour in the image with exactly four points is our piece of paper to
    # be scanned.

    # looping over the contours
    for c in cnts:
        ### Approximating the contour

        # Calculates a contour perimeter or a curve length
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)  # 0.02

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        screenCnt = approx
        if len(approx) == 4:
            screenCnt = approx
            break

    # show the contour (outline)
    print("STEP 2: Finding Boundary")

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Boundary", image)
    cv2.imwrite('Boundary-' + temp + '.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.imread(imagepath)

    harris, shitomasi, silhoutte, out3 = find_corners(img)
    cv2.imwrite('corners-' + temp + '.jpg', out3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.imread(imagepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    houghCircleImage = houghCircles(gray)
    houghLineImage = houghLines(houghCircleImage)

    cv2.imshow('Circles and Lines', houghLineImage)
    cv2.imwrite('Circles-' + temp + '.jpg', houghLineImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
