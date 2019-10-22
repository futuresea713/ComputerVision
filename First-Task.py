import imutils
import cv2
import sys
import os



# Checks if the path exists

def img_check(path):
    if not os.path.exists(path):
        print('ERROR! The given path does not exist.')
        sys.exit(0)




def main():
    imagepath = input("please.. typing image name : ")
    temp = imagepath[:imagepath.index(".")]
    img_check(imagepath)

    # loading image
    image = cv2.imread(imagepath)

    # Compute the ratio of the old height to the new height, clone it,
    # and resize it easier for compute and viewing
    image = imutils.resize(image, height=500)

    ### convert the image to grayscale, blur it, and find edges in the image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blurring to remove high frequency noise helping in
    # Contour Detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny Edge Detection
    edged = cv2.Canny(gray, 75, 200)

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

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Boundary", image)
    cv2.imwrite('Boundary-' + temp + '.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
