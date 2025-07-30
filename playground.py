import cv2 as cv
import numpy as np

images_loc = "./images/"

def filter_image(image):
    gray_scale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # blurred = cv.GaussianBlur(gray_scale, (7, 7), 0)
    th3 = cv.adaptiveThreshold(gray_scale, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    kernel = np.ones((2,2), np.uint8)
    #img_erode = cv.erode(th3, kernel, iterations=1)
    img_dilation = cv.dilate(th3, kernel, iterations=1)

    opening = cv.morphologyEx(img_dilation, cv.MORPH_OPEN, kernel)
    
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN,np.ones((1,1), np.uint8))
    img_erode = cv.erode(opening, np.ones((3,3), np.uint8), iterations=2)
    
    return img_erode

def ocr_function(image):
    # Placeholder for OCR functionality
    # This function would typically use an OCR library to read text from the image
    pass

def cv_function():
    # Load the image
    image = cv.imread(images_loc + 'sudoku_dim.jpg')

    # Check if image is loaded correctly
    if image is None:
        print("Error: Image not found!")
    else:
        # Display the original image
        #cv.imshow('Original Image', image)
        f_image = filter_image(image)
        contours, hierarchy = cv.findContours(image=f_image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
      
        image_copy = image.copy()
        cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
        cv.imshow('Edge Image', f_image)
        # cv.imshow('Adaptive Threshold Image', image_copy)
        

if __name__ == "__main__":
    cv_function()
    cv.waitKey(0)  # Wait for a key press to close the window
    cv.destroyAllWindows()  # Close all OpenCV windows