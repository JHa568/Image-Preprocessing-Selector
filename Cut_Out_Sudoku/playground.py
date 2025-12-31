import cv2 as cv 
import numpy as np

images_loc = "../images/"

img_w = 450
img_h = 450


def create_image_0():
    img = cv.imread(images_loc + 'processed_sudoku_dim.png')
    img = cv.resize(img, (img_w, img_h))
    # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img


def create_image_1():
    img = cv.imread(images_loc + 'sudoku_dark.jpg')
    img = cv.resize(img, (img_w, img_h))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def create_image_2():
    img = cv.imread(images_loc + 'sudoku.jpg')
    img = cv.resize(img, (img_w, img_h))
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def adjust_brightness(image):
    #print("Raw_image:", image)
    matrix = np.asmatrix(image)
    #print("Matrix:", matrix)
    mean_brightness = matrix.mean()
    print("Average bright values:", mean_brightness)
    
def adjust_saturation_value(image):
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_image)
    sat_factor = 1.5
    value_factor = 50
    s = np.clip(s + sat_factor, 0, 255).astype(np.uint8)
    v = np.clip(v + value_factor, 0, 255).astype(np.uint8) 
    mod_hsv_image =  cv.merge([h, s, v])
    bgr_image_modified = cv.cvtColor(mod_hsv_image, cv.COLOR_HSV2BGR)
    return bgr_image_modified

if __name__ == "__main__":
    image_0 = create_image_0()
    mod_img = adjust_saturation_value(image_0) # dim - 7 aperture
    cv.imshow("original", image_0)
    cv.imshow("Adjust Saturation", mod_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
