import cv2 as cv
import numpy as np

images_loc = "./images/"

### Current issues
"""_TODO_
If the edges and break lines of the puzzle
are the same colour as the background 
the solve / computer vision algorithm will FAIL.
"""

# TODO: Adjust this based on how far away the puzzle is from the camera
# TODO: Reduce the img size if close. 
# TODO: Increase the img size if far.

img_w = 300
img_h = 300

upscale_img_w = 500
upscale_img_h = 500

### Computer vision Tune Paramter ####
low_canny_thresh = 0
high_canny_thresh = 0
aperture_size = 7 # 3

BRIGHTNESS_THRESHOLD_LOWER = 20
BRIGHTNESS_THRESHOLD_MID = 127
BRIGHTNESS_THRESHOLD_HIGHER = 235
BRIGHTNESS_DIFF = 30

def create_image():
    img = cv.imread(images_loc + 'COMP5.jpg', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (img_w, img_h))
    return img

def aperture_adjustment(image):
    # Calculate the final aperture value 
    # based off of the average brightness values in the image
    matrix = np.asmatrix(image.copy())
    mean_brightness = matrix.mean()
    print("Mean Brightness:", mean_brightness)
    if (mean_brightness > BRIGHTNESS_THRESHOLD_LOWER + BRIGHTNESS_DIFF and
        mean_brightness < BRIGHTNESS_THRESHOLD_HIGHER - BRIGHTNESS_DIFF):
        return 5
    elif (mean_brightness <= BRIGHTNESS_THRESHOLD_LOWER + BRIGHTNESS_DIFF and
          mean_brightness >= BRIGHTNESS_THRESHOLD_LOWER):
        return 7
    elif (mean_brightness >= BRIGHTNESS_THRESHOLD_HIGHER - BRIGHTNESS_DIFF and
          mean_brightness <= BRIGHTNESS_THRESHOLD_HIGHER):
        return 3
    else:
        return -1

def filter_image(image):
    # Canny edge detection
    #new_img = cv.cvtColor(image.copy(), cv.COLOR_GRAY2BGR)
    aperture_size = aperture_adjustment(image)
    print("Aperture Size:", aperture_size)
    if (aperture_size < 0):
        return None, False
    
    thresh = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    thresh = cv.Canny(thresh, low_canny_thresh, high_canny_thresh, apertureSize=aperture_size)
    
    return thresh, True

def find_contour(thresh):
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours
    
def cut_off(image, contours):
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    largest = max(contours, key=cv.contourArea)
    
    mask = np.zeros_like(image) # return the same array size full of zeroes  
    cv.drawContours(mask, [largest], -1, 255, -1)  # filled white shape

    # Apply mask to original image
    result = cv.bitwise_and(image, image, mask=mask)

    # Optional: Crop the bounding box
    x, y, w, h = cv.boundingRect(largest)
    cropped = result[y:y+h, x:x+w]
    
    print(f"x: {x}, y: {y}, w: {w}, h: {h}")

    cropped = cv.resize(cropped, (img_w, img_h))

    _, thresh = cv.threshold(cropped, 0, 255, cv.THRESH_BINARY)
    
    contours, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    
    cnt = max(contours, key=cv.contourArea)

    epsilon = 0.02 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    color_image = cv.cvtColor(cropped, cv.COLOR_GRAY2BGR)
    hieght, width, _ = color_image.shape

    pts1 = []
    pts2 = [[0,0],[0,img_h],[img_w,img_h],[img_w,0]]
    
    arrange = [1,2,3,4]
    count = 1
    for point in approx:
        x, y = point[0]
        if count == 1:
            cv.circle(color_image, (x, y), 2, (0, 0, 255), -1)
        if count == 2:
            cv.circle(color_image, (x, y), 2, (0, 255, 0), -1)
        if count == 3:
            cv.circle(color_image, (x, y), 2, (255, 0, 0), -1)
        if count == 4:
            cv.circle(color_image, (x, y), 2, (255, 0, 255), -1)

        pts1.append([x,y])
        count += 1

    print(f"width {width}, hieght {hieght}")
    print(f"pts1 {pts1}, pts2 {pts2}")
    cv.imshow("thresh_image", thresh)
    cv.imshow("rotated_image", color_image)

    M = cv.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    
    dst = cv.warpPerspective(color_image, M, (img_w,img_h))
    cv.imshow("warped imaged", dst)
    return dst

def realignment(cropped_img):
    # Realign the sudoku puzzle
    
    return None
    
if __name__ == "__main__":
    original = create_image()
    copy_original = original.copy()
    fil_image, error = filter_image(copy_original)
    
    if error == False:
        print("fil_image is None")
    else:
        contours = find_contour(fil_image)
        processed_image = cut_off(copy_original, contours)
        
        image_for_contours = cv.cvtColor(copy_original, cv.COLOR_GRAY2BGR)  # Convert to BGR for contour drawing
        contour_img = cv.drawContours(image_for_contours, contours, -1, (0, 255, 0), 1)
        # cv.imshow("processed_image", cv.resize(processed_image, (upscale_img_w, upscale_img_h)))
        cv.waitKey(0)
        cv.destroyAllWindows()
        