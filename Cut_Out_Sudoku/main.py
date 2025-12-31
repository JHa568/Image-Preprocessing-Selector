import cv2 as cv
import numpy as np
import math 

images_loc = "../images/"

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

### Computer vision Tune Paramter ####
low_canny_thresh = 0
high_canny_thresh = 0
aperture_size = 7 # 3

BRIGHTNESS_THRESHOLD_LOWER = 20
BRIGHTNESS_THRESHOLD_MID = 127
BRIGHTNESS_THRESHOLD_HIGHER = 235
BRIGHTNESS_DIFF = 30

def create_image():
    img = cv.imread(images_loc + 'Skewed_Real.jpg')
    img = cv.resize(img, (img_w, img_h))
    # norm = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    # blurred = cv.GaussianBlur(norm, (9, 9), 0)
    # 1. Global normalization
    norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)

    # 2. CLAHE for local contrast boost
    lab = cv.cvtColor(norm, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv.merge((l_clahe, a, b))
    result = cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)
    return result

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
        return 5

def filter_image(image):
    # Canny edge detection
    new_img = cv.cvtColor(image.copy(), cv.COLOR_BGR2GRAY)
    aperture_size = aperture_adjustment(new_img)
    print("Aperture Size:", aperture_size)
    if (aperture_size < 0):
        return None, False
    
    thresh = cv.cvtColor(new_img, cv.COLOR_BGR2RGB)
    thresh = cv.Canny(thresh, low_canny_thresh, high_canny_thresh, apertureSize=aperture_size)
    #cv.imshow("Thresh", thresh)
    return thresh, True

def find_contour(thresh):
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    lines = cv.HoughLines(thresh, 0.7, np.pi / 180, 115, None, 100, 100)
    print(lines)

    return contours, lines
    
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left

    return rect

def cut_off(image, contours):
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    largest = max(contours, key=cv.contourArea)
    
    mask = np.zeros_like(gray) # return the same array size full of zeroes  
    cv.drawContours(mask, [largest], -1, 255, -1)  # filled white shape

    # Apply mask to original image
    result = cv.bitwise_and(gray, gray, mask=mask)

    # Optional: Crop the bounding box
    x, y, w, h = cv.boundingRect(largest)

    cropped = result[y:y+h, x:x+w]
    
    return cropped, largest 

def realignment(normal_img, largest):
    # Realign the sudoku puzzle
    peri = cv.arcLength(largest, True)
    approx = cv.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) == 4:
        sudoku_corners = approx.reshape(4, 2)
    else:
        print("Couldn't find 4 corners!")
        exit()
        
    ordered_corners = order_points(sudoku_corners)
    widthA = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    widthB = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(ordered_corners[1] - ordered_corners[2])
    heightB = np.linalg.norm(ordered_corners[0] - ordered_corners[3])
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(ordered_corners, dst)
    warped = cv.warpPerspective(normal_img, M, (maxWidth, maxHeight))
    warped = cv.resize(warped, (img_w, img_h))
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    return warped

def get_contour_cells(non_skewed_image):
    copy_original_skewed = non_skewed_image.copy()
    thresh = cv.cvtColor(copy_original_skewed, cv.COLOR_GRAY2BGR)
    #thresh = non_skewed_image # cv.resize(non_skewed_image, (img_w, img_h))
    gray = copy_original_skewed
    #### Parameters
    dilate_ksize = 2
    erode_ksize = 1
    open_ksize = 12
    close_ksize = 4
    grad_ksize = 2
    tophat_ksize = 1
    blackhat_ksize = 1
    aperture_size = 3
    low_canny_thresh = 176
    high_canny_thresh = 75
    ####
    
    median_val = np.median(gray)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    print(f"Lower: {lower} | Upper: {upper}")
    # TODO: Add the histogram to normalise brightness level on entire image
    # Canny Edge detection 
    thresh = cv.Canny(thresh, low_canny_thresh, high_canny_thresh, apertureSize=aperture_size)
    
    # Dialate
    kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (dilate_ksize *dilate_ksize, 1))
    kernel_v = cv.getStructuringElement(cv.MORPH_RECT, (1, dilate_ksize *dilate_ksize))
    h = cv.dilate(thresh, kernel_h, iterations=1)
    v  = cv.dilate(thresh, kernel_v, iterations=1)
    thresh = cv.add(h, v)
    
    # Erode
    # kernel_h = cv.getStructuringElement(cv.MORPH_RECT, (erode_ksize * erode_ksize, 1))
    # kernel_v = cv.getStructuringElement(cv.MORPH_RECT, (1, erode_ksize * erode_ksize))
    # h = cv.erode(thresh, kernel_h, iterations=1)
    # v = cv.erode(thresh, kernel_v, iterations=1)
    # thresh = cv.add(h, v)
    
    # Open
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (open_ksize * open_ksize, 1))
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, open_ksize * open_ksize))
    h = cv.morphologyEx(thresh, cv.MORPH_OPEN, h_kernel)
    v = cv.morphologyEx(thresh, cv.MORPH_OPEN, v_kernel)
    thresh = cv.add(h, v)
        
    # Close
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (close_ksize * close_ksize, 1))
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, close_ksize * close_ksize))
    h = cv.morphologyEx(thresh, cv.MORPH_CLOSE, h_kernel)
    v = cv.morphologyEx(thresh, cv.MORPH_CLOSE, v_kernel)
    thresh = cv.add(h, v)
    
    # Gradient
    kernel = np.ones((grad_ksize, grad_ksize), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_GRADIENT, kernel)

    # Tophat
    # kernel = np.ones((tophat_ksize, tophat_ksize), np.uint8)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_TOPHAT, kernel)
    
    # # blackhat
    # kernel = np.ones((blackhat_ksize, blackhat_ksize), np.uint8)
    # thresh = cv.morphologyEx(thresh, cv.MORPH_BLACKHAT, kernel)
    
    contours_cells = find_contour(thresh)
    cv.imshow("thresh", thresh)
    
    image_for_contours = cv.cvtColor(copy_original_skewed, cv.COLOR_GRAY2BGR)  # Convert to BGR for contour drawing
    contour_img = cv.drawContours(image_for_contours, contours_cells, -1, (0, 255, 0), 1)
    cv.imshow("original", contour_img)
    areas = [cv.contourArea(c) for c in contours_cells]
    print(areas)
    average_area = sum(areas) / len(areas) if areas else 0
    
    tolerance = 0.0  # e.g., allow ±30% variation
    min_area = average_area * (1 - tolerance)
    max_area = average_area * (1 + tolerance)
    print(f"min area: {min_area}\n max area: {max_area}")
    filtered = []
    for c in contours_cells:
        if cv.contourArea(c) >= 2:
            filtered.append(c)
    
    print("Contour cells:", len(filtered))
    return filtered

def sort_contours_into_grid(contours, grid_size=9):
    # Step 1: Sort contours top-to-bottom
    bounding_boxes = [cv.boundingRect(c) for c in contours]
    contours_with_boxes = sorted(zip(contours, bounding_boxes), key=lambda b: b[1][1])  # sort by y (top)

    # Step 2: Group into rows and sort left-to-right
    row_step = len(contours) // grid_size
    rows = []
    for i in range(0, len(contours_with_boxes), row_step):
        row = contours_with_boxes[i:i + row_step]
        row_sorted = sorted(row, key=lambda b: b[1][0])  # sort by x (left)
        rows.append([c for c, _ in row_sorted])

    return rows  # rows[row][col] → contour

def normalize_for_grid_detection(image, debug=False):
    """
    Preprocess a 300x300 Sudoku image to enhance faint and broken grid lines.
    Returns a binary mask with connected horizontal and vertical lines.
    """
    # 1. Convert to grayscale
    gray = image #cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 2. Apply CLAHE to normalize brightness and contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # Smaller grid for smaller image
    enhanced = clahe.apply(gray)

    # 3. Slight Gaussian blur to reduce noise
    blurred = cv.GaussianBlur(enhanced, (3, 3), 0)  # Smaller kernel
    
    
    
    
    test = cv.bitwise_not(blurred)
    _, thresh = cv.threshold(test,110,255,cv.THRESH_BINARY)
    
    # 4. Adaptive threshold for better local contrast
    binary = cv.adaptiveThreshold(
        thresh, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11, 2  # Reduced constant for smaller features
    )

    # kernel = np.ones((3, 3), np.uint8)
    # closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)

    # cv.imshow("Blured", thresh)
    cv.imshow("Bin", thresh)
    # test = remove_small_components(thresh, min_area=10)
    
    # # Then remove tiny specks with opening
    # kernel_small = np.ones((2, 2), np.uint8)
    # opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel_small, iterations=1)

    # 5. Morphological ops to connect broken lines (scaled for 300x300)
    
    # a. Vertical line enhancement
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 9))
    vertical_lines = cv.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv.dilate(vertical_lines, vertical_kernel, iterations=2)

    # b. Horizontal line enhancement
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 1))
    horizontal_lines = cv.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv.dilate(horizontal_lines, horizontal_kernel, iterations=2)

    # c. Combine vertical and horizontal lines
    grid_mask = cv.bitwise_or(vertical_lines, horizontal_lines)
    
    cv.imshow("AT", grid_mask)
    _, thresh = cv.threshold(grid_mask,250,255,cv.THRESH_BINARY)
    test = remove_small_components(thresh, min_area=20000)
    cv.imshow("thresh", test)
    
    return grid_mask

def remove_small_components(binary_img, min_area=100):
    """Remove small components based on area threshold."""
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary_img, connectivity=8)
    cleaned = np.zeros_like(binary_img)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned

def get_individual_cells(processed, image):
    blurred = cv.GaussianBlur(processed, (5, 5), 0)
    # gray = cv.bitwise_not(processed)
    
    _, clean_grid = cv.threshold(blurred,200,255,cv.THRESH_BINARY)
    # cv.imshow("bitwised", clean_grid)
    # cv.imshow("processed", processed)
    # clean_grid = cv.Canny(thresh, 200, 255)
    # cv.imshow("Edge", clean_grid)

    #clean_grid = remove_small_components(edges, min_area=0)
    
    
    # Hough Line Transform
    lines = cv.HoughLinesP(clean_grid, 1, np.pi / 180, threshold=130, minLineLength=100, maxLineGap=100)

    line_mask = np.zeros_like(clean_grid)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi)
            if angle < 10 or abs(angle - 90) < 10:  # horizontal or vertical
                cv.line(line_mask, (x1, y1), (x2, y2), 255, 1)

    clean_grid = line_mask
    cv.imshow("Clean grid", clean_grid)
    h, w = image.shape[:2]
    cell_h, cell_w = h // 9, w // 9

    cells = []
    cell_images = []
    for i in range(9):
        row = []
        for j in range(9):
            x1 = j * cell_w
            y1 = i * cell_h
            x2 = (j + 1) * cell_w
            y2 = (i + 1) * cell_h
            cell = image[y1:y2, x1:x2]
            row.append(cell)
        cell_images.append(row)
        cells.append(np.hstack(row))

    mosaic = np.vstack(cells)
    print("Total cells:", len(cell_images[0]) * len(cell_images))
    cv.imshow("mosaic", mosaic)
    return cell_images

def number_recognition(cell):
    number = -1
    
    return number
    
def digitise_sudoku_image(cell_images):
    grid_size = 9
    sudoku_matrix = [['*'] * grid_size] * grid_size
    
    for i in range(0, grid_size):
        for j in range(0, grid_size):
            # 1. Detect the number in the cell image
            # 2. Check whether it is a character between 0 - 9
            # 3. If so modify the array to have the correct position 
            # 4. else move to the next one 
            break
        
    return sudoku_matrix
    
if __name__ == "__main__":
    original = create_image()
    copy_original = original.copy()
    fil_image, error = filter_image(copy_original)
    
    if error == False:
        print("fil_image is None")
    else:
        contours, lines = find_contour(fil_image)
        processed_image, contour = cut_off(copy_original, contours)
        non_skewed_image = realignment(copy_original, contour)
        norm_grid = normalize_for_grid_detection(non_skewed_image)
        cell_images = get_individual_cells(norm_grid, non_skewed_image)
        #digital_sudoku = digitise_sudoku_image(cell_images)
        cv.imshow("first cell", cell_images[7][0])
        #cv.imshow("non-skew", non_skewed_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
