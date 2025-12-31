import cv2
import numpy as np

images_loc = "../images/"

import cv2
import numpy as np

def normalize_for_grid_detection(image, debug=False):
    """
    Preprocess a 300x300 Sudoku image to enhance faint and broken grid lines.
    Returns a binary mask with connected horizontal and vertical lines.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Apply CLAHE to normalize brightness and contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # Smaller grid for smaller image
    enhanced = clahe.apply(gray)

    # 3. Slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)  # Smaller kernel

    # 4. Adaptive threshold for better local contrast
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2  # Reduced constant for smaller features
    )

    # 5. Morphological ops to connect broken lines (scaled for 300x300)

    # a. Vertical line enhancement
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)

    # b. Horizontal line enhancement
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)

    # c. Combine vertical and horizontal lines
    grid_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)
    
    return grid_mask

def remove_small_components(binary_img, min_area=100):
    """Remove small components based on area threshold."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    cleaned = np.zeros_like(binary_img)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned

# Load and preprocess image
image = cv2.imread(images_loc + 'processed_test2.png')
image = cv2.resize(image, (300, 300))
processed = normalize_for_grid_detection(image)

blurred = cv2.GaussianBlur(processed, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
cv2.imshow("original", image)
cv2.imshow("Edge", edges)

clean_grid = remove_small_components(edges, min_area=100)

# # Hough Line Transform
lines = cv2.HoughLinesP(clean_grid, 1, np.pi / 180, threshold=130, minLineLength=100, maxLineGap=50)

line_mask = np.zeros_like(clean_grid)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi)
        if angle < 10 or abs(angle - 90) < 10:  # horizontal or vertical
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

clean_grid = line_mask
cv2.imshow("Clean grid", clean_grid)
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

# Optional: Show first cell
cv2.imshow("Single cell", cell_images[3][8])
#cv2.imshow("Sudoku Cells Mosaic", mosaic)
cv2.waitKey(0)
cv2.destroyAllWindows()
