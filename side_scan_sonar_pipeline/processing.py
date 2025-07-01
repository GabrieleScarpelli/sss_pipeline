import numpy as np
from scipy.optimize import curve_fit
import cv2

def model_function(x, a, b, c):
    return a / ((x + b)**2) + c


def fit_inverse_square_model(y_input):
    x = np.arange(len(y_input))
    initial_guess = [1.0, 1.0, 0.0]

    try:
        # Fit the model
        params, _ = curve_fit(model_function, x, y_input, p0=initial_guess, maxfev=10000)
        a, b, c = params
        y_fitted = model_function(x, a, b, c)

        # Define the derived function
        def derived_function(x_val):
            x_val = np.asarray(x_val)
            numerator = c + a / (b**2)
            denominator = a / ((x_val + b)**2) + c
            return numerator / denominator

        return y_fitted, derived_function(x)

    except RuntimeError:
        print("Fit did not converge.")
        return np.full_like(y_input, np.nan), lambda x: np.nan
    

def get_min_area_rect(image_mask):
    """
    Takes a binary mask with non-zero pixels in the insonified region.
    Returns the rotated bounding box and the rotation matrix to deskew it.
    """
    # Get the coordinates of non-zero pixels
    coords = np.column_stack(np.where(image_mask > 0))
    
    # Convert to (x, y) format
    coords = np.flip(coords, axis=1)  # From (row, col) to (x, y)
    
    # Get the rotated bounding box
    rect = cv2.minAreaRect(coords.astype(np.float32))  # (center (x,y), (width, height), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    return rect, box


def rotate_and_crop(image, rect, crop_out_percentage=0.05):
    """
    Rotates the image so the min-area rect is axis-aligned and crops the rect region.
    """
    center, size, angle = rect
    cx, cy = center
    w, h = size

    # Get the rotation matrix for the original center
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # Compute the new bounding size
    h_img, w_img = image.shape[:2]
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h_img * sin + w_img * cos)
    new_h = int(h_img * cos + w_img * sin)

    # Adjust the transformation matrix to shift the image
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy

    # Apply warpAffine with expanded size
    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    # Compute the new box center after shifting
    new_center = (new_w // 2, new_h // 2)
    box = cv2.boxPoints(((new_center[0], new_center[1]), size, 0))  # angle now zero
    box = np.intp(box)

    # Crop the rotated rectangle
    x, y, w, h = cv2.boundingRect(box)
    # reduce the crop size by crop_out_percentage
    x = int(x + w * crop_out_percentage)
    y = int(y + h * crop_out_percentage)
    w = int(w * (1 - 2 * crop_out_percentage))
    h = int(h * (1 - 2 * crop_out_percentage))
    cropped = rotated[y:y+h, x:x+w]

    #print(f"Center: ({cx}, {cy}), Width: {w}, Height: {h}, Angle: {angle}")
    #print(f"x: {x}, y: {y}, w: {w}, h: {h}")

    return cropped, rotated
