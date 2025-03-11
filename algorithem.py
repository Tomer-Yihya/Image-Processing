import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pytesseract
import json
from heapq import nlargest
from PIL import Image
from datetime import datetime
import argparse




"""
Identify the sticker in the raw image and resize it
"""

def process_image(image_path):
    """Main function to detect and resize the yellow sticker"""
    
    # Verify file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    # Detect the yellow sticker
    rect, image = detect_yellow_sticker(image_path)
    if rect is None:
        raise ValueError("Could not detect yellow sticker")
    
    # Rotate and crop the sticker
    cropped = rotate_box(image, rect)
    
    # Calculate new height maintaining aspect ratio
    target_width = 900
    h, w = cropped.shape[:2]
    aspect_ratio = h / w
    new_height = int(target_width * aspect_ratio)
    
    # Resize the cropped image
    resized = cv2.resize(cropped, (target_width, new_height))
   
    return resized


def detect_yellow_sticker(image_path):
    """Detect and extract the main yellow sticker from the image"""
    # Read the image
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define broader yellow color range in HSV to capture all variations
    lower_yellow = np.array([15, 20, 120])  # Broader lower bound
    upper_yellow = np.array([50, 255, 255]) # Broader upper bound
    
    # Create mask for yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the sticker)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        return rect, image  # Returns (center(x,y), (width,height), angle), original image
    
    return None, image


def rotate_box(image, rect):
    """Rotate and crop the image to extract the sticker"""
    # Get the center, size, and angle from the rect
    center, (width, height), angle = rect
    
    # Normalize angle to make the rectangle horizontal
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
        
    # Clamp angle to ±20 degrees
    #angle = max(-20, min(20, angle))
    
    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((image.shape[1] * cos) + (image.shape[0] * sin))
    new_h = int((image.shape[1] * sin) + (image.shape[0] * cos))
    
    # Adjust the rotation matrix
    M[0, 2] += (new_w / 2) - image.shape[1] / 2
    M[1, 2] += (new_h / 2) - image.shape[0] / 2
    
    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
    
    # Calculate the new bounding box
    box = cv2.boxPoints(rect)
    #pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts = np.int64(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0
    
    # Get the bounds
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    
    # Crop the image
    cropped = rotated[y_min:y_max, x_min:x_max]
    
    return cropped





"""
Find all barcodes in the resized image
"""

def detect_barcodes_in_strip(image, strip_config, display_results=False):
    """
    Detect barcodes in a specific strip of the image, with adaptive detection.
    
    Args:
        image: The full image
        strip_config: Dictionary with strip configuration parameters
        display_results: Whether to display intermediate results
        
    Returns:
        list: List of detected barcode coordinates [(x, y, w, h), ...]
    """
    # Extract strip parameters
    y_start = strip_config['y_start']
    y_end = strip_config['y_end']
    window_width = strip_config['window_width']
    window_height = strip_config['window_height']
    step_size = strip_config.get('step_size', 10)
    min_score_threshold = strip_config.get('min_score', 30)  # Minimum score to be considered a barcode
    
    # Extract the strip
    strip = image[y_start:y_end, :]
    
    # Convert to grayscale
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple thresholding approaches
    # Standard Otsu thresholding
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # More conservative thresholding to reduce false positives
    threshold_value = 130  # Higher threshold to only catch true black pixels
    _, binary_aggressive = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Combine the binary images
    binary = cv2.bitwise_or(cv2.bitwise_or(binary_otsu, binary_adaptive), binary_aggressive)
    
    # Additional validation for strip 3 - check if there's actually any content
    if strip_config['id'] == 3:
        # Calculate overall black pixel density in the strip
        total_black = np.sum(binary > 0)
        total_pixels = binary.size
        black_percentage = (total_black / total_pixels) * 100
        
        # If the entire strip has very few black pixels, it's likely empty
        if black_percentage < 5:  # Very low black pixel percentage indicates empty strip
            print(f"Strip {strip_config['id']} appears empty (black pixel % = {black_percentage:.2f}%)")
            return []
    
    # Window scores dictionary
    window_scores = {}
    
    # Sliding window search across the entire strip
    for y in range(0, gray.shape[0] - window_height, step_size):
        for x in range(0, gray.shape[1] - window_width, step_size):
            # Extract window from binary image
            window = binary[y:y+window_height, x:x+window_width]
            
            # Calculate percentage of black pixels
            black_pixel_count = np.sum(window > 0)
            total_pixels = window_width * window_height
            black_pixel_percentage = (black_pixel_count / total_pixels) * 100
            
            # Skip windows with very low black pixel percentage
            if black_pixel_percentage < 15:
                continue
                
            # Apply Sobel filter to detect vertical edges (characteristic of barcodes)
            sobelx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
            sobelx = np.abs(sobelx)
            
            # Count strong vertical edges
            edge_threshold = np.max(sobelx) * 0.25
            edge_score = np.sum(sobelx > edge_threshold) / total_pixels * 100
            
            # Verify this is likely a barcode by checking edge pattern frequency
            # Barcodes have regular patterns of vertical edges
            if edge_score < 8:  # Minimum edge score to be considered a barcode
                continue
                
            # Combined score with weights
            combined_score = 0.7 * black_pixel_percentage + 0.3 * edge_score
            
            # Only store if the combined score is above threshold
            if combined_score > min_score_threshold:
                window_scores[(x, y)] = combined_score
    
    # If no candidates found, return empty list
    if not window_scores:
        return []
    
    # Get top candidates
    top_candidates = nlargest(20, window_scores.items(), key=lambda x: x[1])
    
    # Apply non-maximum suppression to avoid overlapping detections
    barcode_regions = []
    used_positions = set()
    
    # Adjust IoU threshold based on strip
    iou_threshold = 0.4  # Increased to prevent multiple detections of the same barcode
    
    for (x, y), score in top_candidates:
        # Check if this window significantly overlaps with any selected window
        overlap = False
        for fx, fy in used_positions:
            # Calculate IoU (Intersection over Union)
            x_overlap = max(0, min(x + window_width, fx + window_width) - max(x, fx))
            y_overlap = max(0, min(y + window_height, fy + window_height) - max(y, fy))
            
            if x_overlap > 0 and y_overlap > 0:
                overlap_area = x_overlap * y_overlap
                union_area = (window_width * window_height * 2) - overlap_area
                iou = overlap_area / union_area
                
                if iou > iou_threshold:
                    overlap = True
                    break
        
        if not overlap:
            # Add to regions and mark position as used
            barcode_regions.append((x, y + y_start, window_width, window_height, score))
            used_positions.add((x, y))
            
            # Limit to max_count barcodes per strip if specified
            max_count = strip_config.get('max_count', 3)
            if len(barcode_regions) >= max_count:
                break
    
    # Sort by score (highest first) and remove score from output
    barcode_regions.sort(key=lambda x: x[4], reverse=True)
    barcode_regions = [(x, y, w, h) for x, y, w, h, _ in barcode_regions]
    
    # Display strip-specific results if requested
    if display_results and barcode_regions:
        output_strip = strip.copy()
        
        # Draw detected barcode regions on the strip
        for i, (x, y_rel, w, h) in enumerate(barcode_regions):
            # y_rel is relative to the full image, so adjust for strip
            y_in_strip = y_rel - y_start
            cv2.rectangle(output_strip, (x, y_in_strip), (x + w, y_in_strip + h), (0, 255, 0), 2)
            cv2.putText(output_strip, f"Strip {strip_config['id']} #{i+1}", (x, y_in_strip - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(output_strip, cv2.COLOR_BGR2RGB))
        plt.title(f"Strip {strip_config['id']} Detected Barcodes: {len(barcode_regions)}")
        plt.tight_layout()
        plt.savefig(f"strip_{strip_config['id']}_barcodes.png")
    
    return barcode_regions


def detect_all_barcodes(image, display_results=False):
    """
    Detect barcodes in all strips of the image with adaptive detection.
    
    Args:
        image: The image to process
        display_results: Whether to display and save visualization
        
    Returns:
        dict: Dictionary with strip IDs and their detected barcode coordinates
    """
    # Define strip configurations with parameters optimized for detection, not counts
    strip_configs = [
        {
            'id': 1,
            'y_start': 0,
            'y_end': 190,
            'window_width': 170,
            'window_height': 40,
            'step_size': 10,
            'min_score': 35,    # Higher threshold to reduce false positives
            'max_count': 3      # Maximum expected barcodes
        },
        {
            'id': 2,
            'y_start': 190,
            'y_end': 500,
            'window_width': 170,
            'window_height': 50,
            'step_size': 5,
            'min_score': 35,
            'max_count': 3
        },
        {
            'id': 3,
            'y_start': 500,
            'y_end': image.shape[0],  # End of image
            'window_width': 350,
            'window_height': 90,
            'step_size': 10,
            'min_score': 30,    # Much higher threshold for the bottom strip
            'max_count': 1
        }
    ]
    
    # Process each strip
    all_barcodes = {}
    for config in strip_configs:
        strip_id = config['id']
        barcode_regions = detect_barcodes_in_strip(image, config, display_results)
        all_barcodes[strip_id] = barcode_regions
        #print(f"Strip {strip_id}: Found {len(barcode_regions)} barcodes")
    
    # Display results on the full image if requested
    if display_results:
        output = image.copy()
        
        # Color map for different strips
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
        
        # Draw detected barcode regions on the full image
        for strip_id, regions in all_barcodes.items():
            color = colors[(strip_id - 1) % len(colors)]
            for i, (x, y, w, h) in enumerate(regions):
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                cv2.putText(output, f"Strip {strip_id} #{i+1}", (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title('All Detected Barcodes')
        plt.tight_layout()
        plt.savefig('all_detected_barcodes.png')
        cv2.imwrite('all_detected_barcodes.jpg', output)
    
    return all_barcodes

# For debugging: draw bounding boxes on the image
def draw_bounding_boxes_on_image(image, detected_barcodes):
    """
    Draw bounding boxes for all detected barcodes on the image.
    
    Args:
        image: The image to draw on
        detected_barcodes: Dictionary with strip IDs and barcode locations
        
    Returns:
        image: The image with bounding boxes drawn
    """
    # Create a copy of the image to draw on
    output = image.copy()
    
    # Color map for different strips
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red
    
    # Draw all detected barcode regions
    for strip_id, regions in detected_barcodes.items():
        color = colors[(strip_id - 1) % len(colors)]
        for i, (x, y, w, h) in enumerate(regions):
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output, f"Strip {strip_id} #{i+1}", (x, y - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return output






"""
Extract the needed fields from the image using shifts relative to the barcodes
"""


def extract_text_field(image, roi_coords, config_str="--psm 7", lang="heb"):
    """
    Extract text from a region of interest in an image.
    
    Args:
        image: BGR image (OpenCV format)
        roi_coords: (x, y, w, h) bounding box
        config_str: Tesseract configuration string
        lang: Language for OCR (e.g., 'heb', 'eng')
        
    Returns:
        str: Extracted text (cleaned)
    """
    x, y, w, h = roi_coords
    
    # Extract the ROI from the original image
    roi = image[y:y + h, x:x + w]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Binarize (adaptive threshold)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    # (Optional) Upscale if text is small
    bin_img = cv2.resize(bin_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert array to PIL image
    pil_img = Image.fromarray(bin_img)

    # OCR with Tesseract
    text_raw = pytesseract.image_to_string(pil_img, config=config_str, lang=lang)
    #print(f"Raw Tesseract output for Case ID: '{text_raw}'")

    # Clean up the text
    text_clean = text_raw.strip()

    return text_clean


def fix_date_format(date_str):
    """
    Validate and fix date format to ensure it's dd.mm.yy
    
    Args:
        date_str: Raw extracted date string
        
    Returns:
        str: Validated date string in dd.mm.yy format
    """
    # Remove any leading digits if the first part has more than 2 digits
    parts = date_str.split('.')
    
    if len(parts) >= 1 and len(parts[0]) > 2:
        # Keep only the last 2 digits for the day
        parts[0] = parts[0][-2:]
    
    # Handle case where there might be extra digits in month or year parts
    if len(parts) >= 2 and len(parts[1]) > 2:
        parts[1] = parts[1][-2:]
    
    if len(parts) >= 3 and len(parts[2]) > 2:
        parts[2] = parts[2][-2:]
    
    # Ensure we have exactly 3 parts (day, month, year)
    while len(parts) < 3:
        parts.append('01')  # Default value
    
    # Reconstruct the date string
    fixed_date = '.'.join(parts[:3])
    
    return fixed_date


def keep_hebrew_and_spaces(text):
    """
    Keep only Hebrew letters and spaces, remove everything else.
    Also condenses multiple spaces into a single space.
    
    Args:
        text: Input text
        
    Returns:
        str: Filtered text with only Hebrew characters and spaces
    """
    # Allow Hebrew characters (U+0590–U+05FF) and whitespace
    filtered = re.sub(r"[^\u0590-\u05FF\s]", "", text)
    # Convert multiple spaces/newlines/tabs into a single space
    filtered = re.sub(r"\s+", " ", filtered)
    # Strip leading/trailing spaces
    return filtered.strip()


def extract_fields_from_barcode(image, barcode_location, strip_id, display=False):
    """
    Extract fields from the area around a detected barcode.
    
    Args:
        image: The full image
        barcode_location: Tuple of (x, y, width, height) for the barcode
        strip_id: The strip ID where the barcode was found (1, 2, or 3)
        display: Whether to display visualization of extracted fields
    
    Returns:
        dict: Dictionary containing the extracted fields
    """
    bx, by, bw, bh = barcode_location
    fields = {}
    
    # Shift vectors for different fields relative to barcode position
    if strip_id == 1:
        # First strip barcode shifts
        name_shift = (40, -62, 200, 30)  # (offset_x, offset_y, width, height)
        id_shift = (-20, -30, 130, 30)
        case_id_shift = (50, 37, 120, 26)
        
    elif strip_id == 2:
        # Second strip barcode shifts
        name_shift = (25, -200, 220, 45)
        id_shift = (-25, -138, 140, 37)
        date_shift = (-25, -101, 130, 36)
        case_id_shift = (-25, -42, 160, 41)
    
    elif strip_id == 3:
        # Third strip barcode shifts
        name_shift = (210, -168, 230, 37)
        id_shift = (520, -170, 190, 45)
        date_shift = (510, 45, 160, 35)
        case_id_shift = (0, -220, 145, 45)
        
    # Extract name (Hebrew)
    if strip_id in [1, 2, 3]:
        name_roi = (bx + name_shift[0], by + name_shift[1], name_shift[2], name_shift[3])
        #config_str = "--psm 7 --oem 1"
        config_str = "--psm 10 --oem 1"
        lang = "heb"
        extracted_name = extract_text_field(image, name_roi, config_str, lang)
        clean_txt = keep_hebrew_and_spaces(extracted_name)
        fields["name"] = clean_txt 
        
        """
        #debug - test various PSM's
        # Try different PSM modes
        
        print("DEBUG !!!")
        psm_modes = [6, 7, 8, 9, 10, 11, 13]
        for psm in psm_modes:
            config_str = f"--psm {psm} --oem 1"
            lang = "heb"
            extracted_name = extract_text_field(image, name_roi, config_str, lang)
            clean_txt = keep_hebrew_and_spaces(extracted_name)
                        
            print(f"  PSM {psm}: Raw: '{extracted_name}'")

        print("END DEBUG !!!")
        """
    
    # Extract person ID
    if strip_id in [1, 2, 3]:
        id_roi = (bx + id_shift[0], by + id_shift[1], id_shift[2], id_shift[3])
        #config_str = "--psm 10 -c tessedit_char_whitelist=0123456789"
        config_str = "--psm 7 -c tessedit_char_whitelist=0123456789"
        lang = "eng"
        extracted_id = extract_text_field(image, id_roi, config_str, lang)
        # Clean up the ID - keep only digits
        cleaned_id = ''.join(ch for ch in extracted_id if ch.isdigit())
        fields["person_id"] = cleaned_id
    
    # Extract date (only for strips 2 and 3)
    if strip_id in [2, 3]:
        date_roi = (bx + date_shift[0], by + date_shift[1], date_shift[2], date_shift[3])
        config_str = "--psm 7 -c tessedit_char_whitelist=0123456789."
        lang = "eng"
        extracted_date = extract_text_field(image, date_roi, config_str, lang)
        # Clean up the date - keep only digits and dots
        cleaned_date = ''.join(ch for ch in extracted_date if ch.isdigit() or ch == '.')
        # Validate and fix the date format (dd.mm.yy)
        cleaned_date = fix_date_format(cleaned_date)
        fields["date"] = cleaned_date
    else:
        # For strip 1, use current date
        current_date = datetime.now().strftime("%d.%m.%Y")
        fields["date"] = current_date

    # Extract case ID
    if strip_id in [1, 2, 3]:
        case_id_roi = (bx + case_id_shift[0], by + case_id_shift[1], case_id_shift[2], case_id_shift[3])
        
        """
        # Save the ROI for debugging
        roi_debug = image[by + case_id_shift[1]:by + case_id_shift[1] + case_id_shift[3], 
                        bx + case_id_shift[0]:bx + case_id_shift[0] + case_id_shift[2]]
        cv2.imwrite(f'case_id_roi_strip_{strip_id}.jpg', roi_debug)
        
        # Try different PSM modes
        psm_modes = [6, 7, 8, 9, 10, 11, 13]
        best_result = ""
        
        print(f"Trying different PSM modes for Case ID in strip {strip_id}:")
        for psm in psm_modes:
            config_str = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
            lang = "eng"
            extracted_case_id = extract_text_field(image, case_id_roi, config_str, lang)
            cleaned_case_id = ''.join(ch for ch in extracted_case_id if ch.isdigit())
            
            print(f"  PSM {psm}: Raw: '{extracted_case_id}', Cleaned: '{cleaned_case_id}'")
            
            # Keep the longest result (usually the most complete)
            if len(cleaned_case_id) > len(best_result):
                best_result = cleaned_case_id
        
        # Use the best result
        fields["case_id"] = best_result
        print(f"Selected case_id: '{best_result}'")
        """
    
    """
    # Extract case ID
    if strip_id in [1, 2, 3]:
        case_id_roi = (bx + case_id_shift[0], by + case_id_shift[1], case_id_shift[2], case_id_shift[3])
        config_str = "--psm 7 -c tessedit_char_whitelist=0123456789"
        #config_str = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789"
        lang = "eng"
        extracted_case_id = extract_text_field(image, case_id_roi, config_str, lang)
        # Clean up the case ID - keep only digits
        cleaned_case_id = ''.join(ch for ch in extracted_case_id if ch.isdigit())
        fields["case_id"] = cleaned_case_id
    """

    if display:
        # Create a visualization with all bounding boxes
        visualization_image = image.copy()
        
        # Draw barcode bounding box
        cv2.rectangle(visualization_image, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        
        # Define colors for different fields
        colors = {
            "name": (255, 0, 0),      # Blue
            "person_id": (0, 0, 255),  # Red
            "date": (255, 0, 255),     # Magenta
            "case_id": (0, 255, 255)   # Yellow
        }
        
        # Draw field ROIs using the already calculated ROIs
        # Name field
        x, y, w, h = name_roi
        cv2.rectangle(visualization_image, (x, y), (x + w, y + h), colors["name"], 2)
        cv2.putText(visualization_image, "Name", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["name"], 2)
        
        # Person ID field
        x, y, w, h = id_roi
        cv2.rectangle(visualization_image, (x, y), (x + w, y + h), colors["person_id"], 2)
        cv2.putText(visualization_image, "Person ID", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["person_id"], 2)
        
        # Date field (only for strips 2 and 3)
        if strip_id in [2, 3]:
            x, y, w, h = date_roi
            cv2.rectangle(visualization_image, (x, y), (x + w, y + h), colors["date"], 2)
            cv2.putText(visualization_image, "Date", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["date"], 2)
        
        # Case ID field
        x, y, w, h = case_id_roi
        cv2.rectangle(visualization_image, (x, y), (x + w, y + h), colors["case_id"], 2)
        cv2.putText(visualization_image, "Case ID", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["case_id"], 2)
        
        # Display the visualization
        plt.figure(figsize=(12, 12))
        plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Extracted Fields from Strip {strip_id}')
        plt.axis('on')  # Show axis to help with coordinate reference
        plt.tight_layout()
        plt.savefig(f'strip_{strip_id}_fields.png')
        plt.show()

    return fields


def process_barcodes_and_extract_fields(image, barcodes_dict, display=False):
    """
    Process the detected barcodes and extract fields according to priority order.
    
    Args:
        image: The image to process
        barcodes_dict: Dictionary with strip IDs as keys and lists of barcode locations as values
        display: Whether to display and save visualization of extracted fields
    
    Returns:
        dict: JSON-serializable dictionary with extracted fields
    """
    # Priority order: Strip 2 (Middle) > Strip 3 (Bottom) > Strip 1 (Top)
    priority_order = [2, 3, 1]
    #priority_order = [1, 3, 2]

    # Find the first available barcode according to priority
    selected_barcode = None
    selected_strip_id = None
    
    for strip_id in priority_order:
        if strip_id in barcodes_dict and barcodes_dict[strip_id]:
            # Take the first barcode from this strip
            selected_barcode = barcodes_dict[strip_id][0]
            selected_strip_id = strip_id
            #print(f"Selected barcode from strip {selected_strip_id}: {selected_barcode}")
            break
    
    # If no barcode found, return empty result
    if selected_barcode is None:
        return {"error": "No barcodes detected in the image"}
    
    # Extract fields from the selected barcode
    fields = extract_fields_from_barcode(image, selected_barcode, selected_strip_id, display)
    
    # Add metadata about which barcode was used
    fields["strip_id"] = selected_strip_id

    return fields


def save_fields_to_json(fields, output_path):
    """
    Save extracted fields to a JSON file.
    
    Args:
        fields: Dictionary with extracted fields
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)
    print(f"Fields saved to {output_path}")








def extract_text_as_json(image_path):
    """
    Extracts data from the given image path and returns it as a JSON string.
    """

    # Step 1: Detect and resize the yellow sticker
    resized = process_image(image_path)

    # Step 2: Find all barcodes in the resized image
    detected_barcodes = detect_all_barcodes(resized, display_results=False)

    # Step 3: Process barcodes and extract fields
    extracted_fields = process_barcodes_and_extract_fields(resized, detected_barcodes, display=False)

    # Convert to JSON format
    json_output = json.dumps(extracted_fields, ensure_ascii=False, indent=2)
    
    return json_output











#!/usr/bin/env python3
"""
Yellow Sticker Barcode Detection and Field Extraction Script
This script processes an image containing a yellow sticker with barcodes,
detects the barcodes, and extracts relevant patient fields.

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Detect barcodes and extract fields from medical stickers')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    parser.add_argument('--save', type=str, help='Path to save the extracted fields as JSON (optional)')
    args = parser.parse_args()
    
    # Step 1: Detect and resize the yellow sticker
    resized = process_image(args.image_path)
    
    # Step 2: Find all barcodes in the resized image
    detected_barcodes = detect_all_barcodes(resized, display_results=False)
    
    # Step 3: Process barcodes and extract fields
    extracted_fields = process_barcodes_and_extract_fields(resized, detected_barcodes, display=False)
    
    # Reverse Hebrew names for proper right-to-left display
    extracted_fields['name'] = extracted_fields['name'][::-1]
               
    # Print the extracted fields as JSON
    print(json.dumps(extracted_fields, ensure_ascii=False, indent=2))
    
    # save the results to a file
    save_fields_to_json(extracted_fields, "output.json")    


if __name__ == "__main__":
    main()

"""