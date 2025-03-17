import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
import json
from heapq import nlargest
from PIL import Image
from datetime import datetime
import sympy
import math
from pyzbar import pyzbar
from pyzbar.pyzbar import decode
from typing import List, Tuple
import shutil




def appx_best_fit_ngon(mask, n: int = 4) -> List[Tuple[int, int]]:

    # convex hull of the input mask
    # mask_cv2_gray = cv2.cvtColor(mask_cv2, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(contours[0])
    hull = np.array(hull).reshape((len(hull), 2))

    # to sympy land
    hull = [sympy.Point(*pt) for pt in hull]

    # run until we cut down to n vertices
    while len(hull) > n:
        best_candidate = None

        # for all edges in hull ( <edge_idx_1>, <edge_idx_2> ) ->
        for edge_idx_1 in range(len(hull)):
            edge_idx_2 = (edge_idx_1 + 1) % len(hull)

            adj_idx_1 = (edge_idx_1 - 1) % len(hull)
            adj_idx_2 = (edge_idx_1 + 2) % len(hull)

            edge_pt_1 = sympy.Point(*hull[edge_idx_1])
            edge_pt_2 = sympy.Point(*hull[edge_idx_2])
            adj_pt_1 = sympy.Point(*hull[adj_idx_1])
            adj_pt_2 = sympy.Point(*hull[adj_idx_2])

            subpoly = sympy.Polygon(adj_pt_1, edge_pt_1, edge_pt_2, adj_pt_2)
            angle1 = subpoly.angles[edge_pt_1]
            angle2 = subpoly.angles[edge_pt_2]

            # we need to first make sure that the sum of the interior angles the edge
            # makes with the two adjacent edges is more than 180°
            if sympy.N(angle1 + angle2) <= sympy.pi:
                continue

            # find the new vertex if we delete this edge
            adj_edge_1 = sympy.Line(adj_pt_1, edge_pt_1)
            adj_edge_2 = sympy.Line(edge_pt_2, adj_pt_2)
            intersect = adj_edge_1.intersection(adj_edge_2)[0]

            # the area of the triangle we'll be adding
            area = sympy.N(sympy.Triangle(edge_pt_1, intersect, edge_pt_2).area)
            # should be the lowest
            if best_candidate and best_candidate[1] < area:
                continue

            # delete the edge and add the intersection of adjacent edges to the hull
            better_hull = list(hull)
            better_hull[edge_idx_1] = intersect
            del better_hull[edge_idx_2]
            best_candidate = (better_hull, area)

        if not best_candidate:
            raise ValueError("Could not find the best fit n-gon!")

        hull = best_candidate[0]

    # back to python land
    hull = [(int(x), int(y)) for x, y in hull]

    return hull



def create_debug_dir(base_path):
    """
    Create a directory for debug images based on original image name
    
    Args:
        base_path: Path to the folder containing the original image
        image_name: Name of the original image file
    
    Returns:
        Path to the debug directory
    """
    # Remove extension from image name
    
    # Create debug directory
    debug_dir = os.path.join(base_path, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    return debug_dir



def understandSticker(image, base_path=None, image_name=None):
    """Main function to detect and resize the yellow sticker"""
    debug_dir = create_debug_dir(base_path)

    # yellow or white type
    #print('-> Understarnd image type ')
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     
    # Define colors broader yellow color range in HSV to capture all variations
    lower_yellow_sticker = np.array([20, 120, 0])  
    upper_yellow_sticker = np.array([50, 255, 255]) 
    
    lower_white_sticker = np.array([0, 0, 150])  
    upper_white_sticker = np.array([255, 60, 255]) 
    
    lower_bg = np.array([15, 40, 40])  
    upper_bg = np.array([40, 90, 200]) 
    
    # Create mask for all  regions
    yellowStickerMask = cv2.inRange(hsv, lower_yellow_sticker, upper_yellow_sticker)
    whiteStickerMask = cv2.inRange(hsv, lower_white_sticker, upper_white_sticker)
    yellowBGMask = cv2.inRange(hsv, lower_bg, upper_bg)
    #print('base_path dir - ' ,base_path)
    
    
    name = os.path.splitext(image_name)[0]
    
    # cv2.imwrite(os.path.join(debug_dir, name +"_yellowMask.jpg"), yellowStickerMask)
    # cv2.imwrite(os.path.join(debug_dir, name + "_whiteMask.jpg"), whiteStickerMask)
    # cv2.imwrite(os.path.join(debug_dir, name + "_BGMask.jpg"), yellowBGMask)
    
  
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    maskY = cv2.morphologyEx(yellowStickerMask, cv2.MORPH_OPEN, kernel)
    maskY = cv2.morphologyEx(maskY, cv2.MORPH_CLOSE, kernel)
    
    maskBg = cv2.morphologyEx(yellowBGMask, cv2.MORPH_OPEN, kernel)
    maskBg = cv2.morphologyEx(maskBg, cv2.MORPH_CLOSE, kernel)
   
   
    maskW = cv2.morphologyEx(whiteStickerMask, cv2.MORPH_OPEN, kernel)
    maskW = cv2.morphologyEx(maskW, cv2.MORPH_CLOSE, kernel)
      
    # cv2.imwrite(os.path.join(debug_dir,name + "_maskY.jpg"), maskY)
    # cv2.imwrite(os.path.join(debug_dir,name + "_maskBg.jpg"), maskBg)
    # cv2.imwrite(os.path.join(debug_dir,name + "_maskW.jpg"), maskW)
    
    # logic to understand which type of sticker, and anount of bg
    sizeY = cv2.countNonZero(maskY)
    sizeW = cv2.countNonZero(maskW)
    
    # the decision on the color of the sticker:
    if sizeY < sizeW:
        foundColor = "White"
        _, maskW = cv2.threshold(maskW, 127, 255, cv2.THRESH_BINARY)
        
        fullMask = cv2.bitwise_or(maskW, maskBg)
    else: 
        _, maskY = cv2.threshold(maskY, 127, 255, cv2.THRESH_BINARY)
        foundColor = "Yellow"
        fullMask = cv2.bitwise_or(maskY, maskBg)


    _, maskBg = cv2.threshold(maskBg, 127, 255, cv2.THRESH_BINARY)


    # Findind the binding quadrangle
    contours, _ = cv2.findContours(fullMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if there are any contours
    if len(contours) == 0:
        #print("No blobs found!")
        exit()

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create an empty mask of the same size as the original image
    mask = np.zeros_like(fullMask)

    # Draw the largest blob onto the mask
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Extract only the largest blob from the original image
    largest_blob = cv2.bitwise_and(fullMask, mask)
    
    # finding the binding quadrangle
    hull = appx_best_fit_ngon(largest_blob)

    
    for idx in range(len(hull)):
        next_idx = (idx + 1) % len(hull)
        cv2.line(image, hull[idx], hull[next_idx], (0, 255, 0), 5)

    for pt in hull:
        cv2.circle(image, pt, 2, (255, 0, 0), 6)
    # cv2.imwrite('largest_blob.png', largest_blob)

    # cv2.imwrite(os.path.join(debug_dir, name + foundColor + "Contour.jpg"), image)
    # cv2.imwrite(os.path.join(debug_dir, name + "_fullMask.jpg"), fullMask)
    # cv2.imwrite(os.path.join(debug_dir, name + "_largest_blob.jpg"), largest_blob)
   
   
    return foundColor,fullMask ,maskBg, hull



def rectifyImage(image, fullMask, bgMask, hull,base_path=None, image_name=None):
    
    
    #print('-> Understarnd image type ')    
    # Read the image
    rows, cols = 2000,1500
    
    # Define the 3 pairs of corresponding points 
    # Convert hull points to sympy Points
    hull_points = [sympy.Point(pt[0], pt[1]) for pt in hull]

    center_x = sum(point.x for point in hull_points) / len(hull_points)
    center_y = sum(point.y for point in hull_points) / len(hull_points)
    
    # Define a function to compute the angle of a point with respect to the center
    def get_angle(point):
        dx = point.x - center_x
        dy = point.y - center_y
        return math.atan2(dy, dx)
    
    # Sort the points based on their angle with respect to the center 
    # by their coordinates so they fit the order(bottom right corner, top right corner, top left corner, bottom left corner)
    sorted_points = sorted(hull_points, key=get_angle, reverse=True)
    input_pts = np.float32([hull_points[0].coordinates, 
                                hull_points[1].coordinates, 
                                hull_points[2].coordinates,
                                hull_points[3].coordinates])
    #,) 
                                # hull_points[3].coordinates])   
    output_pts = np.float32([[1250,500], [1250,1500],[250,1500], [250,500],])
    
    # Calculate the transformation matrix using cv2.getAffineTransform()
    M= cv2.getPerspectiveTransform(input_pts , output_pts)
    
    # Apply the affine transformation using cv2.warpAffine()
    rectedImage = cv2.warpPerspective (image, M, (cols,rows))
    
    
    # retectedFullMask = cv2.warpPerspective (fullMask, M, (cols,rows))
    retectedBgMask = cv2.warpPerspective (bgMask, M, (cols,rows))

    debug_dir = create_debug_dir(base_path)
    name = os.path.splitext(image_name)[0]

    #crop the image and BGmask to have them netofor next stages
    retectedsticker = rectedImage[500:1500,250:1250]  
    retectedBgNeto = retectedBgMask[500:1500,250:1250] 
    # cv2.imwrite(os.path.join(debug_dir, name + "_rectedImage.jpg"), rectedImage)
    # cv2.imwrite(os.path.join(debug_dir, name + "_retectedsticker.jpg"), retectedsticker)
    # cv2.imwrite(os.path.join(debug_dir, name + "_retectedBgNeto.jpg"), retectedBgNeto)


       
    return rectedImage,retectedsticker, retectedBgNeto
        


def detect_barcodes_in_strip(image, strip_config, base_path=None, image_name=None):
    """
    Detect barcodes in a specific strip of the image, with adaptive detection.
    
    Args:
        image: The full image
        strip_config: Dictionary with strip configuration parameters
        display_results: Whether to display intermediate results
        
    Returns:
        list: List of detected barcode coordinates [(x, y, w, h), ...]
    """
    debug_dir = create_debug_dir(base_path)
    name = os.path.splitext(image_name)[0]
    
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
    # binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        #    cv2.THRESH_BINARY_INV, 11, 2)
    
    # More conservative thresholding to reduce false positives
    # threshold_value = 130  # Higher threshold to only catch true black pixels
    # _, binary_aggressive = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    # Combine the binary images
    # binary = cv2.bitwise_or(cv2.bitwise_or(binary_otsu, binary_adaptive), binary_aggressive)
    binary = binary_otsu
    # cv2.imwrite(os.path.join(debug_dir, name +"_binary_adaptive.jpg"), binary_adaptive)
    # cv2.imwrite(os.path.join(debug_dir, name +"_binary_otsu.jpg"), binary_otsu)
    # cv2.imwrite(os.path.join(debug_dir, name +"_binary_aggressive.jpg"), binary_aggressive)
    # cv2.imwrite(os.path.join(debug_dir, name +"_binary.jpg"), binary)

    # Additional validation for strip 3 - check if there's actually any content
    if strip_config['id'] == 3:
        # Calculate overall black pixel density in the strip
        total_black = np.sum(binary > 0)
        total_pixels = binary.size
        black_percentage = (total_black / total_pixels) * 100
        
        # If the entire strip has very few black pixels, it's likely empty
        if black_percentage < 5:  # Very low black pixel percentage indicates empty strip
            #print(f"Strip {strip_config['id']} appears empty (black pixel % = {black_percentage:.2f}%)")
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
            combined_score = 0.4 * black_pixel_percentage + 0.6 * edge_score
            
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
    barcode_regions = [(x-5, y-5, w, h) for x, y, w, h, _ in barcode_regions]
    
    # Display strip-specific results if requested
    if  barcode_regions:
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



def detect_barcodes_in_strip2(image, strip_config):
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
    
    if True:
        # barcode = decode(strip,symbols=[pyzbar.ZBarSymbol.CODE128])
        for d in decode(strip,symbols=[pyzbar.ZBarSymbol.CODE128]):
            strip = cv2.rectangle(strip, (d.rect.left, d.rect.top),
                                (d.rect.left + d.rect.width, d.rect.top + d.rect.height), (255, 0, 0), 2)
            strip = cv2.polylines(strip, [np.array(d.polygon)], True, (0, 255, 0), 2)
            strip = cv2.putText(strip, d.data.decode(), (d.rect.left, d.rect.top + d.rect.height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        # print('---------------------------')
        # print(d.rect)
        # print('---------------------------')
        if not d==None:
            print(d)
        return d  
    else:  #old method
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
        if  barcode_regions:
            output_strip = strip.copy()
            
            # Draw detected barcode regions on the strip
            for i, (x, y_rel, w, h) in enumerate(barcode_regions):
                # y_rel is relative to the full image, so adjust for strip
                y_in_strip = y_rel - y_start
                cv2.rectangle(output_strip, (x, y_in_strip), (x + w, y_in_strip + h), (0, 255, 0), 2)
                cv2.putText(output_strip, f"Strip {strip_config['id']} #{i+1}", (x, y_in_strip - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     # extract the rect
     
     # convert the rect to stick   
     
    return None



def detect_all_barcodes(image, retectedBgMask, foundColor,base_path=None, image_name=None):
        
    """
    Detect barcodes in all strips of the image with adaptive detection.
    
    Args:
        image: The image to process
        display_results: Whether to display and save visualization
        
    Returns:
        dict: Dictionary with strip IDs and their detected barcode coordinates
    """
    
    debug_dir = create_debug_dir(base_path)
    name = os.path.splitext(image_name)[0]

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
        barcode = detect_barcodes_in_strip(image, config, base_path, image_name)  # return a zbar barcode struct
        
        # print(('Barcode - rect' ,barcode.rect))
        all_barcodes[strip_id] = barcode
        #print(f"Strip {strip_id}: Found {len(barcode)} barcodes")
    
    if len(all_barcodes) == 0:
        return None
             
        
    # Display results on the full image if requested

    output = image.copy()
    
    # Color map for different strips
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red 
    x1 = 0
    y1 = 0
    x2 = 1
    y2 = 1
    # Draw detected barcode regions on the full image
    for strip_id, regions in all_barcodes.items():
        color = colors[(strip_id - 1) % len(colors)]
        for i, (x, y, w, h) in enumerate(regions):
            #print('rrrr  -- ',regions)
            #print('x  -- ',x,y,w,h)
            h_img, w_img = retectedBgMask.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

        # Check if rectangle is outside image bounds
        if x1 >= x2 or y1 >= y2:
            is_acceptable = True  # No intersection
        else:
            # Extract the region of interest
            roi = retectedBgMask[y1:y2, x1:x2]
            
            # Count positive pixels in the ROI
            positive_count = np.sum(roi > 0)
            
            # Check if the count is within the threshold (maximum 10)
            is_acceptable = positive_count <= 10
            
            # if is_acceptable:
            #     cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(output, f"Strip {strip_id} #{i+1}", (x, y - 5), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # else remove this barcode from the list of detected
               
   
    # cv2.imwrite('all_detected_barcodes.jpg', output)
    # cv2.imwrite(os.path.join(debug_dir, name + "_barcodes.jpg"), output)

        
    return all_barcodes


# # For debugging: draw bounding boxes on the image
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
    image_height, image_width = image.shape[:2]  # Get image dimensions
    # check if roi is inside the image:
    is_inside = (x >= 0 and y >= 0 and 
            x + w <= image_width and 
            y + h <= image_height)
    if not is_inside:
        x, y, w, h = (30, 245, 220, 45)
        #print("Bad bad roi.....breaking here")
        
    # Extract the ROI from the original image
    #print((x,y,w,h))
    
    roi = image[y:y + h, x:x + w] 
    

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Binarize (adaptive threshold)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,31 , 2
    )
    # Convert array to PIL image
    pil_img = Image.fromarray(bin_img)

    # OCR with Tesseract
    text_raw = pytesseract.image_to_string(pil_img, config="--psm 7", lang="heb+eng")

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



def extract_fields_from_barcode(image, barcode_location, strip_id, foundColor, folder_path, fileName):
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
        name_shift = (25, -220, 220, 45)
        id_shift = (-37, -140, 140, 40)
        date_shift = (-37, -101, 130, 36)
        case_id_shift = (-25, -42, 160, 41)
    
    elif strip_id == 3:
        # Third strip barcode shifts
        case_id_shift = (-125, -264, 165, 48)
        name_shift = (165, -215, 245, 48)
        date_shift = (550, 50, 170, 55)
        id_shift = (440, -210, 280, 45)
        
    fields = {}

    # different postion in yellow and white
    if foundColor == "White" and strip_id == 3:
        case_id_shift = (-34, -265, 164, 49)
        name_shift = (220, -210, 245, 45)
        date_shift = (550, 50, 190, 55)         
        id_shift = (540, -210, 240, 48)
        
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
        
        
        # Save the ROI for debugging
        roi_debug = image[by + case_id_shift[1]:by + case_id_shift[1] + case_id_shift[3], 
                        bx + case_id_shift[0]:bx + case_id_shift[0] + case_id_shift[2]]
        cv2.imwrite(f'case_id_roi_strip_{strip_id}.jpg', roi_debug)
        
        # Try different PSM modes
        psm_modes = [6, 7, 8, 9, 10, 11, 13]
        best_result = ""
        
        #print(f"Trying different PSM modes for Case ID in strip {strip_id}:")
        for psm in psm_modes:
            config_str = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
            lang = "eng"
            extracted_case_id = extract_text_field(image, case_id_roi, config_str, lang)
            cleaned_case_id = ''.join(ch for ch in extracted_case_id if ch.isdigit())
            
            #print(f"  PSM {psm}: Raw: '{extracted_case_id}', Cleaned: '{cleaned_case_id}'")
            
            # Keep the longest result (usually the most complete)
            if len(cleaned_case_id) > len(best_result):
                best_result = cleaned_case_id
        
        # Use the best result
        fields["case_id"] = best_result
        #print(f"Selected case_id: '{best_result}'")
        
    
    
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
    

    if True:
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
        debug_dir = create_debug_dir(folder_path)
        name = os.path.splitext(fileName)[0]
    
        cv2.imwrite(os.path.join(debug_dir, name +"_visualization.jpg"), visualization_image)
        #print(os.path.join(debug_dir, name +"_visualization.jpg"))



    return fields


 
def process_barcodes_and_extract_fields(image, barcodes_dict, color, folder_path, fileName):
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
    debug_dir = create_debug_dir(folder_path)
    name = os.path.splitext(fileName)[0]
    
    priority_order = [3, 2, 1]
    #priority_order = [1, 3, 2]

    # Find the first available barcode according to priority
    selected_barcode = None
    selected_strip_id = None
    
    for strip_id in priority_order:
        if strip_id in barcodes_dict and barcodes_dict[strip_id]:
            # Take the first barcode from this strip
            selected_barcode = barcodes_dict[strip_id][0]
            #print('selected_barcode ==',selected_barcode)
            selected_strip_id = strip_id
            #print(f"Selected barcode from strip {selected_strip_id}: {selected_barcode}")
            break
    
    # If no barcode found, return empty result
    if selected_barcode is None:
        return {"error": "No barcodes detected in the image"}
    
    # Extract fields from the selected barcode
    fields = extract_fields_from_barcode(image, selected_barcode, selected_strip_id, color, folder_path, fileName)
    
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
    #print(f"Fields saved to {output_path}")



def extract_text_as_json(image_path):
    """
    Extracts text data from a given image path and returns it as JSON.
    
    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: JSON-encoded extracted text data.
    """
    folder_path, fileName = os.path.split(image_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return json.dumps({"error": f"Could not load image: {fileName}"})

    # Resize image while maintaining aspect ratio
    target_width = 900
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_height = int(target_width * aspect_ratio)
    image = cv2.resize(image, (target_width, new_height))

    # Step 1: Detect and segment the sticker
    foundColor, fullMask, bgMask, hull = understandSticker(image, folder_path, fileName)

    # Step 2: Rectify the sticker's orientation
    rectedImage, rectedSticker, retectedBgMask = rectifyImage(image, fullMask, bgMask, hull, folder_path, fileName)

    # Step 3: Detect barcodes
    detected_barcodes = detect_all_barcodes(rectedSticker, retectedBgMask, foundColor, folder_path, fileName)

    # Step 4: Extract fields from barcodes
    extracted_fields = process_barcodes_and_extract_fields(rectedSticker, detected_barcodes, foundColor, folder_path, fileName)

    # Step 5: Fix Hebrew text encoding issues
    fixed_output = extracted_fields


    # Convert the extracted data to JSON format
    return json.dumps(fixed_output, ensure_ascii=False, indent=2)

