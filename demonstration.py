import json
import sys
sys.path.append('.')

from algorithms import *
from PIL import Image, ImageDraw, ImageFont
import os
from pyzbar import pyzbar
from pyzbar.pyzbar import decode
import cv2
import arabic_reshaper
from bidi.algorithm import get_display



# Convert extracted fields dictionary values to properly displayed Hebrew
def fix_hebrew_text(data):
    if isinstance(data, dict):
        return {k: fix_hebrew_text(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [fix_hebrew_text(v) for v in data]
    elif isinstance(data, str):
        return get_display(arabic_reshaper.reshape(data))
    return data



def processImage(folder_path, fileName):
           
    image_path = os.path.join(folder_path, fileName)
    
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image '{fileName}'.")
        return None
            
    # Calculate new height maintaining aspect ratio
    target_width = 900
    h, w = image.shape[:2]
    aspect_ratio = h / w
    new_height = int(target_width * aspect_ratio)
    
    # Resize the cropped image
    image = cv2.resize(image, (target_width, new_height))
    
    # 1. Find and Understand  - yellow/white. extract fulls mask and a quadrangle to be rectified 
    foundColor, fullMask, bgMask, hull =  understandSticker(image,folder_path, fileName) 
     
    # cv2.imshow(foundColor,image)
    print('color - ',foundColor )
    
    # 2. rectify the sticker
    rectedImage,rectedSticker, retectedBgMask  = rectifyImage(image, fullMask, bgMask, hull,folder_path, fileName)
      
    # Just to make sure that all the stickers are oriented correctly 
    decoded_list = decode(rectedSticker,symbols=[pyzbar.ZBarSymbol.CODE128])
    if len(decoded_list) != 0:
        if decoded_list[0].type == 'CODE128' and not decoded_list[0].orientation == 'UP':
            if decoded_list[0].orientation == 'LEFT':
                # Rotate by 90 degrees clockwise
                rectedSticker = cv2.rotate(rectedSticker, cv2.ROTATE_90_CLOCKWISE)
                retectedBgMask = cv2.rotate(retectedBgMask, cv2.ROTATE_90_CLOCKWISE)
            elif decoded_list[0].orientation == 'RIGHT':
                # Rotate by 270 degrees clockwise
                rectedSticker = cv2.rotate(rectedSticker, cv2.ROTATE_90_COUNTERCLOCKWISE)
                retectedBgMask = cv2.rotate(retectedBgMask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif decoded_list[0].orientation == 'DOWN':
                # Rotate by 180 degrees clockwise
                rectedSticker = cv2.rotate(rectedSticker, cv2.ROTATE_180)
                retectedBgMask = cv2.rotate(retectedBgMask, cv2.ROTATE_180)
            # cv2.imwrite(os.path.join(debug_dir, name + "_Rotated_rectedImage.jpg"), rectedImage)

                
    # print(decoded_list[0].orientation)
     # 3. Find all barcodes in the resized image. locations are slighltly different yellow/whit
    detected_barcodes = detect_all_barcodes(rectedSticker, retectedBgMask, foundColor, folder_path, fileName)
    
    
    # Extract fields's data
    extracted_fields = process_barcodes_and_extract_fields(rectedSticker, detected_barcodes, foundColor, folder_path, fileName)
    
    # allExtractedData = extractAllData (rectedImage, foundColor)
    # finalData = refineAllData(allExtractedData)            
        
    # Print the extracted fields as JSON
    #print(json.dumps(extracted_fields, ensure_ascii=False, indent=2))
    fixed_output = fix_hebrew_text(extracted_fields)
    print(json.dumps(fixed_output, ensure_ascii=False, indent=2))
    return None,None
    # json = data2json(finalData)

    # return None,None
     
    
    # Step 3: Process barcodes and extract fields


def process_images_in_folder(folder_path):

# Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Get list of JPG files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No JPG images found in '{folder_path}'.")
        return
    
    print(f"Found {len(image_files)} JPG images. Starting processing...")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(folder_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    
    for idx, image_file in enumerate(image_files):
        
        print(f"Processing image {idx+1}/{len(image_files)}: {image_file}") 
        
        data, finalimages = processImage(folder_path, image_file)
        
        
        # pack to json = 


if __name__ == "__main__":
    
    # Path to the input image file
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        #dbPath =  "C:/Users/zvika/Documents/Computer Vision/Tomer/data base Wood/data base"
        # dbPath =  "C:/Users/zvika/Documents/Computer Vision/Tomer/data base/data base/pictures"
        dbPath =  "C:/Users/A3DC~1/Desktop/data base/pictures"
        # C:\Users\zvika\Documents\Computer Vision\Tomer\Data Base2\Data Base
        # folder_path = input("Enter the path to the folder containing JPG images: ")
    
    process_images_in_folder(dbPath)
      
    '''  
    dbPath =  "C:/Users/zvika/Documents/Computer Vision/Tomer/data base/data base/pictures"
      
    image_path = "C:/Users/zvika/Documents/Computer Vision/Tomer/data base/data base/pictures/3.jpg" 
    # image_path = "C:/Users/zvika/Documents/Computer Vision/Tomer/data base/data base/   2/7.jpg"
    img = Image.open(image_path)
    
    
    
        print('----------------------------------------------')
    '''
    
    
    
    # # Step 1: Detect and resize the yellow sticker
    # resized = algorithms.process_image(image_path)
    
    # # Step 2: Find all barcodes in the resized image
    # detected_barcodes = algorithms.detect_all_barcodes(resized, display_results=True)
    
    # # Step 3: Process barcodes and extract fields
    # print('Finished Detecting..... extracting next:')
    # extracted_fields = algorithms.process_barcodes_and_extract_fields(resized, detected_barcodes, display=True)
    
    # # Print the extracted fields as JSON
    # print(json.dumps(extracted_fields, ensure_ascii=False, indent=2))
    
    # Optionally save the results to a file
    # save_fields_to_json(extracted_fields, "extracted_fields.json")
    
