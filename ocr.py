import cv2 # pip3 install opencv-python

#sudo apt  install tesseract-ocr
#pip3 install pytesseract
#sudo apt-get install tesseract-ocr-heb

import pytesseract
from collections import defaultdict
import random
import os
import numpy as np
from dataclasses import dataclass
import math
import json
from matplotlib import pyplot as plt

# הפרטים שצריך מתמונה: 
# - שם מלא
# - תעודת זהות
# - מספר מקרה
# - תאריך (התאריך בו צולמה המדבקה, היום בו רץ האלגוריתם על התמונה)
# 
@dataclass
class WordInfo:
    text: str
    line: int
    rect: tuple  # (x, y, w, h)
    pixel_center: tuple  # (center_x, center_y)
    conf: int  # Confidence level
    def __str__(self):
        return f"Text: '{self.text}', Rect: {self.rect}, Pixel Center: {self.pixel_center}, Confidence: {self.conf}"
    
class OCRProcessor:
    def __init__(self ):
        
        self.custom_config = f'--oem 3 --psm 6 -l heb'
        #pytesseract.pytesseract.tesseract_cmd = r"
        
        
    def rotate_image(self, bgr_image, bgr_color, threshold=40):
        if bgr_image is None:
            raise ValueError("Input image is None. Ensure the image path is correct and accessible.")
        
        if not isinstance(bgr_image, np.ndarray):
            raise TypeError("Input is not a valid NumPy array.")     

        # Calculate Euclidean distance in BGR color space
        bgr_distance = np.sqrt(np.sum((bgr_image - bgr_color) ** 2, axis=2))

        # Create a binary mask where distance is less than the threshold
        mask = (bgr_distance < threshold).astype(np.uint8) * 255

        # Find contours to identify the largest connected component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No regions found with the specified color.")

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Find the rotated rectangle (minAreaRect)
        rotated_rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rotated_rect)  # Get the 4 points of the rotated rectangle
        box_points = np.int0(box_points)  # Convert the points to integers

        # Draw the rotated rectangle on the original image
        cv2.drawContours(bgr_image, [box_points], 0, (0, 255, 0), 2)  # Draw green rotated rectangle

        # Get the top-left and top-right points of the rotated rectangle
        top_left = box_points[0]
        top_right = box_points[1]

        # Compute the angle of the top side with respect to the horizontal axis
        delta_y = top_right[1] - top_left[1]
        delta_x = top_right[0] - top_left[0]
        angle = np.degrees(np.arctan2(delta_y, delta_x))  # Angle in degrees

        # Ensure the angle is positive (0 to 180 degrees)
        # if angle < 0:
        #     angle += 180 + 90

        print(f" angle {angle}")

        if math.fabs(angle) > 45:
            angle+= 90
        # Get the center of the rotated rectangle for rotation
        center = rotated_rect[0]

        # Compute the rotation matrix to align the top horizontal line
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate the new bounding box size after rotation
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust the rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Rotate the image without cropping
        rotated_image = cv2.warpAffine(bgr_image, rotation_matrix, (new_w, new_h))

        # Display the rotated image and the original image with the rotated rectangle
        # cv2.imshow('Aligned Image', rotated_image)
        # cv2.imshow('Original Image with Rotated Rect', bgr_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return rotated_image




    def crop_by_color(self, bgr_image, bgr_color, threshold=40):
        if bgr_image is None:
            raise ValueError("Input image is None. Ensure the image path is correct and accessible.")
        
        if not isinstance(bgr_image, np.ndarray):
            raise TypeError("Input is not a valid NumPy array.")     

        # Calculate Euclidean distance in BGR color space
        bgr_distance = np.sqrt(np.sum((bgr_image - bgr_color) ** 2, axis=2))

        # Create a binary mask where distance is less than the threshold
        mask = (bgr_distance < threshold).astype(np.uint8) * 255

        # Find contours to identify the largest connected component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No regions found with the specified color.")

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the original image
        cropped_image = bgr_image[y:y + h, x:x + w]        

        return cropped_image
 
    
    
    def process(self, image, delta=25):
        # Create a copy of the image for debugging
        debug_image = image.copy()
       
        # Get OCR data
        data = pytesseract.image_to_data(image, config=self.custom_config, output_type=pytesseract.Output.DICT)

        # List to store WordInfo objects
        word_info_list = []

        # Iterate through each detected word
        for i in range(len(data['text'])):
            if int(data['conf'][i]) >= 0:  
                word = data['text'][i]
                if word == '|':
                    continue
                
                conf = int(data['conf'][i])

                # Get bounding box coordinates
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red box for original rectangle

                # Calculate the center of the bounding box
                center_x = x + w // 2
                center_y = y + h // 2

                # Expand the width and height of the bounding box by delta
                new_w = w + delta
                new_h = h + delta

                # Adjust x and y to shift the bounding box and keep the center
                new_x = center_x - new_w // 2
                new_y = center_y - new_h // 2

                # Store the word details in WordInfo with expanded bounding box
                word_info = WordInfo(
                    text=word,
                    line=data["line_num"][i],
                    rect=(new_x, new_y, new_w, new_h),
                    pixel_center=(center_x, center_y),
                    conf=conf
                )
                word_info_list.append(word_info)

                # Draw the expanded rectangle on the image

                # Write the word above the rectangle
                text_x = new_x
                text_y = new_y - 10  # Adjust to place text above the box

        # Save and display the image (optional)
        #cv2.imshow("Detected Words", debug_image)
        #cv2.waitKey(0)

        # Return the list of WordInfo objects
        return word_info_list


    def detect_name(self, words_info):

        name_from_large_sticker = self.detect_name_on_large_sticker(words_info)
        if name_from_large_sticker != '':
            return name_from_large_sticker
        
        name_from_small_sticker  = self.detect_name_on_small_sticker(words_info)
        print(f' name_from_small_sticker {name_from_small_sticker}')
        return name_from_small_sticker

    
    def detect_name_on_large_sticker(self, words_info, y_threshold=10):
        full_name = []  # List to store the detected full name
        
        for index, word in enumerate(words_info):
            if 'שם' in word.text:
                line_of_name = word.line
                name = ""
                if ":" in word.text:
                    name += word.text[(word.text.find(":") + 1):]
                    index = index + 1
                    word = words_info[index]
                    while word.line == line_of_name:
                        name += " " + word.text
                        index += 1
                        word = words_info[index]
                else:
                    raise(NotImplementedError)
                return name


                # Check subsequent words for the same y-range
                for i in range(index + 1, len(words_info)):
                    next_word = words_info[i]
                    
                    # If the y-coordinate is close enough, consider it part of the same name
                    if abs(next_word.rect[1] - word.rect[1]) <= y_threshold:
                        full_name.append(next_word.text)
                        # print(f"Added '{next_word.text}' to full_name.")
                    else:
                        # If the y-coordinate is too far apart, stop checking further
                        break
                
                # Print the full name once detected
                return ' '.join(full_name)  # Return the full name once it's detected

        return ''

    def detect_name_on_small_sticker(self, words_info):
        full_name = []  # List to store the detected full name
        
        for index, word in enumerate(words_info):
            # the word is the the age of the person, break
            if word.text.isdigit() :           
                break
            
            full_name.append(word.text)
                

        if len(full_name) > 0:
            full_name_string = " ".join(full_name)
            return full_name_string

        return ''

    def detect_person_id(self, words_info, image):
        
        max_length = 0  # To track the maximum length of number sequence
        detected_id = ''  # To store the ID with the longest sequence of numbers
        image_height, image_width = image.shape[:2]  # Get image dimensions
        
        detected_ids = []
        for index, word in enumerate(words_info):
            # Check if the word contains 'מ.ז' or 'מז'
            if 'מ.ז' in word.text or 'מז' in word.text:
                # Check only the next word, if exists
                if ":" in word.text:
                    next_word = words_info[index + 1]
                else:
                    next_word = words_info[index + 2]


                # Extract the bounding box coordinates of the next word
                x, y, w, h = next_word.rect

                # Ensure the bounding box stays within the image bounds
                x = max(0, x)  # Clamp x to be at least 0
                y = max(0, y)  # Clamp y to be at least 0
                w = min(w, image_width - x)  # Ensure the width does not go beyond the image
                h = min(h, image_height - y)  # Ensure the height does not go beyond the image


                # Crop the image using the bounding box of the next word
                cropped_image = image[y:y + h, x:x + w]
                # Convert to grayscale
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
                
                # Use pytesseract to extract text from the cropped image
                cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')
                

                detected_ids.append(cropped_text)                   
                   
        
        
        if len(detected_ids) > 0:
            print('multiple ids !')
            # Find the ID with the maximum number of numeric characters
            largest_id = max(detected_ids, key=lambda x: sum(c.isdigit() for c in x))
            
            # Create a string containing only the numeric characters from the largest ID
            numeric_id = ''.join(c for c in largest_id if c.isdigit())
            
            return numeric_id
            
        # Return the detected ID with the longest number sequence
        return detected_id

    def detect_case_number_on_large_sticker(self, words_info, image):

        detected_case_number = ''  # To store the case  with the longest sequence of numbers
        image_height, image_width = image.shape[:2]  # Get image dimensions

        for index, word in enumerate(words_info):
            if 'מקר' in word.text:  # and any(c.isdigit() for c in word.text):

                word = words_info[index + 1]
                return word.text

                # Extract the bounding box coordinates of the next word
                x, y, w, h = word.rect

                # Ensure the bounding box stays within the image bounds
                x = max(0, x)  # Clamp x to be at least 0
                y = max(0, y)  # Clamp y to be at least 0
                w = min(w, image_width - x)  # Ensure the width does not go beyond the image
                h = min(h, image_height - y)  # Ensure the height does not go beyond the image

                # Crop the image using the bounding box of the next word
                cropped_image = image[y:y + h, x:x + w]
                # Convert to grayscale
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

                # Use pytesseract to extract text from the cropped image
                cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')

                detected_case_number = ''.join(c for c in cropped_text if c.isdigit())

                # Display the cropped image (optional for debugging)
                # cv2.imshow(f'Index {index}', cropped_image)
                # cv2.imshow(f'binary_image', binary_image)

                # cv2.waitKey(0)

                return detected_case_number

        return detected_case_number
    
    def detect_case_number(self, words_info, image):
        
        res = self.detect_case_number_on_large_sticker(words_info, image)
        
        if res != '':
            return res

        max_length = 0  # To track the maximum length of number sequence
        detected_id = ''  # To store the ID with the longest sequence of numbers
        image_height, image_width = image.shape[:2]  # Get image dimensions
        
        detected_ids = []
        for index, word in enumerate(words_info):
            if 'מקרה' in word.text  and not any(c.isdigit() for c in word.text):
                # Check only the next word, if exists
                next_word = words_info[index+1]   
                            
                # Extract the bounding box coordinates of the next word
                x, y, w, h = next_word.rect

                # Ensure the bounding box stays within the image bounds
                x = max(0, x)  # Clamp x to be at least 0
                y = max(0, y)  # Clamp y to be at least 0
                w = min(w, image_width - x)  # Ensure the width does not go beyond the image
                h = min(h, image_height - y)  # Ensure the height does not go beyond the image


                # Crop the image using the bounding box of the next word
                cropped_image = image[y:y + h, x:x + w]
                # Convert to grayscale
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
                
                # Use pytesseract to extract text from the cropped image
                cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')
                # Skipping: Contains letters
                if any(c.isalpha() for c in cropped_text):
                    continue
                
                # Display the cropped image (optional for debugging)
                #cv2.imshow(f'Index {index}', cropped_image)
                #cv2.imshow(f'binary_image', binary_image)

                #cv2.waitKey(0)

                detected_ids.append(cropped_text)                   
                   
        
        
        if len(detected_ids) > 0:
            # Find the ID with the maximum number of numeric characters
            largest_id = max(detected_ids, key=lambda x: sum(c.isdigit() for c in x))
            
            # Create a string containing only the numeric characters from the largest ID
            numeric_id = ''.join(c for c in largest_id if c.isdigit())
            
            return numeric_id
            
        # Return the detected ID with the longest number sequence
        return detected_id
    
    def detect_date(self, words_info, image):        
     
        image_height, image_width = image.shape[:2]  # Get image dimensions
        
        for index, word in enumerate(words_info):
            if 'קבלה' in word.text :
                # Check only the next word, if exists
                next_word = words_info[index+1]
                return next_word.text # don't try to improve contrast, it may get worse
                            
                # Extract the bounding box coordinates of the next word
                x, y, w, h = next_word.rect

                # Ensure the bounding box stays within the image bounds
                x = max(0, x)  # Clamp x to be at least 0
                y = max(0, y)  # Clamp y to be at least 0
                w = min(w, image_width - x)  # Ensure the width does not go beyond the image
                h = min(h, image_height - y)  # Ensure the height does not go beyond the image


                # Crop the image using the bounding box of the next word
                cropped_image = image[y:y + h, x:x + w]
                # Convert to grayscale
                gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                # Apply binary thresholding
                _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
                
                # Use pytesseract to extract text from the cropped image
                cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')
                # Skipping: Contains letters
                if any(c.isalpha() for c in cropped_text):
                    continue
                print(f'index: {index}, Cropped text: {cropped_text}')

                # Display the cropped image (optional for debugging)
                #cv2.imshow(f'Index {index}', cropped_image)
                #cv2.imshow(f'binary_image', binary_image)

                #cv2.waitKey(0)
       
            
                return cropped_text

        return ''




if __name__ == "__main__":   
   
    # test =  cv2.imread("/home/yakir/sticker_ws/new/test.jpeg",1)   
    # gray_image = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    # _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)    
    # cropped_text = pytesseract.image_to_string(binary_image, config='--psm 6')
    # print(f' cropped_text {cropped_text}')            
    # exit(1)            
    
    ocr_processor = OCRProcessor()   

    #image = cv2.imread("/home/cogniteam-user/sticker_ws/sticker_recognition/new/3.jpeg",1)
    image = cv2.imread("pictures/IMG-20250219-WA0029.jpg", 1)
   
    
    yellow_bgr_color =   (0, 187, 200)  

    #rotate_image = ocr_processor.rotate_image(image, yellow_bgr_color) #todo debug
    rotate_image = image

    plt.figure()
    plt.imshow(image)
    plt.show()
    #cv2.imshow(f'rotate_image', rotate_image) #debug
    #cv2.waitKey(0)

    cropped_image = ocr_processor.crop_by_color(rotate_image, yellow_bgr_color)
    
    words_info = ocr_processor.process(cropped_image)
    for word in words_info:
        print(word)
    
    full_name = ocr_processor.detect_name(words_info)
    #print(f'*** name is: {full_name}')
    id = ocr_processor.detect_person_id(words_info, cropped_image)
    #print(f'*** id is: {id}')   
    case_number = ocr_processor.detect_case_number(words_info, cropped_image)
    #print(f'*** case_number is: {case_number}')
    date =  ocr_processor.detect_date(words_info, cropped_image)
    #print(f'*** date is: {date}')

    # Store the values in a dictionary
    data = {
        "name": full_name,
        "id": id,
        "case_number": case_number,
        "date": date
    }

    # Convert the dictionary to a JSON string and print it
    json_data = json.dumps(data, ensure_ascii=False, indent=4)
    print(f"*** JSON data: {json_data}")