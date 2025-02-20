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

from ocr import OCRProcessor, WordInfo
ocr_processor = OCRProcessor()
image = cv2.imread("pictures/easy_example_medium.jpg", 1)
words_info = ocr_processor.process(image)
res = pytesseract.image_to_data(image, config=ocr_processor.custom_config, output_type=pytesseract.Output.DICT)

for word in words_info:
    print(word)

full_name = ocr_processor.detect_name(words_info)
# print(f'*** name is: {full_name}')
id = ocr_processor.detect_person_id(words_info, image)
# print(f'*** id is: {id}')
case_number = ocr_processor.detect_case_number(words_info, image)
# print(f'*** case_number is: {case_number}')
date = ocr_processor.detect_date(words_info, image)

# Store the values in a dictionary
data = {
    "name": full_name,
    "id": id,
    "case_number": case_number,
    "date": date
}

print(data)