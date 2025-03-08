#!/usr/bin/env python3
"""
Yellow Sticker Barcode Detection and Field Extraction Script
This script processes an image containing a yellow sticker with barcodes,
detects the barcodes, and extracts relevant patient fields.
"""

import json
import argparse
from helper_resize import process_image
from helper_find_barcodes import detect_all_barcodes
from helper_extract import process_barcodes_and_extract_fields, save_fields_to_json

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
    
    # Optionally save the results to a file
    if args.save:
        save_fields_to_json(extracted_fields, args.save)
        print(f"Results saved to {args.save}")

if __name__ == "__main__":
    main()