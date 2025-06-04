import xml.etree.ElementTree as ET
import os
import cv2 #OpenCV
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import openslide
import pandas as pd

# ### Check whether the image has ImageDescription metadata

# try:
#     # if directory contains Scan1 folder, use Scan1 folde   
#     for i in range(149,400):
#         base_dir = r'C:\Users\yoon\Desktop\multiplexIHC\data'
#         file_path = os.path.join(base_dir, str(i), 'Scan1', f'{i}_Scan1.qptiff')
#         with tifffile.TiffFile(file_path) as tif:
#             # ImageDescription
#             if 270 in tif.pages[0].tags:
#                 image_description_tag = tif.pages[0].tags[270]
#                 metadata_str = image_description_tag.value
#                 print("ImageDescription 메타데이터:")
#                 print(metadata_str)
                
# except Exception as e:
#     print(f"오류 발생: {e}")

# -- Configuration --

# -- Load data --
QPTIFF_DIR = r'C:\Users\yoon\Desktop\multiplexIHC\data'
OUTPUT_DIR = r'C:\Users\yoon\Desktop\multiplexIHC\data_load'
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_metadata = os.path.join(OUTPUT_DIR, 'metadata.csv')

patch_size = 1024
level = 0 # WSI level (0 for highest resolution)
overlap = 0
threshold_for_tissue = 220 # Adjust based on image

# --- Helper functions ---
def parse_metadata(metadata_str):
    """
    Parse the metadata string from the ImageDescription tag
    """
    metadata = {}
    try:
        root = ET.fromstring(metadata_str)
        
        # Define a helper to safely get text from an element
        def get_text(element_name):
            element = root.find(element_name)
            return element.text if element is not None else None
        
        metadata['DescriptionVersion'] = get_text('DescriptionVersion')
        metadata['AcquisitionSoftware'] = get_text('AcquisitionSoftware')
        metadata['ImageType'] = get_text('ImageType')
        metadata['Identifier'] = get_text('Identifier')
        metadata['SlideID'] = get_text('SlideID')  # Important for linking!
        metadata['ComputerName'] = get_text('ComputerName')
        metadata['ExposureTime'] = get_text('ExposureTime')
        metadata['SignalUnits'] = get_text('SignalUnits')
        
        # Channel information - look for both 'Name' and 'ChannelName'
        metadata['ChannelName'] = get_text('Name') or get_text('ChannelName')  # e.g., DAPI
        metadata['ChannelColor'] = get_text('Color')
        metadata['Objective'] = get_text('Objective')
        metadata['CameraName'] = get_text('CameraName')

        # Handle nested ScanProfile
        scan_profile = root.find('ScanProfile/root')  # Accessing nested root
        if scan_profile is not None:
            metadata['OpalKitType'] = scan_profile.findtext('OpalKitType')
            metadata['SampleIsTMA'] = scan_profile.findtext('SampleIsTMA')
            
            camera_settings = scan_profile.find('CameraSettings')
            if camera_settings is not None:
                metadata['CameraGain'] = camera_settings.findtext('Gain')
                metadata['CameraBits'] = camera_settings.findtext('Bits')
                metadata['CameraBinning'] = camera_settings.findtext('Binning')
        
        # Alternative scan profile structure (sometimes it's directly under root)
        else:
            scan_profile_alt = root.find('ScanProfile')
            if scan_profile_alt is not None:
                metadata['OpalKitType'] = scan_profile_alt.findtext('OpalKitType')
                metadata['SampleIsTMA'] = scan_profile_alt.findtext('SampleIsTMA')

    except ET.ParseError as e:
        print(f"XML ParseError: {e}. Storing raw metadata.")
        metadata['raw_description'] = metadata_str  # Fallback
        metadata['parser_error'] = f"XML Parse Error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error parsing metadata: {e}")
        metadata['raw_description'] = metadata_str  # Fallback
        metadata['parser_error'] = f"Unexpected error: {str(e)}"
        
    return metadata

def get_qptiff_path(base_dir, i):
    """Helper function to construct QPTIFF file paths"""
    return os.path.join(base_dir, str(i), 'Scan1', f'{i}_Scan1.qptiff')

def is_tissue_patch(patch_array, threshold=230):
    """
    Determine if a patch contains tissue based on intensity threshold.
    Returns True if patch contains tissue (should be kept), False if background.
    """
    # Convert to grayscale for analysis
    if len(patch_array.shape) == 3:
        grayscale = np.mean(patch_array, axis=2)
    else:
        grayscale = patch_array
    
    # Calculate percentage of pixels below threshold (tissue is darker)
    tissue_pixels = np.sum(grayscale < threshold)
    total_pixels = grayscale.size
    tissue_percentage = tissue_pixels / total_pixels
    
    # Keep patches with at least 10% tissue
    return tissue_percentage > 0.1

def extract_patches(slide_path, output_dir, base_filename, patch_size, level, overlap, slide_metadata):
    """
    Extracts patches from a WSI and associates them with slide-level metadata.
    """
    patches_info = []
    try:
        slide = openslide.OpenSlide(slide_path)
        
        # Check if the requested level exists
        if level >= slide.level_count:
            print(f"Warning: Level {level} not available. Using level 0 instead.")
            level = 0
            
        width, height = slide.level_dimensions[level]
        downsample_factor = slide.level_downsamples[level]

        print(f"Processing {base_filename}: Dimensions (Level {level}) {width}x{height}, Downsample: {downsample_factor:.2f}x")

        patch_count = 0
        total_patches = 0
        
        for y_coord in range(0, height - patch_size + 1, patch_size - overlap):
            for x_coord in range(0, width - patch_size + 1, patch_size - overlap):
                total_patches += 1
                
                # Calculate level 0 coordinates
                patch_l0_x = int(x_coord * downsample_factor)
                patch_l0_y = int(y_coord * downsample_factor)

                # Read patch
                patch_img = slide.read_region((patch_l0_x, patch_l0_y), level, (patch_size, patch_size))
                patch_img_rgb = patch_img.convert('RGB')
                patch_array = np.array(patch_img_rgb)

                # Check if patch contains tissue
                if not is_tissue_patch(patch_array, threshold_for_tissue):
                    continue  # Skip background patch

                patch_count += 1
                patch_filename = f"{base_filename}_patch_L{level}_x{x_coord:06d}_y{y_coord:06d}.png"
                patch_filepath = os.path.join(output_dir, patch_filename)
                patch_img_rgb.save(patch_filepath)

                # --- USER CUSTOMIZATION: Label Assignment ---
                patch_label = slide_metadata.get('SlideID', 'unknown_slide')
                if slide_metadata.get('SampleIsTMA') == 'true':
                    patch_label = f"TMA_{slide_metadata.get('SlideID', 'unknown')}"
                
                # Add channel information to label if available
                channel_name = slide_metadata.get('ChannelName')
                if channel_name:
                    patch_label = f"{patch_label}_{channel_name}"
                # --- END USER CUSTOMIZATION ---

                patch_info_entry = {
                    'slide_basename': base_filename,  # Original qptiff filename without extension
                    'patch_filename': patch_filename,
                    'level': level,
                    'patch_x_coord_lvl': x_coord,  # Patch coordinate at current level
                    'patch_y_coord_lvl': y_coord,  # Patch coordinate at current level
                    'patch_l0_x': patch_l0_x,  # Level 0 coordinates
                    'patch_l0_y': patch_l0_y,  # Level 0 coordinates
                    'patch_size': patch_size,
                    'label': patch_label,
                    'tissue_percentage': np.sum(np.mean(patch_array, axis=2) < threshold_for_tissue) / (patch_size * patch_size)
                }
                
                # Add all extracted slide-level metadata to each patch entry
                patch_info_entry.update(slide_metadata) 
                patches_info.append(patch_info_entry)
        
        print(f"  -> Kept {patch_count}/{total_patches} patches with tissue content")
        slide.close()
        
    except openslide.OpenSlideError as e:
        print(f"OpenSlideError for {slide_path}: {e}")
    except Exception as e:
        print(f"Failed to process patches for {slide_path}: {e}")
        
    return patches_info

# --- Main Batch Processing ---
def main():
    """Main function to process all QPTIFF files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all image files
    all_qptiff_files = []
    for ext in ['*.qptiff', '*.tif', '*.tiff', '*.svs']:  # Added more common formats
        all_qptiff_files.extend(glob.glob(os.path.join(QPTIFF_DIR, ext)))
        all_qptiff_files.extend(glob.glob(os.path.join(QPTIFF_DIR, '**', ext), recursive=True))  # Recursive search
    
    all_qptiff_files = list(set(all_qptiff_files))  # Remove duplicates

    if not all_qptiff_files:
        print(f"No image files found in {QPTIFF_DIR}")
        print("Supported formats: .qptiff, .tif, .tiff, .svs")
        return

    print(f"Found {len(all_qptiff_files)} image files:")
    for file in all_qptiff_files:
        print(f"  - {file}")

    master_patch_list = []

    for i, qptiff_file_path in enumerate(all_qptiff_files, 1):
        base_filename = os.path.splitext(os.path.basename(qptiff_file_path))[0]
        print(f"\n[{i}/{len(all_qptiff_files)}] Processing file: {qptiff_file_path}")

        slide_level_metadata = {
            'original_filename': os.path.basename(qptiff_file_path),
            'file_path': qptiff_file_path
        }
        
        # Try to extract metadata
        try:
            with tifffile.TiffFile(qptiff_file_path) as tif:
                if tif.pages and len(tif.pages) > 0:
                    # Look for ImageDescription tag (270)
                    if 270 in tif.pages[0].tags:
                        metadata_tag = tif.pages[0].tags[270]
                        metadata_str = metadata_tag.value
                        
                        if isinstance(metadata_str, bytes):
                            # Try different encodings
                            for encoding in ['utf-16', 'utf-8', 'latin-1']:
                                try:
                                    metadata_str = metadata_str.decode(encoding)
                                    break
                                except UnicodeDecodeError:
                                    continue
                            else:
                                metadata_str = metadata_str.decode('utf-8', errors='ignore')
                        
                        slide_level_metadata.update(parse_metadata(metadata_str))
                    else:
                        print(f"  Warning: ImageDescription tag not found in {base_filename}")
                        slide_level_metadata['parser_error'] = 'ImageDescription tag not found'
                else:
                    print(f"  Warning: No pages found in {base_filename}")
                    slide_level_metadata['parser_error'] = 'No pages found'
        
        except Exception as e:
            print(f"  Error reading metadata for {base_filename}: {e}")
            slide_level_metadata['parser_error'] = str(e)

        # Extract patches
        file_patches_info = extract_patches(
            qptiff_file_path, OUTPUT_DIR, base_filename,
            patch_size, level, overlap, slide_level_metadata
        )
        
        if file_patches_info:
            master_patch_list.extend(file_patches_info)
            print(f"  -> Generated {len(file_patches_info)} patches")
        else:
            print(f"  -> No patches generated for {base_filename}")

    # Save results
    if master_patch_list:
        df = pd.DataFrame(master_patch_list)
        csv_path = os.path.join(OUTPUT_DIR, output_metadata)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total files processed: {len(all_qptiff_files)}")
        print(f"Total patches generated: {len(master_patch_list)}")
        print(f"Patches saved to: {OUTPUT_DIR}")
        print(f"Metadata saved to: {csv_path}")
        
        # Print summary statistics
        if len(master_patch_list) > 0:
            print(f"\n=== SUMMARY STATISTICS ===")
            print(f"Average patches per slide: {len(master_patch_list)/len(all_qptiff_files):.1f}")
            print(f"Unique slides: {df['slide_basename'].nunique()}")
            if 'ChannelName' in df.columns:
                print(f"Channels found: {df['ChannelName'].value_counts().to_dict()}")
    else:
        print("\nNo patches were generated from any files.")

if __name__ == '__main__':

    for i in range(149, 400):
        file_path = get_qptiff_path(QPTIFF_DIR, i)
        print(f"Trying to open: {file_path}")
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")  
            continue
        with tifffile.TiffFile(file_path) as tif: 
            # Example: read the first page as an array
            image = tif.pages[0].asarray()
            print(f"Loaded image shape for {i}: {image.shape}")
            # You can now process 'image' as needed
