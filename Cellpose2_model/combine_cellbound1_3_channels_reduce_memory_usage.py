# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 15:30:41 2025

@author: Chenxi Zhou
"""

import tifffile as tiff
import numpy as np
import os
import psutil
import gc

# Paths
image1_path = "mosaic_Cellbound3_z3.tif"
image2_path = "mosaic_Cellbound1_z3.tif"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Open input images
image1 = tiff.memmap(image1_path, dtype=np.uint16)
image2 = tiff.memmap(image2_path, dtype=np.uint16)
height, width = image1.shape

# Output files
added_image_path = os.path.join(output_dir, "added_image.tif")
average_image_path = os.path.join(output_dir, "average_image.tif")
max_image_path = os.path.join(output_dir, "max_image.tif")

# reduce the size of chunk to 512
chunk_size = 512
num_chunks = height // chunk_size + (1 if height % chunk_size else 0)

# Pre-allocated output files 
added_image = tiff.memmap(added_image_path, shape=(height, width), dtype=np.uint16, bigtiff=True)
average_image = tiff.memmap(average_image_path, shape=(height, width), dtype=np.uint16, bigtiff=True)
max_image = tiff.memmap(max_image_path, shape=(height, width), dtype=np.uint16, bigtiff=True)

process = psutil.Process(os.getpid())

try:
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, height)

        # Process chunks and monitor memory
        # Here, Cellbound3 (Figure 1) is normalized by multiplying by 1.69 to make its signal comparable to Cellbound1.
        
        chunk1 = image1[start:end, :].astype(np.float32) * 1.69
        chunk2 = image2[start:end, :].astype(np.float32)
        
        added = (chunk1 + chunk2).clip(0, 65535).astype(np.uint16)  # add two channels
        avg = ((chunk1 + chunk2) / 2).clip(0, 65535).astype(np.uint16)   # average two channels
        max_val = np.maximum(chunk1, chunk2).astype(np.uint16)   # choose the maximum from two channels
        
        # write the output
        added_image[start:end, :] = added
        average_image[start:end, :] = avg
        max_image[start:end, :] = max_val
        
        # release memory
        del chunk1, chunk2, added, avg, max_val
        gc.collect()
        
        print(f"Chunk {i+1}/{num_chunks}, Memory: {process.memory_info().rss / 1024**3:.2f} GB")

finally:
    del added_image, average_image, max_image
    gc.collect()

print("Done！")