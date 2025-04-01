#crop
    # # Get the color of the pixel at row 0, column 0

    # top_left_pixel = image.getpixel((0, 0))
    # print(f"Top-left pixel color: {top_left_pixel}")

    # # Get the colors of all four sides of the image
    # width, height = image.size
    # top_side = [image.getpixel((x, 0)) for x in range(width)]  # Top side
    # bottom_side = [image.getpixel((x, height - 1)) for x in range(width)]  # Bottom side
    # left_side = [image.getpixel((0, y)) for y in range(height)]  # Left side
    # right_side = [image.getpixel((width - 1, y)) for y in range(height)]  # Right side

    # # Combine all side pixels into a single list
    # all_side_pixels = top_side + bottom_side + left_side + right_side

    # # Use collections.Counter to count the occurrences of each color
    # color_counts = Counter(all_side_pixels)
    # print(f"Side pixel color counts: {color_counts}")

# matching

    # # Template Matching using edges
    # result = cv2.matchTemplate(big_map_edges, patch_edges, cv2.TM_CCOEFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # # Get match location
    # x, y = max_loc
    # h, w = patch.size[:2]

    # # Draw rectangle around matched area
    # # cv2.rectangle(big_map_gray, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # # Add 50-pixel margin while cropping
    # margin = 50
    # x1 = max(max_loc[0] - margin, 0)
    # y1 = max(max_loc[1] - margin, 0)
    # x2 = min(max_loc[0] + w + margin, np.array(big_map).shape[1])
    # y2 = min(max_loc[1] + h + margin, np.array(big_map).shape[0])

    # cropped_image = big_map_gray[y1:y2, x1:x2]  
    # print(cropped_image.shape)
    # cv2.imshow("Edge Matching", cropped_image)

    # # # Crop the region of interest (ROI)
    # cropped_image = big_map_gray[5199:5393, 1360:1520]
    # print("Cropped shape:   ", cropped_image.shape)
    # # Stack the cropped image and patch_gray side by side
    # # Resize patch_gray to match cropped image size for comparison
    # resized_patch_gray = cv2.resize(patch_gray, (cropped_image.shape[1], int(cropped_image.shape[1] * patch_gray.shape[0] / patch_gray.shape[1])))
    # print("resized shape: ", resized_patch_gray.shape)

    # # Calculate padding for top/bottom (height) and left/right (width)
    # top_padding = (cropped_image.shape[0] - resized_patch_gray.shape[0]) // 2
    # bottom_padding = cropped_image.shape[0] - resized_patch_gray.shape[0] - top_padding
    # left_padding = (cropped_image.shape[1] - resized_patch_gray.shape[1]) // 2
    # right_padding = cropped_image.shape[1] - resized_patch_gray.shape[1] - left_padding

    # # Pad the resized patch with zeroes (black padding) to match the target size
    # padded_patch_gray = cv2.copyMakeBorder(resized_patch_gray, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=0)

    # # Stack them horizontally
    # combined_image = np.hstack((cropped_image, padded_patch_gray))

    # cv2.imshow("Edge Matching", combined_image)
   
    # ### DOESNT WORK: COLOUR DIFF 
    # # Template matching
    # result = cv2.matchTemplate(big_map_gray, patch_gray, cv2.TM_CCOEFF_NORMED)

    # # Get best match location
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # # Draw a rectangle around the best match
    # h, w = patch.size[:2]
    # cv2.rectangle(np.array(big_map), max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)

    # # Add 50-pixel margin while cropping
    # margin = 50
    # x1 = max(max_loc[0] - margin, 0)
    # y1 = max(max_loc[1] - margin, 0)
    # x2 = min(max_loc[0] + w + margin, np.array(big_map).shape[1])
    # y2 = min(max_loc[1] + h + margin, np.array(big_map).shape[0])

    # # Crop the region of interest (ROI)
    # cropped_image = np.array(big_map)[y1:y2, x1:x2]

    # # Show result
    # cv2.imshow('Matched', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# # check image for teleport patches (if teleport, add new level)
# def contains_teleport_patch(image):
# # check pairwise images for overlap
# def check_overlap(image1, image2):
# # stitch images together based on overlap
# def stitch_images(image1, image2):
# 	# Placeholder for stitching logic
# 	return image1c
# 	return False
# 	return False
# 	return False