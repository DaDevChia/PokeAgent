

def tryfoundobject(big_map, patch):
    img_rgb = np.array(big_map)
    template = np.array(patch)
    # img_rgb = cv2.imread(screenascanner)
    # template = cv2.imread(elementachercher, cv2.IMREAD_UNCHANGED)
    # Check if the image has an alpha channel
    if img_rgb.shape[-1] == 4:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    if template.shape[-1] == 4:
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #resize img_gray to 100/16, 115/16
    # img_gray = cv2.resize(img_gray, (int(img_gray.shape[1] * 16/100), int(img_gray.shape[0] * 16/100)), interpolation=cv2.INTER_CUBIC)

    # #make it bigger so I can actually see it
    # img_gray = cv2.resize(img_gray, (int(img_gray.shape[1] * 100/16), int(img_gray.shape[0] * 100/16)), interpolation=cv2.INTER_CUBIC)
    # template = cv2.resize(template, (int(template.shape[1] * 107/16), int(template.shape[0] * 117/16)))
    
    
    cv2.imshow("imggray", img_gray)
    cv2.imshow("template", template)
    print(img_gray.shape, template.shape)

    # single match
    # w, h = template.shape[::-1]
    # res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    # threshold = 0.8
    # loc = np.where( res >= threshold)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    #     if pt != None:
    #         print(elementachercher+" matched")
    #         break
    #     else:
    #         continue

    # multiple matches @ Patch size (100x100)
    patch_size = 100
    template_h, template_w = template.shape
    matchfound = False

    # Iterate through the template, breaking it into 100x100 patches
    for y in range(0, template_h, patch_size):
        if matchfound:
            break
        for x in range(0, template_w, patch_size):
            if matchfound:
                break
            # Ensure patch does not exceed template bounds
            patch = template[y:y + patch_size, x:x + patch_size]
            cv2.imshow("template", patch)
            print(patch.shape)
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue  # Skip if patch is smaller than 100x100

            #max-diff supression
            template_mask = (patch >= 128).astype(np.uint8)

            # theoretical maximum difference for this template
            # (largest possible difference in every pixel and every color channel)
            #maxdiff = template_mask.sum() * 255**2
            maxdiff = np.maximum(patch, 255-patch) # worst case difference
            # maxdiff *= np.uint8(template_mask)[:,:,np.newaxis] # apply mask
            maxdiff *= np.uint8(template_mask)[:,:]
            maxdiff = (maxdiff.astype(np.uint32)**2).sum() # sum of squared differences
            print("maximal difference:", maxdiff)

            sqdiff_values = cv2.matchTemplate(image=img_gray, templ=patch, method=cv2.TM_SQDIFF, mask=template_mask)

            # only supposed to kill slopes of peaks, not anything that's below threshold
            # radius 1 already effective if data is smooth; removes anything on a slope
            # have to pass `-sqdiff_values` because sqdiff_values contains minima, this looks for maxima
            peakmask = non_maximum_suppression(-sqdiff_values, radius=1)

            # kill everything below threshold too
            threshold = 0.05 # applies to sum of SQUARED differences
            peakmask[sqdiff_values > maxdiff * threshold] = False

            # (i,j)
            loc = np.array(np.where(peakmask)).T
            print("matches:", len(loc))
            if len(loc) > 0:
                matchfound = True
                # print biggest difference
                # Find the point with the biggest normalized diff
                max_pt = max(loc, key=lambda pt: sqdiff_values[pt[0], pt[1]] / maxdiff)
                # for pt in loc:
                (i,j) = max_pt
                print("pt", max_pt, "diff", (sqdiff_values[i,j] / maxdiff))
                cv2.rectangle(img_rgb, (j,i), (j + patch_size, i + patch_size), (0,0,255), 2)

            # # Perform template matching for each patch
            # res = cv2.matchTemplate(img_gray, patch, cv2.TM_CCOEFF_NORMED)
            # threshold = 0.8
            # loc = np.where(res >= threshold)

            # # Draw rectangles for each matched patch
            # for pt in zip(*loc[::-1]):
            #     cv2.rectangle(img_rgb, pt, (pt[0] + patch_size, pt[1] + patch_size), (0, 0, 255), 2)
            #     print(f"Patch at ({x},{y}) matched at {pt}")
    cv2.imshow("Canny images", img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# try to match character_front to House3.png
sprite_path = os.path.join(os.getcwd(),'mapstitching_incomplete/house3patchtest.png')
sprite_path = os.path.join(os.getcwd(),'mapstitching_incomplete/characters_front_house_notrans.png')
sprite_path = os.path.join(os.getcwd(),'mapstitching_incomplete/images/House6.png')
sprite_img = load_img(sprite_path)
# tryfoundobject(sprite_img, crop2_image)

# tryfoundobject(sprite_path, test_path)

def align_and_stitch_maps(original_map, new_map):
    """
    Aligns and stitches a new map onto the original map using feature matching.
    
    :param original_map: The existing map (numpy array)
    :param new_map: The newly loaded map section (numpy array)
    :return: The updated stitched map
    """
    original_map = np.array(original_map)
    new_map = np.array(new_map)
    # img_rgb = cv2.imread(screenascanner)
    # template = cv2.imread(elementachercher, cv2.IMREAD_UNCHANGED)
    # Check if the image has an alpha channel
    if original_map.shape[-1] == 4:
        gray_old = cv2.cvtColor(original_map, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        gray_old = cv2.cvtColor(original_map, cv2.COLOR_BGR2GRAY)

    if new_map.shape[-1] == 4:
        gray_new = cv2.cvtColor(new_map, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        gray_new = cv2.cvtColor(new_map, cv2.COLOR_BGR2GRAY)    

    # Detect keypoints and descriptors using ORB (can use SIFT/SURF if needed)
    orb = cv2.ORB_create(nfeatures=500)  
    keypoints_old, descriptors_old = orb.detectAndCompute(gray_old, None)
    keypoints_new, descriptors_new = orb.detectAndCompute(gray_new, None)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_old, descriptors_new)

    # Sort matches based on distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    old_pts = np.float32([keypoints_old[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    new_pts = np.float32([keypoints_new[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    H, mask = cv2.findHomography(new_pts, old_pts, cv2.RANSAC)

    # Warp the new map to align it with the old map
    h_old, w_old, _ = original_map.shape
    warped_new_map = cv2.warpPerspective(new_map, H, (w_old, h_old))

    # Blend the old and new maps (keeping static features)
    blended_map = np.where(warped_new_map > 0, warped_new_map, original_map)
    cv2.imshow("Updated Map", blended_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blended_map

def stitch_pixel_maps(old_map, new_map):
    """
    Stitches two pixel-based maps while handling moving sprites.
    
    :param old_map: Old map as a NumPy array (BGR)
    :param new_map: New map as a NumPy array (BGR)
    :return: Stitched map as a NumPy array
    """
    original_map = np.array(old_map)
    new_map = np.array(new_map)
    # img_rgb = cv2.imread(screenascanner)
    # template = cv2.imread(elementachercher, cv2.IMREAD_UNCHANGED)
    # Check if the image has an alpha channel
    if original_map.shape[-1] == 4:
        gray_old = cv2.cvtColor(original_map, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        gray_old = cv2.cvtColor(original_map, cv2.COLOR_BGR2GRAY)

    if new_map.shape[-1] == 4:
        gray_new = cv2.cvtColor(new_map, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        gray_new = cv2.cvtColor(new_map, cv2.COLOR_BGR2GRAY) 

    # Compute absolute difference between maps
    diff = cv2.absdiff(gray_old, gray_new)

    # Threshold the difference to detect movement (ignore small changes)
    _, movement_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Expand the movement mask slightly (to ensure full sprite capture)
    kernel = np.ones((3, 3), np.uint8)
    movement_mask = cv2.dilate(movement_mask, kernel, iterations=1)

    # Create a 3-channel mask for blending
    movement_mask_colored = cv2.cvtColor(movement_mask, cv2.COLOR_GRAY2BGR)

    # Use the mask to merge the maps:
    # - Keep old_map where there is no movement
    # - Replace with new_map where movement is detected
    stitched_map = np.where(movement_mask_colored > 0, new_map, old_map)
    # Save and display results
    cv2.imshow("Stitched Map", stitched_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return stitched_map

def split_into_patches(img, grid_size=(9, 10)):
    """Splits an image into grid of (h, w)."""
    img_h, img_w = img.shape
    gh, gw = grid_size
    ph, pw = img_h // gh, img_w // gw  # Patch height and width
    patches = []

    # Ensure the image dimensions are divisible by patch size
    for y in range(0, (ph+1)*gh, ph):
        row_patches = []
        for x in range(0, (pw+1)*gw, pw):
            rangeh = y+ph if y+ph<img_h else img_h
            rangew = x+pw if x+pw<img_w else img_w
            row_patches.append(img[y:rangeh, x:rangew])
        patches.append(row_patches)

    return np.array(patches)  # Shape: (rows, cols, 9, 10)

def remove_outer_rim(patches):
    """Removes the outermost patches, keeping only the inner ones."""
    return patches[1:-1, 1:-1]  # Removes first & last row/col

def match_patches(inner_patches, template_patches, threshold=0.8):
    """Compares inner patches with template patches to determine movement."""
    movement_map = np.zeros(inner_patches.shape[:2], dtype=str)  # Store directions

    for i in range(inner_patches.shape[0]):
        for j in range(inner_patches.shape[1]):
            best_match = None
            best_score = -1

            # Compare with all possible shifts (left, right, up, down)
            shifts = {
                "left": (i, j-1),
                "right": (i, j+1),
                "up": (i-1, j),
                "down": (i+1, j)
            }

            for direction, (ni, nj) in shifts.items():
                if 0 <= ni < inner_patches.shape[0] and 0 <= nj < inner_patches.shape[1]:
                    patch = inner_patches[i, j]
                    shifted_patch = template_patches[ni, nj]

                    res = cv2.matchTemplate(patch, shifted_patch, cv2.TM_CCOEFF_NORMED)
                    _, score, _, _ = cv2.minMaxLoc(res)

                    if score > best_score:
                        best_score = score
                        best_match = direction

            # Assign direction if confidence is high
            if best_score >= threshold:
                movement_map[i, j] = best_match
            else:
                movement_map[i, j] = "none"

    return movement_map


def stitch_images(img_gray, template, patch_size=(9,10), threshold=0.8):
    """Stitch img_gray and template together based on detected movement."""
    patches_gray = split_into_patches(img_gray, patch_size)
    patches_template = split_into_patches(template, patch_size)

    inner_patches_gray = remove_outer_rim(patches_gray)
    inner_patches_template = remove_outer_rim(patches_template)

    # Step 1: Detect movement direction
    movement_map = match_patches(inner_patches_gray, inner_patches_template, threshold)

    # Step 2: Initialize stitched image
    stitched_h = img_gray.shape[0] + patch_size[0]  # Allow expansion in any direction
    stitched_w = img_gray.shape[1] + patch_size[1]
    stitched_image = np.full((stitched_h, stitched_w), 255, dtype=np.uint8)  # Fill with white

    # Step 3: Copy original image into the center
    offset_h = patch_size[0] // 2
    offset_w = patch_size[1] // 2
    stitched_image[offset_h:offset_h+img_gray.shape[0], offset_w:offset_w+img_gray.shape[1]] = img_gray

    # Step 4: Stitch based on movement direction
    for i in range(inner_patches_gray.shape[0]):
        for j in range(inner_patches_gray.shape[1]):
            direction = movement_map[i, j]
            patch = inner_patches_template[i, j]  # Use template's patch

            # Calculate new position
            y = offset_h + i * patch_size[0]
            x = offset_w + j * patch_size[1]

            if direction == "left":
                stitched_image[y:y+patch_size[0], x-patch_size[1]:x] = patch
            elif direction == "right":
                stitched_image[y:y+patch_size[0], x+patch_size[1]:x+2*patch_size[1]] = patch
            elif direction == "up":
                stitched_image[y-patch_size[0]:y, x:x+patch_size[1]] = patch
            elif direction == "down":
                stitched_image[y+patch_size[0]:y+2*patch_size[0], x:x+patch_size[1]] = patch

    return stitched_image





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