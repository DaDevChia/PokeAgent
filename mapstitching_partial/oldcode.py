def clean_patches(patch_heights, patch_widths, img):
    """
    Generates patches based on the provided image and patch dimensions.

    Parameters:
        patch_heights (list): List containing heights of patches [ph, ph2].
        patch_widths (list): List containing widths of patches [pw, pw2].
        img (numpy array): The original image for reference.

    Returns:
        patches (list of lists): A 2D list of generated patches.
    """
    patches = []
    img_h, img_w = img.shape[:2]
    curr_y = 0

    for ph in patch_heights:
        curr_x = 0
        row_patches = []
        for pw in patch_widths:
            # Extract patch from the image
            patch = img[curr_y:curr_y + ph, curr_x:curr_x + pw]
            row_patches.append(patch)
            curr_x += pw
        patches.append(row_patches)
        curr_y += ph

    return patches

def stitch_image_old(patches):
    """
    Reconstructs an image from patches with varying sizes.

    Parameters:
        patches (list of lists): A 2D list where each element is a cv2 image patch.

    Returns:
        stitched_img (numpy array): The reconstructed full image.
    """
    patches = clean_patches(patches)
    print([Counter([x.shape for x in row]) for row in patches])
    rows, cols = len(patches), len(patches[0])  # Get grid size
    
    # Determine the total stitched image height and width dynamically
    stitched_h = sum(patches[i][0].shape[0] for i in range(rows))  # Sum of patch heights per row
    stitched_w = sum(patches[0][j].shape[1] for j in range(cols))  # Sum of patch widths per column
    
    # Create an empty canvas for the stitched image
    stitched_img = np.zeros((stitched_h, stitched_w), dtype=patches[0][0].dtype)

    # Fill in the patches at correct positions
    y_offset = 0
    for i in range(rows):
        x_offset = 0
        for j in range(cols):
            patch = patches[i][j]
            h, w = patch.shape[:2]  # Get patch height and width
            stitched_img[y_offset:y_offset + h, x_offset:x_offset + w] = patch
            x_offset += w  # Move right for the next patch
        y_offset += patches[i][0].shape[0]  # Move down for the next row

    return stitched_img


    # if canvas is None:
    #     print(f"New image {img_path.split('/')[-1]} due to no match.")
    #     crop_image_cv, crop_color = next_crop_image_cv, next_color
    #     cv2.imshow(f"new image {img_path.split('/')[-1]}", resize_image(crop_color))
    # else:
    #     stitched_image_mid = stitch_images(canvas, array1_coord, array2_coord, crop_color, next_color)
    #     cv2.imshow(f"merged {img_path.split('/')[-1]}", resize_image(stitched_image_mid))
    #     crop_image_cv, crop_color = next_crop_image_cv, next_color


# print(os.getcwd())
# map_path = os.path.join(os.getcwd(),'mapstitching_incomplete/manual_stitch.png')
# map_img = load_img(map_path)
# counter_map = Counter(map_img.getdata())
# print(counter_map)
# print((248, 248, 248, 255) in counter_map) # False: can use dialogue checker
# print((0, 0, 0, 255) in counter_map) # True: careful of matching blacks
# print((243, 243, 243, 255) in counter_map) # False: Pyboy interface (Grey) is ok to crop out

# sprite_path = os.path.join(os.getcwd(),'mapstitching_incomplete/characters_front_house_notrans.png')
# sprite_img = load_img(sprite_path)

# test_path = os.path.join(os.getcwd(),'mapstitching_incomplete/images/House4.png')
# test_img = load_img(test_path)
# cropped_img = crop_to_game_area(test_img)
# crop2_image, has_dialogue = contains_dialogue(cropped_img)
# print(crop2_image.size)

# if has_dialogue:
#     print("TRUE")

# crop2_image_cv = convertPILToCV2(crop2_image)
# crop2_image_cv_patches = split_into_patches(crop2_image_cv, grid_size=(9, 10))
# print(crop2_image_cv_patches.shape)

# crop2_image_cv_gridded = draw_grid_from_patch_array(crop2_image_cv, crop2_image_cv_patches)

# next_path = os.path.join(os.getcwd(),'mapstitching_incomplete/images/House5.png')
# next_img = load_img(next_path)
# next_img = crop_to_game_area(next_img)
# next_crop_image, next_has_dialogue = contains_dialogue(next_img)
# next_crop_image_cv = convertPILToCV2(next_crop_image)

# next_patches = split_into_patches(next_crop_image_cv, grid_size=(9, 10))
# print(next_patches.shape)

# next_patches_gridded = draw_grid_from_patch_array(next_crop_image_cv, next_patches)


def draw_grid_from_patch_array(image, patches, color=(255, 0, 0), thickness=1):
    """
    Draws a grid on the image using the actual patches array from split_into_patches.

    Parameters:
        image (numpy array): The input image.
        patches (numpy array): The split patches array from split_into_patches().
        color (tuple): Color of the grid lines (default is green).
        thickness (int): Thickness of grid lines (default is 1).

    Returns:
        numpy array: Image with the grid drawn.
    """
    grid_img = image.copy()
    gh, gw = patches.shape  # Get patch grid dimensions

    currx, curry = 0, 0
    for row in range(gh):
        if row%2 == 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        for col in range(gw):
            # Calculate top-left and bottom-right corners of each patch
            y1, x1 = patches[row][col].shape
            # Draw rectangle on the image
            cv2.rectangle(grid_img, (currx, curry), (x1, curry+y1), color, thickness)
            currx = currx+x1
        y1, x1 = patches[row][-1].shape
        cv2.rectangle(grid_img, (currx, curry), (x1, y1), color, thickness)
        curry = curry+y1
        currx = 0
    return grid_img

def match_patches(patch1, patch2):
    """Uses cv2.matchTemplate to compare two patches and return the match score."""
    if patch1 is None or patch2 is None or np.all(patch1 == 0) or np.all(patch2 == 0):
        return -1  # Skip matching if any patch is missing (None)
    try:
        result = cv2.matchTemplate(patch1, patch2, cv2.TM_CCOEFF_NORMED)
    except:
        try:
            result = cv2.matchTemplate(patch2, patch1, cv2.TM_CCOEFF_NORMED)
        except Exception as e:
            print(f"Error matching patches: {e}")
            return -1   
    return result.max()  # Extract the best match score

def determine_displacement_and_stitch_OLD(array1, array2):
    """
    Determines the best displacement direction by generating patches for array1 and array2
    based on their respective patch settings. Returns the coordinates for stitching array2
    to array1 and the direction.

    Parameters:
        array1 (numpy array): First image.
        array2 (numpy array): Second image.
        array1_patch_settings (tuple): Patch settings for array1 ([ph, ph2], [pw, pw2], grid_size).
        array2_patch_settings (tuple): Patch settings for array2 ([ph, ph2], [pw, pw2], grid_size).

    Returns:
        coordinates (tuple): Coordinates for stitching array2 to array1.
        best_direction (str): The determined displacement direction ('up', 'down', 'left', 'right').
    """
    best_score = -1
    best_direction = None
    best_coordinates = None

    # Check Up (array2 is above array1)
    up_scores = [
        match_patches(array1_patches[i][j], array2_patches[i + 1][j])
        for i in range(len(array1_patches) - 1)
        for j in range(len(array1_patches[0]))
    ]
    up_score = np.mean(up_scores) if up_scores else -1
    if up_score > best_score:
        best_score = up_score
        best_direction = "up"
        best_coordinates = (0, -1)  # Relative position of array2 to array1

    # Check Down (array2 is below array1)
    down_scores = [
        match_patches(array1_patches[i + 1][j], array2_patches[i][j])
        for i in range(len(array1_patches) - 1)
        for j in range(len(array1_patches[0]))
    ]
    down_score = np.mean(down_scores) if down_scores else -1
    if down_score > best_score:
        best_score = down_score
        best_direction = "down"
        best_coordinates = (0, 1)

    # Check Left (array2 is left of array1)
    left_scores = [
        match_patches(array1_patches[i][j], array2_patches[i][j + 1])
        for i in range(len(array1_patches))
        for j in range(len(array1_patches[0]) - 1)
    ]
    left_score = np.mean(left_scores) if left_scores else -1
    if left_score > best_score:
        best_score = left_score
        best_direction = "left"
        best_coordinates = (-1, 0)

    # Check Right (array2 is right of array1)
    right_scores = [
        match_patches(array1_patches[i][j + 1], array2_patches[i][j])
        for i in range(len(array1_patches))
        for j in range(len(array1_patches[0]) - 1)
    ]
    right_score = np.mean(right_scores) if right_scores else -1
    if right_score > best_score:
        best_score = right_score
        best_direction = "right"
        best_coordinates = (1, 0)

    return best_coordinates, best_direction

def match_sprites(patch, sprites):
    # Convert PIL image to BGR (OpenCV uses BGR format)
    if sprites.mode == "RGBA":
        sprites = sprites.convert("RGB")
    if patch.mode == "RGBA":
        patch = patch.convert("RGB")
    sprites_cv = cv2.cvtColor(np.array(sprites), cv2.COLOR_RGB2BGR)
    patch_cv = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)

    # Convert to grayscale (optional but improves matching)
    sprites_gray = cv2.cvtColor(sprites_cv, cv2.COLOR_BGR2GRAY)
    patch_gray = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2GRAY)     

    # ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(patch_gray, None)
    kp2, des2 = orb.detectAndCompute(sprites_gray, None)

    # BFMatcher (Brute Force Matching)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    matched_img = cv2.drawMatches(patch_gray, kp1, sprites_gray, kp2, matches[:10], None, flags=2)

    cv2.imshow("ORB Feature Matching", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def remove_sprites(raw_image):
    # Convert PIL image to BGR (OpenCV uses BGR format)
    if raw_image.mode == "RGBA":
        raw_image = raw_image.convert("RGB")
    raw_image_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale (optional but improves matching)
    raw_image_gray = cv2.cvtColor(raw_image_cv, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Detection
    edges = cv2.Canny(raw_image_gray, 50, 200)

    # Show result
    cv2.imshow('Canny Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edges

def non_maximum_suppression(values, radius=1):
    local_area = cv2.dilate(values, kernel=None, iterations=radius)
    mask = (values == local_area)
    return mask

def match_image(patch, big_map):
    # check for uninitialized big_map
    if big_map.size == (1,1):
        edgemap = remove_sprites(patch)
    # Convert PIL image to BGR (OpenCV uses BGR format)
    # if big_map.mode == "RGBA":
    #     big_map = big_map.convert("RGB")
    # if patch.mode == "RGBA":
    #     patch = patch.convert("RGB")
    big_map = np.array(big_map)
    patch = np.array(patch)
    if big_map.shape[-1] == 4:
        big_map_gray = cv2.cvtColor(big_map, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        big_map_gray = cv2.cvtColor(big_map, cv2.COLOR_BGR2GRAY)
    if patch.shape[-1] == 4:
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)        
    # big_map_cv = cv2.cvtColor(np.array(big_map), cv2.COLOR_RGB2BGR)
    # patch_cv = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)

    # Convert to grayscale (optional but improves matching)
    # big_map_gray = cv2.cvtColor(big_map_cv, cv2.COLOR_BGR2GRAY)
    # # Crop the region of interest (ROI)
    # big_map_gray = big_map_gray[5199:5393, 1360:1520]

    # patch_gray = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2GRAY)

    # # Template matching


    picture = big_map_gray
    template =  patch_gray

    print(template.shape, big_map_gray.shape)

    assert template.shape[:2] == (117, 107), "template ought to be pixel perfect"
    # assert template.shape[2] == 4, "template contains no alpha channel!"

    cv2.imshow("bigmap", big_map_gray)
    cv2.imshow("patch_cv", patch_gray)

    # 54x54 empirically
    template = cv2.resize(template, dsize=(54, 54), interpolation=cv2.INTER_CUBIC)
    (th, tw) = template.shape[:2]

    # separate alpha channel, construct binary mask for matchTemplate
    # template_color = template[:,:,:3] if len(template.shape) == 3 else template
    # template_alpha = template[:,:,3] if len(template.shape) == 3 else template template[:,:,3]
    # template_mask = (template_mask >= 128).astype(np.uint8)
    
    template_color = template
    template_alpha = np.ones(template.shape, dtype=np.uint8) * 255  # Default opaque
    template_alpha[template > 250] = 0  # Make white areas transparent    
    template_mask = (template >= 128).astype(np.uint8)

    # theoretical maximum difference for this template
    # (largest possible difference in every pixel and every color channel)
    #maxdiff = template_mask.sum() * 255**2
    maxdiff = np.maximum(template_color, 255-template_color) # worst case difference
    # maxdiff *= np.uint8(template_mask)[:,:,np.newaxis] # apply mask
    maxdiff *= np.uint8(template_mask)[:,:]
    maxdiff = (maxdiff.astype(np.uint32)**2).sum() # sum of squared differences
    print("maximal difference:", maxdiff)

    #cv2.imshow("picture", picture)
    # cv2.imshow("template",
    #     np.hstack([template_color, cv2.cvtColor(template_alpha, cv2.COLOR_GRAY2BGR)])
    # )

    sqdiff_values = cv2.matchTemplate(image=picture, templ=template_color, method=cv2.TM_SQDIFF, mask=template_mask)

    # only supposed to kill slopes of peaks, not anything that's below threshold
    # radius 1 already effective if data is smooth; removes anything on a slope
    # have to pass `-sqdiff_values` because sqdiff_values contains minima, this looks for maxima
    peakmask = non_maximum_suppression(-sqdiff_values, radius=1)

    # kill everything below threshold too
    threshold = 0.10 # applies to sum of SQUARED differences
    peakmask[sqdiff_values > maxdiff * threshold] = False

    # (i,j)
    loc = np.array(np.where(peakmask)).T

    for pt in loc:
        (i,j) = pt
        print("pt", pt, "diff", (sqdiff_values[i,j] / maxdiff))
        cv2.rectangle(picture, (j,i), (j + tw, i + th), (0,0,255), 2)

    # cv2.imwrite("61a178bd038544749c64310a9812b5e35217fdd1-out.png", picture)
    cv2.imshow("picture", picture)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()

    # print(big_map_gray.shape, patch_gray.shape)

    # w, h = patch_gray.shape
    # res = cv2.matchTemplate(big_map_gray,patch_gray,cv2.TM_CCOEFF_NORMED)
    # threshold = 0.8
    # loc = np.where( res >= threshold)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(big_map_cv, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    #     if pt != None:
    #         print(big_map_cv+" matched")
    #         break
    #     else:
    #         continue

    # cv2.imshow("Canny images", big_map_cv)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def yeet():
    return None

def stitchImage(oldcode):
    # #stitch top
    # if array2_coord[0] < 0:
    #     if array2_coord[1] > 0:
    #         stitchedxoffsets = [array2_coord[1], basex-array2_coord[1]+array2.shape[1]] if basex-array2_coord[1]+array2.shape[1]>0 else [array2_coord[1], 0]
    #     if array2_coord[1] < 0:
    #         stitchedxoffsets = [0, basex-array2_coord[1]-array2.shape[1]] if basex-array2_coord[1]+array2.shape[1]>0 else [0, 0]

    #if array 2 is up from array 1: array1_coord = (0, pw)

    # Stitch array2 to array1 based on the best match location
    # Calculate the new dimensions for stitched_array
    new_height = max(array1.shape[0], best_coordinates[1] + array2.shape[0])
    new_width = max(array1.shape[1], best_coordinates[0] + array2.shape[1])

    # Create a blank canvas with the new dimensions
    stitched_array = np.zeros((new_height, new_width), dtype=array1.dtype)

    # Copy array1 into the canvas
    stitched_array[:array1.shape[0], :array1.shape[1]] = array1

    # Calculate the actual coordinates for array2
    x, y = best_coordinates
    actual_x = max(0, x - pw)
    actual_y = max(0, y - ph)

    # Overlay array2 onto stitched_array, filling black for missing rows/columns
    stitched_array[actual_y:actual_y + array2.shape[0], actual_x:actual_x + array2.shape[1]] = array2

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