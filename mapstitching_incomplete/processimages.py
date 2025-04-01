from PIL import Image
import os
from collections import Counter
import cv2
import numpy as np

# load overall map from path
def load_img(path):
    if not os.path.exists(path):
        try:
            # If the file does not exist, create a blank image
            img = Image.new('RGBA', (1, 1), (255, 255, 255, 0))
            img.save(path)
            return Image.open(path)
        except:
            raise FileNotFoundError(f"The file at {path} does not exist.")
    return Image.open(path)

def save_img(img, imgdir, img_name = "output.png"):
    img.save(os.path.join(imgdir, img_name))

# crop image to game area
def crop_to_game_area(image, game_size = (1074, 966)):
    colormap = {"dialogue" :(248, 248, 248, 255), "outside_black": (0, 0, 0, 255), "outside_white": (243, 243, 243, 255)}
    # crop edges if colour is outside_black or outside_white
    l, r, t, b = 0, 0, 0, 0
    width, height = image.size
    while all([image.getpixel((x, t)) == colormap["outside_black"] or image.getpixel((x, t)) == colormap["outside_white"] for x in range(width)]):
        t += 1
    while all([image.getpixel((x, height - b - 1)) == colormap["outside_black"] or image.getpixel((x, height - b - 1)) == colormap["outside_white"] for x in range(width)]):
        b += 1
    while all([image.getpixel((l, y)) == colormap["outside_black"] or image.getpixel((l, y)) == colormap["outside_white"] for y in range(height)]):
        l += 1
    while all([image.getpixel((width - r - 1, y)) == colormap["outside_black"] or image.getpixel((width - r - 1, y)) == colormap["outside_white"] for y in range(height)]):
        r += 1
    print(l, r, t, b)

    #crop image by l r t b
    image = image.crop((l, t, width - r, height - b))
    print(image.size, game_size)
    print("Check image_size is correct: ", image.size == game_size)

    return image


# check image for dialogue (if dialogue, ignore)
def contains_dialogue(image):
    colormap = {"dialogue" :(248, 248, 248, 255), "outside_black": (0, 0, 0, 255), "outside_white": (243, 243, 243, 255)}
    width, height = image.size
    bottom_side = [image.getpixel((x, height - 1)) for x in range(width)]  # Bottom side
    if any([pixel == colormap["dialogue"] for pixel in bottom_side]):
        #dialogue present, crop out dialogue
        b = 0
        while any([image.getpixel((x, height - b - 1)) == colormap["dialogue"] for x in range(width)]):
            b += 1
        print(height, b, b/height)
        image = image.crop((0, 0, width, height - b))
        return (image, True)
    else:
        #no dialogue present
        return (image, False)
     #check image for dialogue at bottom

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
    # Apply Canny Edge Detection
    big_map_edges = cv2.Canny(big_map_gray, 50, 200)
    patch_edges = cv2.Canny(patch_gray, 50, 200)
    print(big_map_edges.shape, patch_edges.shape)
    # # Resize patch_gray to match cropped image size for comparison
    resized_patch_gray = cv2.resize(patch_edges, (big_map_edges.shape[1], int(big_map_edges.shape[1] * patch_edges.shape[0] / patch_edges.shape[1])))
    print("resized shape: ", resized_patch_gray.shape)

    # Calculate padding for top/bottom (height) and left/right (width)
    top_padding = (big_map_edges.shape[0] - resized_patch_gray.shape[0]) // 2
    bottom_padding = big_map_edges.shape[0] - resized_patch_gray.shape[0] - top_padding
    left_padding = (big_map_edges.shape[1] - resized_patch_gray.shape[1]) // 2
    right_padding = big_map_edges.shape[1] - resized_patch_gray.shape[1] - left_padding

    # Pad the resized patch with zeroes (black padding) to match the target size
    padded_patch_gray = cv2.copyMakeBorder(resized_patch_gray, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=0)

    # Stack them horizontally
    combined_image = np.hstack((big_map_edges, padded_patch_gray))

    cv2.imshow("Canny images", combined_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(os.getcwd())
map_path = os.path.join(os.getcwd(),'mapstitching_incomplete/manual_stitch.png')
map_img = load_img(map_path)
# counter_map = Counter(map_img.getdata())
# print(counter_map)
# print((248, 248, 248, 255) in counter_map) # False: can use dialogue checker
# print((0, 0, 0, 255) in counter_map) # True: careful of matching blacks
# print((243, 243, 243, 255) in counter_map) # False: Pyboy interface (Grey) is ok to crop out

sprite_path = os.path.join(os.getcwd(),'mapstitching_incomplete/characters_front_house_notrans.png')
sprite_img = load_img(sprite_path)

test_path = os.path.join(os.getcwd(),'mapstitching_incomplete/images/House4.png')
test_img = load_img(test_path)
cropped_img = crop_to_game_area(test_img)
crop2_image, has_dialogue = contains_dialogue(cropped_img)
print(crop2_image.size)
# matched_coords = match_image(sprite_img, crop2_image)
# save_img(matched_coords, os.path.join(os.getcwd(),'mapstitching_incomplete'), "manual_stitch.png")


# matched_coords = match_sprites(crop2_image, sprite_img)
# cropped_img.show()


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


import cv2
import numpy as np

def find_largest_overlap_region(template, original_map, patch_size=100, threshold=0.8):
    """
    Dynamically expands patch size to stitch together the largest overlapping region.

    :param template: The template image (grayscale)
    :param img_gray: The target image (grayscale)
    :param patch_size: Initial patch size for scanning
    :param threshold: Similarity threshold for expanding overlap
    :return: Overlap mask (same size as img_gray)
    """

    original_map = np.array(original_map)
    new_map = np.array(template)
    # img_rgb = cv2.imread(screenascanner)
    # template = cv2.imread(elementachercher, cv2.IMREAD_UNCHANGED)
    # Check if the image has an alpha channel
    if original_map.shape[-1] == 4:
        img_gray = cv2.cvtColor(original_map, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        img_gray = cv2.cvtColor(original_map, cv2.COLOR_BGR2GRAY)

    if new_map.shape[-1] == 4:
        template = cv2.cvtColor(new_map, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        template = cv2.cvtColor(new_map, cv2.COLOR_BGR2GRAY) 

    template_h, template_w = template.shape
    img_h, img_w = img_gray.shape

    # Initialize overlap mask (same size as target image)
    overlap_mask = np.zeros_like(img_gray, dtype=np.uint8)
    cv2.imshow("gray", img_gray)
    cv2.imshow("tem", template)

    # Find the best initial match using template matching
    result = cv2.matchTemplate(img_gray, template, method=cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        print("No significant overlap found.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return overlap_mask  # Return empty mask

    # Start expanding from the best match
    x, y = max_loc  # Initial match (top-left corner)
    stitched_region = [(x, y, patch_size, patch_size)]  # Store stitched areas

    # Dynamically expand the region
    expansion_step = patch_size // 2  # Expand by half the patch size
    while expansion_step > 5:  # Stop expanding when it gets too small
        expanded = False
        new_regions = []

        for (x, y, w, h) in stitched_region:
            # Try expanding in four directions
            candidates = [
                (x - expansion_step, y, w + expansion_step, h),  # Expand left
                (x, y - expansion_step, w, h + expansion_step),  # Expand up
                (x, y, w + expansion_step, h),  # Expand right
                (x, y, w, h + expansion_step)   # Expand down
            ]

            for (nx, ny, nw, nh) in candidates:
                # Ensure new region is within image bounds
                if nx < 0 or ny < 0 or nx + nw > img_w or ny + nh > img_h:
                    continue

                # Extract new region and compare similarity
                patch = img_gray[ny:ny + nh, nx:nx + nw]
                if patch.shape[0] == 0 or patch.shape[1] == 0:
                    continue

                res = cv2.matchTemplate(img_gray, patch, method=cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)

                if score > threshold:  # Expand if similarity is high
                    new_regions.append((nx, ny, nw, nh))
                    overlap_mask[ny:ny + nh, nx:nx + nw] = 255  # Mark as overlapping
                    expanded = True

        # Update stitched regions
        if expanded:
            stitched_region.extend(new_regions)
        else:
            expansion_step //= 2  # Reduce expansion step
    # Apply mask to highlight overlap
    highlighted_overlap = cv2.bitwise_and(img_gray, img_gray, mask=overlap_mask)

    # Save and display results
    # cv2.imwrite("overlap_mask.png", overlap_mask)
    # cv2.imwrite("highlighted_overlap.png", highlighted_overlap)
    # cv2.imshow("Largest Overlap Mask", overlap_mask)
    cv2.imshow("Highlighted Overlap", highlighted_overlap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return overlap_mask

# Find the largest stitched overlap region
overlap_mask = find_largest_overlap_region(sprite_img, crop2_image)


# # Stitch maps together
# stitched_map = stitch_pixel_maps(sprite_img, crop2_image)

# # Align and merge the maps
# updated_map = align_and_stitch_maps(sprite_img, crop2_image)

# Save and display the updated map
# cv2.imwrite("updated_map.png", updated_map)
