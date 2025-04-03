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
def crop_to_game_area(image, game_size = (966, 970)):
    colormap = {"dialogue" :(248, 248, 248, 255), "outside_black": (0, 0, 0, 255), "outside_white": (243, 243, 243, 255)}
    # crop edges if colour is outside_black or outside_white
    l, r, t, b = 0, 0, 0, 0
    width, height = image.size
    thresh = 150 #pixels to exceed to be counted as outside
    while sum([image.getpixel((x, t)) == colormap["outside_black"] or image.getpixel((x, t)) == colormap["outside_white"] for x in range(width)]) >thresh:
        t += 1
    while sum([image.getpixel((x, height - b - 1)) == colormap["outside_black"] or image.getpixel((x, height - b - 1)) == colormap["outside_white"] or image.getpixel((x, height - b - 1)) in [(19,19,19,255),(18,18,18,255),(17,17,17,255)] for x in range(width)])>thresh:
        b += 1
    while sum([image.getpixel((l, y)) == colormap["outside_black"] or image.getpixel((l, y)) == colormap["outside_white"] for y in range(height)])>thresh:
        l += 1
    while sum([image.getpixel((width - r - 1, y)) == colormap["outside_black"] or image.getpixel((width - r - 1, y)) == colormap["outside_white"] for y in range(height)])>thresh:
        r += 1
    # print(Counter([image.getpixel((x, height - b - 2)) for x in range(width)]))
    # print(sum([image.getpixel((x, height - b - 1)) == colormap["outside_black"] or image.getpixel((x, height - b - 1)) == colormap["outside_white"] or image.getpixel((x, height - b - 1)) == (23,23,23,255) for x in range(width)]))
    print(l, r, t, b)

    #crop image by l r t b
    image = image.crop((l, t, width - r, height - b))
    # print(image.size, game_size)
    # print("Check image_size is correct: ", image.size == game_size)
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

def convertPILToCV2(img):
    img_np = np.array(img)
    if img_np.shape[-1] == 4:
        img_np_cv = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
    else:
        img_np_cv = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    return img_np_cv

#try gridding
def split_into_patches(img, grid_size=(9, 10)):
    """Splits an image into grid of (h, w)."""
    img_h, img_w = img.shape
    gh, gw = grid_size
    ph, pw = img_h // gh, img_w // gw  # Patch height and width
    patches = []

    # Ensure the image dimensions are divisible by patch size
    for y in range(0, ph*(gh+1), ph):
        row_patches = []
        rangeh = y+ph if y+ph<img_h-ph else img_h
        for x in range(0, pw*(gw+1), pw):
            rangew = x+pw if x+pw<img_w-pw else img_w
            row_patches.append(img[y:rangeh, x:rangew])
            if rangew == img_w:
                break
        patches.append(row_patches)
        if rangeh == img_h:
            break
    return np.array(patches)  # Shape: (rows, cols, 9, 10)

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

def determine_displacement_and_stitch(array1, array2, directions=[]):
    """
    Determines the best displacement direction using cv2.matchTemplate
    and stitches the images into either a (10,10) or (9,11) grid.
    
    Parameters:
        array1 (numpy array): First image grid of shape (9,10)
        array2 (numpy array): Second image grid of shape (9,10)
        
    Returns:
        stitched_array (numpy array): The new stitched grid
        best_direction (str): The determined displacement direction ('up', 'down', 'left', 'right')
    """
    best_score = -1
    best_direction = None
    stitched_array = None

    if array1.shape != (9, 10):
        print("WARNING: array1 not of shape (9, 10)")
        numrow, numcol = array1.shape
        array1_rowdisp = sum([1 if x == 'up' else (-1 if x == 'down' else 0) for x in directions])
        array1_coldisp = sum([1 if x == 'right' else (-1 if x == 'left' else 0) for x in directions])
        print("check disp", array1_rowdisp, array1_coldisp, array1.shape, array2.shape)
        if array1_rowdisp>0:
            array2 = np.vstack((array2, np.zeros((array1_rowdisp, array2.shape[1]))))
        elif array1_rowdisp <0:
            array2 = np.vstack((np.zeros((-array1_rowdisp, array2.shape[1])), array2))
        tostack = (numrow-array2.shape[0])//2
        array2 = np.vstack((np.zeros((tostack, array2.shape[1])), array2, np.zeros((tostack, array2.shape[1]))))
        if array1_coldisp<0:
            array2 = np.hstack((np.zeros((array2.shape[0],-array1_coldisp)), array2))
        elif array1_coldisp>0:
            array2 = np.hstack((array2, np.zeros((array1_coldisp, array2.shape[0]))))
        tostack = (numcol-array2.shape[1])//2
        array2 = np.hstack((np.zeros((array2.shape[0], tostack)), array2, np.zeros((array2.shape[0], tostack))))            
        print("check array 1 shape = array 2 shape", array1.shape==array2.shape, array1.shape, array2.shape)
    else:
        numrow, numcol = (9, 10)

    # Check Up (array2 is above array1) Up: Compare array1[0:8, :] with array2[1:, :]
    up_scores = [match_patches(array1[i, j], array2[i+1, j]) for i in range(numrow-1) for j in range(numcol)]
    up_score = np.mean(up_scores) if up_scores else -1
    if up_score > best_score:
        best_score = up_score
        best_direction = "up"
        stitched_array = np.vstack((array2[0:1, :], array1))  # Shape (10,10)

    # Check Down (array2 is below array1) Down: Compare array1[1:, :] with array2[0:8, :]
    down_scores = [match_patches(array1[i+1, j], array2[i, j]) for i in range(numrow-1) for j in range(numcol)]
    down_score = np.mean(down_scores) if down_scores else -1
    if down_score > best_score:
        best_score = down_score
        best_direction = "down"
        stitched_array = np.vstack((array1[0:1, :], array2))  # Shape (10,10)

    # Check Left (array2 is left of array1) Left: Compare array1[:, 0:9] with array2[:, 1:]
    # xxyxx
    # xyxxp
    left_scores = [match_patches(array1[i, j], array2[i, j+1]) for i in range(numrow) for j in range(numcol-1)]
    left_score = np.mean(left_scores) if left_scores else -1
    if left_score > best_score:
        best_score = left_score
        best_direction = "left"
        # print(array2[:, 0].shape, array1.shape, array2[:, 0:1].shape)
        stitched_array = np.hstack((array2[:, 0:1], array1))  # Shape (9,11)

    # Check Right (array2 is right of array1) Right: Compare array1[:, 1:] with array2[:, 0:9]
    right_scores = [match_patches(array1[i, j+1], array2[i, j]) for i in range(numrow) for j in range(numcol-1)]
    right_score = np.mean(right_scores) if right_scores else -1
    if right_score > best_score:
        best_score = right_score
        best_direction = "right"
        stitched_array = np.hstack((array1[:, 0:1], array2))  # Shape (9,11)
    print(stitched_array.shape)
    return stitched_array, best_direction

def clean_patches(patches):
    """
    Cleans patches by checking if any patch contains invalid zeros or NaNs.
    Replaces invalid patches with a zero-filled patch of the correct shape.

    Parameters:
        patches (list of lists): A 2D list where each element is a cv2 image patch.

    Returns:
        cleaned_patches (list of lists): A list of cleaned patches.
    """
    cleaned_patches = []

    # Get the valid height (h) and width (w) from the first valid patch in each row and column
    rows, cols = len(patches), len(patches[0])
    
    # Calculate the height (h) and width (w) based on the first valid patch
    valid_patch_h = [0, 0]
    valid_patch_w = [0, 0]
    
    for row in patches[:-1]:     
        for patch in row[:-1]:
            # Find valid patches
            if valid_patch_h[0] == 0 and not(type(patch) == float or type(patch) == int):
                valid_patch_h[0] = patch.shape[0]
            if valid_patch_w[0] == 0 and not(type(patch) == float or type(patch) == int):
                valid_patch_w[0] = patch.shape[1]
            if valid_patch_h[0] != 0 and valid_patch_w[0] != 0:
                break
        if valid_patch_w[1] == 0 and not(type(row[-1]) == float or type(row[-1]) == int):
            valid_patch_w[1] = row[-1].shape[1]
        if valid_patch_w[0] != 0 and valid_patch_w[1] != 0 and valid_patch_h[0] != 0:
            break
    for i in patches[-1]:
        if valid_patch_h[1] == 0 and not(type(i) == float or type(i) == int):
            valid_patch_h[1] = i.shape[0]
            break
    print(valid_patch_h, valid_patch_w)
    # Clean patches
    for i, row in enumerate(patches):
        # if valid_patch_w is None and not(row[0] is None or np.all(row[0] == 0) or type(row[0]) == float or type(row[0]) == int):
        #     valid_patch_w = row[0].shape[1]
        for j, patch in enumerate(row):
            # #Find valid patches
            # if valid_patch_h is None and not(type(patch) == float or type(patch) == int): #patch is None or np.all(patch == 0) or 
            #     valid_patch_h = patch.shape[0]
            #Find invalid patches
            if type(patch) == float or type(patch) == int: #patch is None or np.all(patch == 0) or 
                print("broken patch", patch)
                if i == len(patches)-1:
                    patches[i][j] = np.zeros((valid_patch_h[1], valid_patch_w[0]), dtype=np.float32) if j<len(row)-1 else np.zeros((valid_patch_h[1], valid_patch_w[1]), dtype=np.float32)  # Zero-filled patch
                else:
                    patches[i][j] = np.zeros((valid_patch_h[0], valid_patch_w[0]), dtype=np.float32) if j<len(row)-1 else np.zeros((valid_patch_h[0], valid_patch_w[1]), dtype=np.float32)  # Zero-filled patch

                # find nearest valid patch
                # if valid_patch_h is None:
                    # print("gg clean broken: no valid patch height in row")
                    # else:
                    #     for k in range(j+1, len(row)):
                    #         curr = row[k]
                    #         if not(curr is None or np.all(curr == 0) or type(curr) == float or type(curr) == int):
                    #             valid_patch_h = curr.shape[0]
                    #             break
                # if valid_patch_w is None:
                #     if i == len(patches)-1:
                #         print("gg clean broken: no valid patch width in column")
                #     else:
                #         for l in range(i+1, len(patches)):
                #             curr = patches[l][j]
                #             if not(curr is None or np.all(curr == 0) or type(curr) == float or type(curr) == int):
                #                 valid_patch_w = curr.shape[1]
                #                 break
                #         if valid_patch_w is None:
                #             for l in range(0, i):
                #                 curr = patches[l][j]
                #                 if not(curr is None or np.all(curr == 0) or type(curr) == float or type(curr) == int):
                #                     valid_patch_w = curr.shape[1]
                #                     break
                #         print("no valid patches in this column???", Counter([patch.shape for patch in row if not(type(patch) == float or type(patch == int))]))
                # patches[i][j] = np.zeros((valid_patch_h, valid_patch_w[0]), dtype=np.float32) if j<len(row)-1 else np.zeros((valid_patch_h, valid_patch_w[1]), dtype=np.float32)  # Zero-filled patch
                # valid_patch_h, valid_patch_w = patch.shape[:2]
        # if valid_patch_h is not None and valid_patch_w is not None:
        #     break
    
    if valid_patch_h is None or valid_patch_w is None:
        print("Error: No valid patches found.")
        return []

    # # Clean patches
    # for i in range(rows):
    #     cleaned_row = []
    #     for j in range(cols):
    #         patch = patches[i][j]

    #         # If patch is invalid (None, float 0, or NaN), replace it with a zero-filled patch
    #         if patch is None or np.any(np.isnan(patch)) or type(patch) == int or type(patch) == float:
    #             cleaned_row.append(np.zeros((valid_patch_h, valid_patch_w), dtype=np.float32))  # Zero-filled patch
    #         else:
    #             cleaned_row.append(patch)
    #     cleaned_patches.append(cleaned_row)

    return patches

def stitch_image(patches):
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


def extract_number(filename):
    """Extracts the numeric part X from 'House_X.png' using isdigit()."""
    parts = filename.split('.')  # Split at '.'
    if len(parts) > 1:  # Check if there are two parts
        return int("".join(filter(str.isdigit, parts[0])))  # Extract the number
    return float('inf')  # If no valid number, push to the end

def get_sorted_images(directory):
    """
    Retrieves image filenames from a directory and sorts them based on the numeric value X in 'House_X.png'.
    
    Parameters:
        directory (str): Path to the directory containing the images.
        
    Returns:
        sorted_files (list): List of sorted file paths.
    """
    files = [f for f in os.listdir(directory) if f.startswith("House") and f.endswith(".png")]

    # Sort files based on the extracted number X
    sorted_files = sorted(files, key=extract_number)

    # Return full paths
    return [os.path.join(directory, f) for f in sorted_files]

def resize_image(cv_img, scale_percent = 50):
    # Resize the image to a smaller size (e.g., 50% of the original size)
    width = int(cv_img.shape[1] * scale_percent / 100)
    height = int(cv_img.shape[0] * scale_percent / 100)
    res = cv2.resize(cv_img, (width, height))
    return res

directory = os.path.join(os.getcwd(),'mapstitching_incomplete/images')
sorted_images = get_sorted_images(directory)


# Stitch images
directions = []
initial = load_img(sorted_images[0])
cropped_initial = crop_to_game_area(initial)
initial_img, has_dialogue = contains_dialogue(cropped_initial)
while has_dialogue:
    print(f"{sorted_images[0].split('/')[-1]} has dialogue, deleting.")
    sorted_images.pop(0)
    initial = load_img(sorted_images[0])
    cropped_initial = crop_to_game_area(initial)
    initial_img, has_dialogue = contains_dialogue(cropped_initial)
crop_image_cv = convertPILToCV2(initial_img)
curr_patches = split_into_patches(crop_image_cv, grid_size=(9, 10))
for img_path in sorted_images[1:5]:
    next_img = load_img(img_path)
    cropped_next = crop_to_game_area(next_img)
    next_img_pil, next_has_dialogue = contains_dialogue(cropped_next)
    if next_has_dialogue:
        print(f"skipping {img_path}")
        continue
    else:
        next_crop_image_cv = convertPILToCV2(next_img_pil)
        next_patches = split_into_patches(next_crop_image_cv, grid_size=(9, 10))
        curr_patches, direction = determine_displacement_and_stitch(curr_patches, next_patches, directions)
        print(f"Best direction: {direction}")
        directions.append(direction)
    stitched_image_mid = stitch_image(curr_patches)
    cv2.imshow(f"merged {img_path.split('/')[-1]}", resize_image(stitched_image_mid))

stitched_image = stitch_image(curr_patches)

cv2.imshow("merged", resize_image(stitched_image))
cv2.waitKey(0)
cv2.destroyAllWindows()

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


# # Example Usage:
# # array1 and array2 should be NumPy arrays of shape (9,10) containing cv2 images
# stitched_image_patches, direction = determine_displacement_and_stitch(crop2_image_cv_patches, next_patches)
# print(f"Best direction: {direction}")

# stitched_image = stitch_image(stitched_image_patches)

# cv2.imshow("crop w grids", crop2_image_cv_gridded)

# cv2.imshow("nextimg w grids", next_patches_gridded)
# cv2.imshow("merged", stitched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# matched_coords = match_image(sprite_img, crop2_image)
# save_img(matched_coords, os.path.join(os.getcwd(),'mapstitching_incomplete'), "manual_stitch.png")


# matched_coords = match_sprites(crop2_image, sprite_img)
# cropped_img.show()
