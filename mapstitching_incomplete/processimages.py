from PIL import Image
import os
from collections import Counter
import cv2
import numpy as np

# load overall map from path
def load_img(path):
	if not os.path.exists(path):
		raise FileNotFoundError(f"The file at {path} does not exist.")
	return Image.open(path)

def save_img(img, imgdir):
    img.save(os.path.join(imgdir, "output.png"))

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


def match_image(patch, big_map):
    # Convert PIL image to BGR (OpenCV uses BGR format)
    if big_map.mode == "RGBA":
        big_map = big_map.convert("RGB")
    if patch.mode == "RGBA":
        patch = patch.convert("RGB")
    big_map_cv = cv2.cvtColor(np.array(big_map), cv2.COLOR_RGB2BGR)
    patch_cv = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)

    # Convert to grayscale (optional but improves matching)
    big_map_gray = cv2.cvtColor(big_map_cv, cv2.COLOR_BGR2GRAY)
    # # Crop the region of interest (ROI)
    big_map_gray = big_map_gray[5199:5393, 1360:1520]

    patch_gray = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2GRAY)

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
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

print(os.getcwd())
map_path = os.path.join(os.getcwd(),'mapstitching_incomplete/kanto_big_done1_uncompressed.png')
map_img = load_img(map_path)
# counter_map = Counter(map_img.getdata())
# print(counter_map)
# print((248, 248, 248, 255) in counter_map) # False: can use dialogue checker
# print((0, 0, 0, 255) in counter_map) # True: careful of matching blacks
# print((243, 243, 243, 255) in counter_map) # False: Pyboy interface (Grey) is ok to crop out

sprite_path = os.path.join(os.getcwd(),'mapstitching_incomplete/characters_transparent.png')
sprite_img = load_img(sprite_path)

test_path = os.path.join(os.getcwd(),'mapstitching_incomplete/images/House3.png')
test_img = load_img(test_path)
cropped_img = crop_to_game_area(test_img)
crop2_image, has_dialogue = contains_dialogue(cropped_img)
print(crop2_image.size)
matched_coords = match_image(crop2_image, map_img)
# matched_coords = match_sprites(crop2_image, sprite_img)
# cropped_img.show()