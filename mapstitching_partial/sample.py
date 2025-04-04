from processimages import (load_img, save_img, convertPILToCV2, 
                           resize_image, contains_dialogue, crop_to_game_area, 
                           stitch_images, determine_displacement, 
                           get_sorted_images)
import os
import cv2

directory = os.path.join(os.getcwd(),'mapstitching_partial/images')
sorted_images = get_sorted_images(directory)

# Stitch images
startind= 0
initial = load_img(sorted_images[startind])
cropped_initial = crop_to_game_area(initial)
initial_img, has_dialogue = contains_dialogue(cropped_initial)
while has_dialogue:
    print(f"{sorted_images[0].split('/')[-1]} has dialogue, deleting.")
    sorted_images.pop(startind)
    initial = load_img(sorted_images[startind])
    cropped_initial = crop_to_game_area(initial)
    initial_img, has_dialogue = contains_dialogue(cropped_initial)
crop_image_cv = convertPILToCV2(initial_img)
crop_color = convertPILToCV2(initial_img, color=True)

for img_path in sorted_images[startind+1:]:
    next_img = load_img(img_path)
    cropped_next = crop_to_game_area(next_img)
    next_img_pil, next_has_dialogue = contains_dialogue(cropped_next)
    if next_has_dialogue:
        print(f"skipping {img_path}")
        continue
    else:
        next_crop_image_cv = convertPILToCV2(next_img_pil)
        next_color = convertPILToCV2(next_img_pil, color=True)
        canvas, array1_coord, array2_coord = determine_displacement(crop_image_cv, next_crop_image_cv)
        print(f"canvas size: {canvas}, array1_coord: {array1_coord}, array2_coord: {array2_coord}")
    if canvas is None:
        print(f"New map for {img_path.split('/')[-1]} due to no match.")
        # Save the stitched image to the /maps directory
        save_img(crop_color, os.path.join(os.getcwd(),'mapstitching_incomplete/maps'), f"Map_1.png")

        crop_image_cv, crop_color = next_crop_image_cv, next_color
    else:
        stitched_image_mid = stitch_images(canvas, array1_coord, array2_coord, crop_color, next_color)
        # cv2.imshow(f"merged {img_path.split('/')[-1]}", resize_image(stitched_image_mid))
        crop_color = stitched_image_mid