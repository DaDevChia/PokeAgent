## How to run
if you have the screengrab/image from pyboy you can git pull pokeagent_new repo and get
/maps
/mapstitching_partial/processimages.py
/mapstitching_partial/run_imagestitching.py

run
`python.exe ~\Pokeagent_new\mapstitching_partial\run_imagestitching.py "your_new_image_name"`

as an example, if you 
1) move /maps/Map_1.png to another directory
2) copy Map_3 to another directory and open it in an image viewer
3) 
run
`python.exe ~\Pokeagent_new\mapstitching_partial\run_imagestitching.py "mapstitching_partial/images/House_X.png"`
for X in range(6,1,-1) (so basically house 6 down to house 2)
4) Map_3 will get stitched upwards to become Map_1. You can open Map_1, the old Map_3, and the stitched Map_3 at `maps/Map_3.png` for comparions.