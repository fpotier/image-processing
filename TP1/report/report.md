# Report
Potier - Boireau
## Loading original image and display it in background
- Load the image
- Create another image with white color and not full alpha
![veiled_orig_image](veiled_orig_image.jpg)
## Loading fragments
- 
## Adding fragments to image

## Channels of original image and fragments' do not match
- adding alpha channel to original image

![fragments_with_borders](fragments_with_borders.png)

## Not copying transparent borders of fragments
- building a mask to filter the pixel with low alpha component and passing it to copyTo()

![reconstruction](reconstruction.jpg)