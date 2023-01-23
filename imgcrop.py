import cv2 
import os

# Define the path to the folder containing the images
folder_path = 'screenshots'

# Define the size to which you want to resize the images
size = (1600, 769)

# Define the regions of interest
regions = [(220, 181, 100, 100), (220, 281, 100, 100), (220, 381, 100, 100),
           (220, 481, 100, 100), (220, 581, 100, 100),
           (1294, 181, 100, 100), (1294, 281, 100, 100), (1294, 381, 100, 100),
           (1294, 481, 100, 100), (1294, 581, 100, 100),(679, 16, 100, 100)]

# Loop through all the files in the folder
for file in os.listdir(folder_path):
  # Load the image
  img = cv2.imread(os.path.join(folder_path, file))
   # Resize the image
  img_resized = cv2.resize(img, size)

  # Crop the image and save it
  for x, y, w, h in regions:
    # Extract the region of interest from the image
    roi = img_resized[y:y+h, x:x+w]

    # Save the cropped image
    cv2.imwrite(f'cropped_{file}_{x}_{y}_{w}_{h}.jpg', roi)
