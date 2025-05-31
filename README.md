# Coin_Detection
## Aim:
To develop a Python program using image processing techniques to detect and count coins in an image by applying filtering, edge detection, and contour detection methods.

## Procedure:
1. Import Libraries
Import necessary Python libraries:
2. Read the Input Image
Load the image containing coins:
3. Preprocess the Image
Apply Gaussian Blur to reduce noise:
4. Apply Edge Detection
Use Canny Edge Detection to find edges:
5. Find Contours
Detect the outlines of coins:
6. Draw Contours and Count Coins
Draw contours and count the number of detected coins:
7. Display the Output
Show the final image with detected coins:

## Program:
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

image=cv2.imread('CoinsA.png')

imageCopy = image.copy()
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.show()

imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12,12))
plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image")
plt.subplot(122); plt.imshow(imageGray,cmap='gray');plt.title("Grayscale Image");
plt.show()

imageB,imageG,imageR = cv2.split(image)

plt.figure(figsize=(20,12))
plt.subplot(141);plt.imshow(image[:,:,::-1]);plt.title("Original Image")
plt.subplot(142);plt.imshow(imageB,cmap='gray');plt.title("Blue Channel")
plt.subplot(143);plt.imshow(imageG,cmap='gray');plt.title("Green Channel")
plt.subplot(144);plt.imshow(imageR,cmap='gray');plt.title("Red Channel");
plt.show()

thresh = 20
maxValue = 255

th,dst_bin=cv2.threshold(imageG,thresh,maxValue,cv2.THRESH_BINARY_INV)

plt.imshow(dst_bin,cmap='gray');
plt.title("Threshold Binary Inverse");
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
imageDilated2 = cv2.dilate(dst_bin, kernel, iterations=2)
dilated_image_rgb = cv2.cvtColor(imageDilated2, cv2.COLOR_BGR2RGB)
plt.imshow(dilated_image_rgb,cmap='gray');plt.title('Dilated Image Iteration 2');plt.show()

kSize=(5,5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kSize)
eroded_image = cv2.erode(dilated_image_rgb, kernel, iterations=2)
imageEroded = cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB)

plt.imshow(imageEroded,cmap='gray');plt.title("Eroded Image");plt.show()

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(imageEroded)
print(f"Number of coins detected: {len(keypoints)}")
```
## Output
![70281571-5ab9-4895-be71-2a7c775c4da3](https://github.com/user-attachments/assets/1cf915a0-7cd9-429e-a014-0bc945266942)
![4bd954da-a311-46f1-a21e-1887d0f3fa46](https://github.com/user-attachments/assets/99ee71f7-50f7-47ab-8288-7395fd87c950)
![c7d9de1e-e3ac-4a4e-8447-535516647705](https://github.com/user-attachments/assets/af0e56f2-f5ae-4a87-8c45-06e0a7cc07ea)


![281b1e78-7da9-43a6-92a5-d45ff9f78e35](https://github.com/user-attachments/assets/ee9041f1-fa2a-4220-83d7-ae460a135af6)

![5901fae9-ffac-45a5-995c-26d3657c7f32](https://github.com/user-attachments/assets/edc7a526-fc2f-45e1-9632-429892715e39)


![f5f36920-7e72-4029-8533-16516c5cae87](https://github.com/user-attachments/assets/cf3da750-02dd-4e01-abee-720b3db9babd)

Number of coins detected: 9





























