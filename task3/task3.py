import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt

task2_dataset = "/datasets/task2_images/"
    
def preprocess(img):
        # Converting image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

img = cv2.imread("datasets/task2_images/images/test_image_1.png")

# show image with matplotlib
plt.imshow(img)

# Applying preprocessing 
gray = preprocess(img)

# Applying SIFT detector
sift=cv2.SIFT_create()
print("Default EdgeThreshold",sift.getEdgeThreshold())
print("Default NFeatures",sift.getNFeatures())
print("Default NOctaveLayers",sift.getNOctaveLayers())
print("Default Sigma",sift.getSigma())
kp = sift.detect(img,None)
 
# Marking the keypoint on the image using circles
img=cv2.drawKeypoints(gray,
                      kp,
                      img,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img)
plt.show()