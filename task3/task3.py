import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift

task2_dataset = "/datasets/task2_images/"
    
def preprocess(img):
        # Converting image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # preproccess with blurring, with 5x5 kernel
        img = cv2.GaussianBlur(img, (5,5), 0)
        return img

img = cv2.imread("datasets/task2_images/images/test_image_1.png")

# show image with matplotlib
plt.imshow(img)

# Applying preprocessing 
gray = preprocess(img)

# Applying SIFT detector
sift=cv2.SIFT_create()
sift.setEdgeThreshold(10)
sift.setNOctaveLayers(6)
print("Default EdgeThreshold",sift.getEdgeThreshold())
print("Default NFeatures",sift.getNFeatures())
print("Default NOctaveLayers",sift.getNOctaveLayers())
print("Default Sigma",sift.getSigma())
kp = sift.detect(img,None)
kps_xy = [pt.pt for pt in kp]

# Meanshift 
# Params need tweaking.  
ms = MeanShift()
labels = ms.fit_predict(kps_xy)
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

plt.figure(1)

# Cluster keypoints on graph 
clustered_pts={}
for idx in range(len(labels)):
    label=labels[idx]
    clustered_pts.setdefault(label, [])
    clustered_pts[label].append(kps_xy[idx])

for label in clustered_pts:
      plt.scatter([pt[0] for pt in clustered_pts[label]],[pt[1] for pt in clustered_pts[label]])

plt.title("Estimated number of clusters: %d" % n_clusters_)

# Marking the keypoint on the image using circles
img=cv2.drawKeypoints(gray,
                      kp,
                      img,
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img)
plt.show()