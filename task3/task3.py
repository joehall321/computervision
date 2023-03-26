import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
from os import listdir
from tqdm import tqdm

task2_dataset = "/datasets/task2_images/"
    
def preprocess(img,gaussain_k):
        # Converting image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # preproccess with blurring, with 5x5 kernel
        img = cv2.GaussianBlur(img, (gaussain_k,gaussain_k), 0)
        return img

def meanshift(kps_xy,bandwidth=None):
    # Meanshift 
    # Params need tweaking.  
    ms = MeanShift(bandwidth=bandwidth)
    pred = ms.fit_predict(kps_xy)
    return pred

def kmeans(kps_xy, clusters):
      kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto")
      return kmeans.fit_predict(kps_xy)

def plotClusters(axis, labels, algo, kps_xy):
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    # Cluster keypoints on graph 
    clustered_pts={}
    for idx in range(len(labels)):
        label=labels[idx]
        clustered_pts.setdefault(label, [])
        clustered_pts[label].append(kps_xy[idx])

    for label in clustered_pts:
        axis.scatter([pt[0] for pt in clustered_pts[label]],[pt[1] for pt in clustered_pts[label]])
    
    axis.title.set_text("Estimated "+algo+" clusters: %d" % n_clusters_)

# Takes (labels_kmeans,"K-means"),(labels_meanshift,"Meanshift"), img, kps_xy
def plotTwoClusteringAlgos(cluster1,cluster2,img, kps_xy):
    labels1, algo1 = cluster1
    labels2, algo2 = cluster2
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plotClusters(axes[0],labels1,algo1,kps_xy)
    plotClusters(axes[1],labels2,algo2,kps_xy)
    axes[0].imshow(img)
    axes[1].imshow(img)
    fig.suptitle("Comparision of clustering algorithms.")
    fig.align_labels()
    fig.tight_layout()

def getObjectNumber(img):
     # threshold for grayscale image
    _, threshold_img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB))
    axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.show()

def checkObjectsThreshold():
    path="datasets/task2_images/images/"
    object_images = [f for f in listdir(path)]
    for file in object_images:
        img = cv2.imread(path+file)
        gray = preprocess(img,gaussain_k=51)
        getObjectNumber(gray)
    exit()

def findBandWidthRange(path, SIFT_params=None):
    object_images = [f for f in listdir(path)]
    bandwidths=[]
    # Create SIFT detector
    sift=cv2.SIFT_create()
    if SIFT_params is not None:
        sift.setEdgeThreshold(SIFT_params[0])
        sift.setNOctaveLayers(SIFT_params[1])
    print("SIFT Parameters:")
    print("Default EdgeThreshold",sift.getEdgeThreshold())
    print("Default NFeatures",sift.getNFeatures())
    print("Default NOctaveLayers",sift.getNOctaveLayers())
    print("Default Sigma",sift.getSigma())
    for file in object_images:
        img = cv2.imread(path+file)
        kp = sift.detect(img,None)
        kps_xy = [pt.pt for pt in kp]
        bandwidth=estimate_bandwidth(kps_xy)
        bandwidths.append(bandwidth)
    return int(min(bandwidths)), int(max(bandwidths))

def GridSearchBandwidthMeanshift(path, SIFT_params=None):
    # Get bandwidth ranges 
    bandwidth_range=findBandWidthRange(path, SIFT_params)

    # Object images and annotations
    object_images = [f for f in listdir(path)]
    object_images.sort()
    annotation = path+"../annotations/"
    object_annotations = [f for f in listdir(annotation)]
    object_annotations.sort()
    print("Bandwidth range of images using sklearn.cluster.estimate_bandwidth:",bandwidth_range)

    # Create SIFT detector
    sift=cv2.SIFT_create()
    if SIFT_params is not None:
        sift.setEdgeThreshold(SIFT_params[0])
        sift.setNOctaveLayers(SIFT_params[1])
    
    # Gridsearch 
    print()
    print("FINDING OPTIMAL BANDWIDTH FOR MEANSHIFT")
    accuracies = []
    # constants = [n/100 for n in range(1,101,1)]
    constants = [0.5]
    for constant in tqdm(constants):
        clustering_accuracy = []
        for idx in range(len(object_images)):
            # Read image
            img = cv2.imread(path+object_images[idx])
            # Read annotation
            with open(annotation+object_annotations[idx]) as f:
                lines = f.readlines()
            n_objects=len(lines)

            # SIFT
            kp = sift.detect(img,None)
            kps_xy = [pt.pt for pt in kp]

            # Meanshift 
            bandwidth = estimate_bandwidth(kps_xy)*constant
            labels_meanshift = meanshift(kps_xy,bandwidth=bandwidth)
            n_clusters = len(np.unique(labels_meanshift))
            clustering_accuracy.append(n_objects==n_clusters)

            if n_objects!=n_clusters:
                fig, axes = plt.subplots(nrows=1, ncols=2)
                plotClusters(axes[1],labels_meanshift,"Meanshift",kps_xy)
                # Marking keypoints on the image using circles
                img=cv2.drawKeypoints(img,
                                    kp,
                                    img,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                axes[0].imshow(img)
                axes[1].imshow(img)
                axes[0].title.set_text("Image")
                plt.show()

        clustering_correctness = len([r for r in clustering_accuracy if r is True])/len(clustering_accuracy)
        accuracies.append(clustering_correctness)
    print("DONE")
    print()
    max_accuracy = max(accuracies)
    max_constants = [constants[idx] for idx in [i for i, j in enumerate(accuracies) if j == max_accuracy]]
    print("Optimal constants:",max_constants, "with accuracy:",max_accuracy)
    exit()
        

        

    

print()
path="datasets/task2_images/images/"

GridSearchBandwidthMeanshift(path)

object_images = [f for f in listdir(path)]
for file in object_images:
    img = cv2.imread(path+file)   

    # show image with matplotlib
    # plt.imshow(img)

    # Applying preprocessing 
    gray = preprocess(img, gaussain_k=9)

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
    labels_meanshift = meanshift(kps_xy)
    bandwidth=estimate_bandwidth(kps_xy)
    print("Meanshift bandwidth:",bandwidth)

    # Marking keypoints on the image using circles
    img=cv2.drawKeypoints(gray,
                        kp,
                        img,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    plotClusters(axes[1],labels_meanshift,"Meanshift",kps_xy)
    axes[0].imshow(img)
    axes[1].imshow(img)
    axes[0].title.set_text("SIFT Keypoints")
    print()
    plt.show()