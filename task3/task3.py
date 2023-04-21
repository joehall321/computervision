import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
from os import listdir
from tqdm import tqdm
import itertools
import re

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

def tryDifferentGaussians(paths):
    for path in paths:
        print("Dataset:"+path)
        print()
        # Object images
        object_images = [f for f in listdir(path)]
        for k in range(3,16,2):
            for idx in range(len(object_images)):
                # Read image
                img = cv2.imread(path+object_images[idx])
                blurred = preprocess(img,k)
                fig, axes = plt.subplots(nrows=1, ncols=2)
                # Marking keypoints on the image using circles
                axes[0].imshow(img)
                axes[1].imshow(cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB))
                axes[0].title.set_text("Original Image")
                axes[1].title.set_text("Gaussian Blur k="+str(k))
                plt.show()


def GridSearchBandwidthMeanshift(paths, plot_errors=False):
    object_images=[]
    object_annotations=[]
    for path in paths:
        print("Dataset:"+path)
        # Object images and annotations
        ims = [path+f for f in listdir(path)]
        ims.sort()
        annotation = path+"../annotations/"
        ans = [annotation+f for f in listdir(annotation)]
        ans.sort()

        object_images = object_images + ims
        object_annotations = object_annotations + ans

    # Create SIFT detector
    sift=cv2.SIFT_create()
    
    # Reading images and annotations
    images=[]
    n_objects=[]
    for idx in range(len(object_images)):
        # Read image
        images.append(preprocess(cv2.imread(object_images[idx]),9))

        # Read annotation
        with open(object_annotations[idx]) as f:
            lines = f.readlines()
        n_objects.append(len(lines))

    # Gridsearch parameters
    accuracies = []
    #constants = [n/100 for n in range(51,64)]
    constants = [n/100 for n in range(50,61)]
    edgesThresholds = [n for n in range(2,16,2)]
    octaves = [n for n in range(2,7)]
    sigmas = [n/10 for n in range(5,21)]
    #parameters = list(itertools.product(constants,octaves,sigmas,edgesThresholds))
    parameters = [(0.51, 5, 1.2, 4)] 

    print()
    print("FINDING OPTIMAL BANDWIDTH FOR MEANSHIFT OBJECT CLUSTERING")
    counter = 0
    for (constant,octave,sigma,edgesThreshold) in tqdm(parameters):
        clustering_accuracy = []

        # Set SIFT params
        sift.setNOctaveLayers(octave)
        sift.setSigma(sigma)
        sift.setEdgeThreshold(edgesThreshold)
        
        for idx in range(len(images)):
            img = images[idx]
            n_img_objects = n_objects[idx]

            # SIFT
            kp = sift.detect(img,None)
            kps_xy = [pt.pt for pt in kp]

            # Meanshift 
            bandwidth = estimate_bandwidth(kps_xy)*constant
            labels_meanshift = meanshift(kps_xy,bandwidth=bandwidth)
            n_clusters = len(np.unique(labels_meanshift))
            clustering_accuracy.append(n_img_objects==n_clusters)

            if n_img_objects!=n_clusters and plot_errors:
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
        # counter+=1
        # print("Counter:",counter)
        # if counter == 3:
        #     break

    print("DONE")
    print()
    max_accuracy = max(accuracies)
    max_parameters = [parameters[idx] for idx in [i for i, j in enumerate(accuracies) if j == max_accuracy]]
    print("Optimal constants:",max_parameters, "with accuracy:",max_accuracy)
    exit()

def getObjectDatabase(path, gaussian_k, SIFT_params):
    objects = [f for f in listdir(path)]
    SIFT_n_octaves, SIFT_sigma, SIFT_edge_threshold = SIFT_params

    sift=cv2.SIFT_create()
    sift.setEdgeThreshold(SIFT_edge_threshold)
    sift.setNOctaveLayers(SIFT_n_octaves)
    sift.setSigma(SIFT_sigma)

    object_database={}
    for object in objects:

        name = object[4:len(object)-4]
        img = cv2.imread(path+object)
        blurred = preprocess(img,gaussian_k)
        kp, desc = sift.detectAndCompute(blurred,None)
        img = cv2.drawKeypoints(blurred,
                            kp,
                            img,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        object_database[name]=(img,kp,desc)
        
    return object_database

def inlier_bounds(data):
    ''' Detection '''
    # IQR
    Q1 = np.percentile(data, 25,
                    method = 'midpoint')
    Q3 = np.percentile(data, 75,
                    method = 'midpoint')
    IQR = Q3 - Q1

    # Upper bound
    upper=Q3+1.5*IQR
    # Lower bound
    lower=Q1-1.5*IQR

    return lower, upper

def check_inbounds(pt, x_bounds, y_bounds):
    return x_bounds[0] <= pt[0] <= x_bounds[1] and y_bounds[0] <= pt[1] <= y_bounds[1]

def remove_outliers(kps_desc):
    kps, desc = kps_desc[0], kps_desc[1]
    x_bounds = inlier_bounds([pt.pt[0] for pt in kps])
    y_bounds = inlier_bounds([pt.pt[1] for pt in kps])

    remove_idxs = []
    for idx in range(len(kps)):
        pt = kps[idx]
        if not check_inbounds(pt.pt, x_bounds, y_bounds):
            remove_idxs.append(idx)
    remove_idxs.sort(reverse=True)

    for idx in remove_idxs:
        kps.pop(idx)
        desc.pop(idx)

    return kps, desc



print()
image_paths="datasets/task2_images/images/","datasets/task3_images/images/"
objects_path="datasets/objects/"

#GridSearchBandwidthMeanshift(paths, plot_errors=True)

# Parameters
gaussain_k=9
ms_bandwidth_constant = 0.51

SIFT_n_octaves = 3
SIFT_sigma = 1.6
SIFT_edge_threshold = 10

#Creating SIFT detector
sift=cv2.SIFT_create()
# sift.setEdgeThreshold(SIFT_edge_threshold)
# sift.setNOctaveLayers(SIFT_n_octaves)
# sift.setSigma(SIFT_sigma)

print("Gaussian Blur k:",gaussain_k)
print("SIFT EdgeThreshold",sift.getEdgeThreshold())
print("SIFT NOctaveLayers",sift.getNOctaveLayers())
print("SIFT Sigma",sift.getSigma())
print("Meanshift bandwidth constant:",ms_bandwidth_constant)

# object_database[name]=(img,kp,desc)
object_database = getObjectDatabase(objects_path, gaussain_k, (SIFT_n_octaves,SIFT_sigma,SIFT_edge_threshold))

for path in image_paths:

    object_images = [f for f in listdir(path)]
    annotation = path+"../annotations/"
    annotations = [f for f in listdir(annotation)]

    for idx in range(len(object_images)):

        img = cv2.imread(path+object_images[idx])   

        # Read annotation
        with open(annotation+annotations[idx]) as f:
            lines = f.readlines()
        n_objects = len(lines)

        # show image with matplotlib
        # plt.imshow(img)

        # Applying preprocessing 
        blurred = preprocess(img, gaussain_k)

        # Applying SIFT detector
        kp, desc = sift.detectAndCompute(blurred,None)
        kps_xy = [pt.pt for pt in kp]

        # Meanshift 
        ms_bandwidth=estimate_bandwidth(kps_xy)*ms_bandwidth_constant
        labels_meanshift = meanshift(kps_xy,bandwidth=ms_bandwidth)
        
        visual_words = {}
        for idx in range(len(labels_meanshift)):
            label = labels_meanshift[idx]
            visual_words.setdefault(label,([],[]))
            visual_words.get(label)[0].append(kp[idx])
            visual_words.get(label)[1].append(desc[idx])
        
        # Marking keypoints on the image using circles
        img=cv2.drawKeypoints(blurred,
                            kp,
                            img,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(img)
        # plt.show()
        
        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # plotClusters(axes[1],labels_meanshift,"Meanshift",kps_xy)
        # axes[0].imshow(img)
        # axes[1].imshow(img)
        # axes[0].title.set_text("SIFT Keypoints")
        # print()
        # plt.show()

        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)

        # object_database[name]=(img,kp,desc)
        for word in visual_words:

            kps, descriptors = remove_outliers(visual_words[word])

            names=[]
            values=[]
            object_matches = []

            # Loops through object database, computes the best matches for each object descriptor
            for object in object_database:
                matches = bf.match(object_database[object][2],np.array(descriptors))
                object_matches.append(matches)
                distances = [d.distance for d in matches]
                names.append(object)
                values.append(sum(distances)/len(distances))

                # print(object+" descriptors:",len(object_database[object][2]))
                # print("visual word descriptors:",len(np.array(descriptors)))
                # print("Matches:",len(matches))
                # print("Distances:",distances)
                # print()
                
            class_label = names[values.index(min(values))]
            
            xs = [p.pt[0] for p in kps]
            ys = [p.pt[1] for p in kps]
            kp_avg_x = sum(xs)/len(xs)
            kp_avg_y = sum(ys)/len(ys)
            dist_x = max(xs)-min(xs)
            dist_y = max(ys)-min(ys)

            x1, y1 = kp_avg_x-(dist_x/2), kp_avg_y+(dist_y/2)

            factor = 1
            box_xs = [x1,x1+(factor*dist_x),x1+(factor*dist_x),x1,x1]
            box_ys = [y1,y1,y1-(factor*dist_y),y1-(factor*dist_y),y1]

            # Check class label with annotations
            color='r'
            for line in lines:
                coords = [int(s) for s in re.findall(r'\b\d+\b', line)]
                if check_inbounds((kp_avg_x,kp_avg_y),(coords[0],coords[2]),(coords[1],coords[3])):
                    if line.split(',')[0] == class_label:
                        color = 'g'

            plt.imshow(img)
            plt.plot(box_xs,box_ys,color=color)
            plt.text(box_xs[0],box_ys[0],class_label,color=color)

            # print("Object classification:",class_label)
            # print()
            # img3 = cv2.drawMatches(object_database[class_label][0], object_database[class_label][1], 
            #                        img, kps, object_matches[values.index(min(values))][:50], img, flags=2)
            # plt.imshow(img3),plt.show()
        plt.show()