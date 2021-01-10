''' Here we are going to find the bounding boxes around the
    objects and then use K-means clustering to predict the 
    prominent colours inside the bounding box '''
    
# Import the required libraries

import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans

# Define the list of colours that we are going to predict

colorsList = {'Red': [255, 0, 0],
              'Green': [0, 128, 0],
              'Blue': [0, 0, 255],
              'Yellow': [255, 255, 0],
              'Violet': [238, 130, 238],
              'Orange': [255, 165, 0],
              'Black': [0, 0, 0],
              'White': [255, 255, 255],
              'Pink': [255, 192, 203],
              'Brown': [165, 42, 42]}

# This class returns the R, G, B values of the dominant colours

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
    
        #read image
        
        #convert to rgb from bgr
        img = self.IMAGE
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)

# This class predicts the bounding box in an image

class BoundBox():
    
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        
    def get_label(self):
        
        if self.label == -1:
            self.label = np.argmax(self.classes)
            
        return self.label
    
    def get_score(self):
        
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score
  
# Sigmoid function
        
def sigmoid(x):
    
    return 1. / (1. + np.exp(-x))

# Docoding the net output of the model

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    boxes = []
    
    netout[..., :2] = sigmoid(netout[..., :2])
    netout[..., 4:] = sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h * grid_w):
        
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            
            objectness = netout[int(row)][int(col)][b][4]
            
            if objectness.all() <= obj_thresh:
                continue
            
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w
            y = (row + y) / grid_h
            w = anchors[2 * b + 0] * np.exp(w) / net_w
            h = anchors[2 * b + 1] * np.exp(h) / net_h
            classes = netout[int(row)][int(col)][b][5:]
            box = BoundBox(x - w/2, y - h/2, x + w/2, y + h/2, objectness, classes)
            boxes.append(box)
            
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    
    new_w, new_h = net_w, net_h
    
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
        
def interval_overlap(interval_a, interval_b):
    
    x1, x2 = interval_a
    x3, x4 = interval_b
    
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3
        
def bbox_iou(box1, box2):
    
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_h * intersect_w
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    
    for c in range(nb_class):
        
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        
        for i in range(len(sorted_indices)):
            
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0:
                continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    
def load_image_pixels(filename, shape):
    
    image = load_img(filename)
    width, height = image.size
    image = load_img(filename, target_size = shape)
    image = img_to_array(image)
    image = image.astype('float32')
    image = image / 255.0
    image = expand_dims(image, 0)
    return image, width, height

def get_boxes(boxes, labels, thresh):
    
    v_boxes, v_labels, v_scores = list(), list(), list()
    
    for box in boxes:
        
        for i in range(len(labels)):
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
                
    return v_boxes, v_labels, v_scores

def draw_boxes(filename, v_boxes, v_labels, v_scores):
    
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    
    for i in range(len(v_boxes)):
        
        color_detected = set()
        color_det = []
        box = v_boxes[i]
        y1, x1, y2, x2, = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'green')
        ax.add_patch(rect)
        label = "%s (%.3f)" % (v_labels[i], v_scores[i]) + " "
        colors = DominantColors(data[y1:y1+width, x1:x1+height], 3).dominantColors()
        
        for rgb in colors:
            mindist = 500
            name = str()
            for color in colorsList:
                dist = np.linalg.norm(rgb - list(colorsList[color]))
                if dist < mindist :
                    name = color
                    mindist = dist
                    
            color_det = color_det + [name]
            
        color_detected = set(color_det)
        for color in color_detected:
            label = label + color + " "
        
        pyplot.text(x1, y1, label, color = 'green')
    
    pyplot.show()
    pyplot.clf()
 
def merge_functions(photo_filename):

    model = load_model('model.h5')
    input_w, input_h = 416, 416
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
    yhat = model.predict(image)

    print([a.shape for a in yhat])
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

    class_threshold = 0.6

    boxes = list()

    for i in range(len(yhat)):
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    do_nms(boxes, 0.5)

    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
    
merge_functions('apple.jpg')