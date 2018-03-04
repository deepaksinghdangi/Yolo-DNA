import sys

sys.path.append('./')
sys.path.append('./')
sys.path.append('./yolo')
sys.path.append('./yolo/dataset')
sys.path.append('./yolo/net')
sys.path.append('./yolo/solver')
sys.path.append('./yolo/utils')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
import cv2
import numpy as np
import colorsys
import imghdr
from PIL import Image, ImageDraw, ImageFont
import random
from keras import backend as K
import math

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Interpret network output (7x7x30) to bb coordinates (long process)
def InterpretPredictions(predicts):
  p_classes = predicts[0, :, :, 0:20] # conditional class probabilitites -> 7x7x20
  C = predicts[0, :, :, 20:22]        # individual box confidence predictions -> 7x7x2 
  coordinate = predicts[0, :, :, 22:] # class coordinates -> 7x7x8
  coordinate = np.reshape(coordinate, (7, 7, 2, 4)) # BoundingBox1 and BoundingBox2 coordinates
  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))
 
  # multiply class probabilities with confidence matrix
  P = C * p_classes

  P = sigmoid(P)  # normalize them into probabilities

  scores = P[P>0.52]
  print("scores shape: "+str(scores.shape))
  find_probs = np.where(P>0.52) #  Find the most cells witht Probabilities > Threshold
  #print(find_probs)
  find_probs_np = []

  # Initialize labels to return
  # 20: max_objects_per_image
  labels = [[0, 0, 0, 0, 0]] * 20
  object_num = 0

  # Read the elements
  for i in range(len(find_probs)):  
    find_probs_np.append(find_probs[i])

  for j in range(0, len(find_probs_np[0])):

    myindex=[] # myindex is for all the objects in the image

    for i in range(0, len(find_probs_np)):         
      myindex.append(find_probs_np[i][j])

    index = np.argmax(P) # Get the elements for the class with the highest probabilities (this returns only one object)
    index = np.unravel_index(index, P.shape)

    # Get the class number
    class_num = myindex[3]

    # Get the coordinates 
    max_coordinate = coordinate[myindex[0], myindex[1], myindex[2], :]

    # Get x,y center of the BB
    xcenter = max_coordinate[0]
    ycenter = max_coordinate[1]
    w = max_coordinate[2]
    h = max_coordinate[3]
    xcenter = (myindex[1] + xcenter) * (448/7.0)
    ycenter = (myindex[0] + ycenter) * (448/7.0)

    w = w * 448
    h = h * 448

    # Convert them to BB coordinates (same as annotations)
    xmin = xcenter - w/2.0
    ymin = ycenter - h/2.0
    xmax = xmin + w
    ymax = ymin + h

    # Append to the final labels
    labels[object_num] = [xmin, ymin, xmax, ymax, class_num]
    object_num += 1
    if object_num >= 20:
          break
  return labels, object_num, scores

#  Felzenszwalb et al.
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]

common_params = {'image_size': 448, 'num_classes': 20, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

np_img = cv2.imread('cat.jpg')
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


np_img = np_img.astype(np.float32)

np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

np_predict = sess.run(predicts, feed_dict={image: np_img})

#xmin, ymin, xmax, ymax, class_num= process_predicts(np_predict)
labels, object_num, scores = InterpretPredictions(np_predict)

print("BEFORE")
for obj_num,object in enumerate(labels):
    if obj_num == object_num:
        break
    print(object)
    class_name = classes_name[int(object[4])]
    print("BEFORE: Printing Classes: ",str(class_name))

objects = non_max_suppression(np.asarray(labels,dtype=np.float32),0.8)
print(objects)

for obj_num,object in enumerate(objects):
    if obj_num == object_num:
        break
    class_name = classes_name[int(object[4])]
    print("Printing Classes: ",str(class_name))
    cv2.rectangle(resized_img, (int(object[0]), int(object[1])), (int(object[2]), int(object[3])), (0, 0, 255))
    cv2.putText(resized_img, class_name, (int(object[0]), int(object[1])), 2, 1.5, (0, 0, 255))
'''
class_name2 = classes_name[class_num2]
cv2.rectangle(resized_img, (int(xmin2), int(ymin2)), (int(xmax2), int(ymax2)), (0, 0, 255))
cv2.putText(resized_img, class_name2, (int(xmin2), int(ymin2)), 2, 1.5, (0, 0, 255))
'''
cv2.imwrite('cat_out.jpg', resized_img)

sess.close()
