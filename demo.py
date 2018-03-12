import sys

sys.path.append('./')
sys.path.append('./yolo')
sys.path.append('./yolo/dataset')
sys.path.append('./yolo/net')
sys.path.append('./yolo/solver')
sys.path.append('./yolo/utils')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cv2
import numpy as np
import random
from keras import backend as K
import math
import colorsys
import imghdr
import os
import random
import scipy.misc
from PIL import Image, ImageDraw, ImageFont

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


# videos to frames

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    import time
    import os
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break


#frames to video
def frames_to_video():
    
    # Arguments
    dir_path = 'outputframes'
    ext = 'jpg'
    output = os.path.join('outvideo','outputvideo.mp4')
    

    images = []
    for f in os.listdir(dir_path):
        #print (f)
        if f.endswith(ext):
            images.append(f)
        
        # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))
            
            
def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    #Loading the font File
    font = ImageFont.truetype(font='FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
   
    #print("Drawing boxes...")
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
     

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Interpret network output (7x7x30) to bb coordinates (long process)
def InterpretPredictions(predicts,sess):
  p_classes = predicts[0, :, :, 0:20] # conditional class probabilitites -> 7x7x20
  C = predicts[0, :, :, 20:22]        # individual box confidence predictions -> 7x7x2 
  coordinate = predicts[0, :, :, 22:] # class coordinates -> 7x7x8
  coordinate = np.reshape(coordinate, (7, 7, 2, 4)) # BoundingBox1 and BoundingBox2 coordinates
  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))
 
  threshold = 0.07
  # multiply class probabilities with confidence matrix
  P = C * p_classes   #7x7x2x20
  P = sigmoid(P)  # normalize them into probabilities

  box_scores = C * p_classes    #7x7x2x20
  box_scores = tf.convert_to_tensor(box_scores)

  box_classes = K.argmax(box_scores,-1)    #7x7x2
  box_class_scores = K.max(box_scores,-1)  #7x7x2
  
  #print(sess.run(box_classes), sess.run(box_class_scores))
  filtering_mask = box_class_scores > threshold
  
  #print(sess.run(filtering_mask))
    
  index = tf.where(box_class_scores > threshold) #  Find the most cells witht Probabilities > Threshold

  #print(sess.run(index))

  scores = tf.boolean_mask(box_class_scores,filtering_mask)
  boxes = tf.boolean_mask(coordinate,filtering_mask)
  classes = tf.boolean_mask(box_classes,filtering_mask)
  
  #print(sess.run(scores),sess.run(boxes),sess.run(classes))
  # Scale boxes back to original image shape
  #boxes = scale_boxes(boxes, (300.,300.))

  max_boxes = 10
  max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
  #print(sess.run(boxes))
  sess.run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
  
  #iou_threshold = 0.5
  #nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes,0.5)
  #scores = K.gather(scores,nms_indices)
  #boxes = K.gather(boxes,nms_indices)
  #classes = K.gather(classes,nms_indices)

  return scores, boxes, classes, index

# import the necessary packages
import numpy as np
 
# Malisiewicz et al.
def non_max_suppression(boxes, object_num, overlapThresh):
    # if there are no boxes, return an empty list
    boxes = np.array(boxes,dtype=np.float32)
    boxes = boxes[:object_num,:]
   
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    #print(boxes)
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    #print("AREA:",area) 
    #print("IDXS:",idxs)
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        #print("HELLO",i)
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        #print("x1[i]:",x1[i])
        #print("x1[idxs[:last]]:",x1[idxs[:last]])
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        #print(idxs[:last])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")



def scale(out_scores, out_boxes, out_classes, index):
    labels = [[0, 0, 0, 0, 0]] * 20
    object_num = 0
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = classes_name[c]
        max_coordinate = out_boxes[i]
        score = out_scores[i]
        
        #Extract the box coordinates
        xcenter = max_coordinate[0]
        ycenter = max_coordinate[1]
        w = max_coordinate[2]
        h = max_coordinate[3]

        xcenter = (index[i][1] + xcenter) * (448/7.0)
        ycenter = (index[i][0] + ycenter) * (448/7.0)
        w = w * 448
        h = h * 448
        
        # Convert them to BB coordinates (same as annotations)
        xmin = xcenter - w/2.0
        ymin = ycenter - h/2.0
        xmax = xmin + w
        ymax = ymin + h
        
        #print(xmin,ymin,xmax,ymax,c)
        
        labels[object_num] = [xmin, ymin, xmax, ymax, c]
        object_num += 1
        if object_num >= 20:
              break
    return labels, object_num
        
common_params = {'image_size': 448, 'num_classes': 20, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))

predicts = net.inference(image)

sess = tf.Session()

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

#input_loc = 'iframes/input.mp4'
output_loc = 'cars/output_frames'
#video_to_frames(input_loc, output_loc)

dir_path = 'cars/input_frames'
ext = 'jpg'
    #output = os.path.join('outvideo','outputvideo.mp4')
    

images = []
for f in os.listdir(dir_path):
 #print (f)
    if f.endswith(ext):
        images.append(f)

for img in images:
    np_img = cv2.imread(os.path.join(dir_path + '/' + img))
    resized_img = cv2.resize(np_img, (448, 448))
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)


    np_img = np_img.astype(np.float32)

    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, 448, 448, 3))

    np_predict = sess.run(predicts, feed_dict={image: np_img})

    #xmin, ymin, xmax, ymax, class_num= process_predicts(np_predict)
    out_scores, out_boxes, out_classes, index = InterpretPredictions(np_predict,sess)

    # Print predictions info
    out_scores, out_boxes, out_classes, index = sess.run([out_scores, out_boxes, out_classes, index])

    iou_threshold = 0.6

    labels,object_num = scale(out_scores,out_boxes,out_classes,index)

    labels = non_max_suppression(labels, object_num, iou_threshold)

    #np_img = cv2.imread('cat.jpg')
    #resized_img = cv2.resize(np_img, (448, 448))
    #np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    for obj_num,object in enumerate(labels):
        class_name = classes_name[object[4]]
        #print("Printing Classes: ",str(class_name))
        cv2.rectangle(resized_img, (int(object[0]), int(object[1])), (int(object[2]), int(object[3])), (0, 0, 255))
        cv2.putText(resized_img, class_name, (int(object[0]), int(object[1])), 2, 1.5, (0, 0, 255))
        
    cv2.imwrite(output_loc + "/" + img , resized_img)
    #cv2.imwrite(img + 'out.jpg', resized_img)

#frames_to_video()
sess.close()

