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

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def generate_colors(class_names):

  hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
  colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
  colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
  random.seed(10101)  # Fixed seed for consistent colors across runs.
  random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
  random.seed(None)  # Reset seed to default.
  return colors


def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
  font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
  thickness = (image.size[0] + image.size[1]) // 300

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
    print(label, (left, top), (right, bottom))
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

def process_predicts(predicts,sess):
  box_class_probs = predicts[0, :, :, 0:20] #p_Classes
  box_confidence = predicts[0, :, :, 20:22] #c
    
  boxes = predicts[0, :, :, 22:] #coordinates

  box_class_probs = np.reshape(box_class_probs, (7, 7, 1, 20))
  box_confidence = np.reshape(box_confidence, (7, 7, 2, 1))
  boxes = np.reshape(boxes,(7,7,2,4))

  #box score 7x7x2x20
  
  #print(sess.run(box_confidence))
  #print(sess.run(box_class_probs))
  box_scores = box_confidence * box_class_probs #P
  #print(sess.run(box_scores))
  #print(box_scores.shape)
  
  #print P[5,1, 0, :]

 # DNA changes

  box_classes = K.argmax(box_scores,axis=-1)
  #print(box_classes.shape)
  #print(box_classes[1][1][0],box_classes[1][1][1])

  box_class_scores = K.max(box_scores,axis=-1,keepdims=False) #7x7x2
  #print(sess.run(box_class_scores))
    
  #print(box_class_scores[1][1][0])

  filtering_mask = box_class_scores >= 0.15
  #print(sess.run(filtering_mask))
    
  scores = tf.boolean_mask(box_class_scores,filtering_mask)
  #print(sess.run(scores))
  #print(boxes)
  boxes = tf.boolean_mask(boxes,filtering_mask)
  #print(boxes.shape)
  classes = tf.boolean_mask(box_classes,filtering_mask)
  #print(classes.shape)
  return scores, classes, boxes
 # DNA changes ends

   #old
  #index = np.argmax(box_scores)
  #print(index.shape)
  #print(index)

  #index = np.unravel_index(index, box_scores.shape)

  #class_num = index[3]

  #boxes = np.reshape(boxes, (7, 7, 2, 4))

"""
  max_coordinate = boxes[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  #f.close()

  return xmin, ymin, xmax, ymax, class_num
"""

def draw(boxes,classes):
  '''
  Arguments:
      boxes : [?,4]
      classes : [?]
  '''
  print("3")
  print(boxes)
  print(classes)
  cell_no = 0
  for cell in boxes:
      print("4")
      
      max_coordinate = [cell[0], cell[1], cell[2], cell[3]]

      xcenter = max_coordinate[0]
      ycenter = max_coordinate[1]
      w = max_coordinate[2]
      h = max_coordinate[3]

      xcenter = (cell[1] + xcenter) * (448/7.0)
      ycenter = (cell[0] + ycenter) * (448/7.0)

      w = w * 448
      h = h * 448

      xmin = xcenter - w/2.0
      ymin = ycenter - h/2.0

      xmax = xmin + w
      ymax = ymin + h
      
      class_name = classes_name[classes[cell_no]]
      print("classes_name:" + class_name)
      cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
      cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
      cv2.imwrite('cat_out.jpg', resized_img)
      cell_no = cell_no + 1

common_params = {'image_size': 448, 'num_classes': 20, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)
np_img = cv2.imread('000331.jpg')
resized_img = cv2.resize(np_img, (448, 448))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
np_img = np_img.astype(np.float32)

np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, 448, 448, 3))


#sess = tf.Session()
with tf.Session() as sess:

    saver = tf.train.Saver(net.trainable_collection)
    saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
    np_predict = sess.run(predicts, feed_dict={image: np_img})

    #print(np_predict)
#xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
    print("1")
    scores, classes, boxes  = process_predicts(np_predict,sess)
    print("2")
    print(boxes.eval())
    draw(boxes.eval(),classes.eval())
    
#image, image_data = preprocess_image("cat.jpg", model_image_size = (448, 448))
#colors = generate_colors(classes_name)
#draw_boxes(image, scores, classes, boxes, classes_name, colors)



"""
class_name = classes_name[class_num]
cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
cv2.imwrite('cat_out.jpg', resized_img)
"""
sess.close()
