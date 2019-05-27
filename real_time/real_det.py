import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils1 as vis_util
import time


cap = cv2.VideoCapture('http://192.168.0.100:4747/mjpegfeed?640x480')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_real7.avi',fourcc, 10.0, (800,600))
MODEL_NAME = 'fire_inference_graph_v1.pb'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'fire_label_map.pbtxt')

fire_detection_graph = tf.Graph()
with fire_detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
  
  fire_session=tf.Session(graph=fire_detection_graph)
fire_category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


fire_image_tensor = fire_detection_graph.get_tensor_by_name('image_tensor:0')
fire_detection_boxes = fire_detection_graph.get_tensor_by_name(
    'detection_boxes:0')
fire_detection_scores = fire_detection_graph.get_tensor_by_name(
    'detection_scores:0')
fire_detection_classes = fire_detection_graph.get_tensor_by_name(
    'detection_classes:0')
fire_num_detections = fire_detection_graph.get_tensor_by_name(
    'num_detections:0')
print('Loaded Fire Model')
	
MODEL_NAME = 'smokes_inference_graph_v1.pb'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'smoke_label_map.pbtxt')

smoke_detection_graph = tf.Graph()
with smoke_detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
  smoke_session=tf.Session(graph=smoke_detection_graph)

smoke_category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


smoke_image_tensor = smoke_detection_graph.get_tensor_by_name('image_tensor:0')
smoke_detection_boxes = smoke_detection_graph.get_tensor_by_name(
    'detection_boxes:0')
smoke_detection_scores = smoke_detection_graph.get_tensor_by_name(
    'detection_scores:0')
smoke_detection_classes = smoke_detection_graph.get_tensor_by_name(
    'detection_classes:0')
smoke_num_detections = smoke_detection_graph.get_tensor_by_name(
    'num_detections:0')
print('Loaded smoke Model')

def fire_detect(image_expanded):
  (boxes, scores, classes, num) = fire_session.run(
      [fire_detection_boxes, fire_detection_scores,
          fire_detection_classes, fire_num_detections],
      feed_dict={fire_image_tensor: image_expanded})
  return (boxes,scores,classes,num)  

def smoke_detect(image_expanded):
  (boxes, scores, classes, num) = smoke_session.run(
      [smoke_detection_boxes, smoke_detection_scores,
          smoke_detection_classes, smoke_num_detections],
      feed_dict={smoke_image_tensor: image_expanded})
  return (boxes,scores,classes,num)    

def playAlarm(flag):
  if flag==True:
    winsound.PlaySound("2.wav", winsound.SND_ASYNC|winsound.SND_LOOP)

def stopAlarm(flag):
  if flag==False:
    winsound.PlaySound(None, winsound.SND_ASYNC)

    
sum_time=0
no_of_frames=0
to_play=True

while True:
  ret, image = cap.read()
  start=time.time()
  if ret==False:
    break
  image_np=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_np_expanded = np.expand_dims(image_np, axis=0)
  (boxes,scores,classes,num)=fire_detect(image_np_expanded)
  if scores[0][0]<0.3:
    (boxes,scores,classes,num)=smoke_detect(image_np_expanded)
    category_index=smoke_category_index
    print('no_fire &', end =" ")
    if scores[0][0]<0.3:
      print('no_smoke')
      stopAlarm(to_play)
      to_play=True
    else:
      print('smoke:',scores[0][0])
      playAlarm(to_play)
      to_play=False	  
    lab=1
  else: 
    category_index=fire_category_index
    print('fire:',scores[0][0])
    playAlarm(to_play)
    to_play=False
    lab=0
  vis_util.visualize_boxes_and_labels_on_image_array(
      image,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=8,min_score_thresh=.3,lab=lab,skip_scores=True)
  out.write(cv2.resize(image, (800, 600)))
  cv2.imshow('object detection', cv2.resize(image, (800, 600)))
  end=time.time()
  time_taken=end-start
  sum_time=sum_time+time_taken
  no_of_frames=no_of_frames+1
  print('time',time_taken)
  print('fps',1/time_taken)

  if cv2.waitKey(25) & 0xFF == ord('q'):
      
      break

cv2.destroyAllWindows()
out.release()
Avg_time=sum_time/no_of_frames
print('Avg time',Avg_time)
print('Avg fps',1/Avg_time)
  