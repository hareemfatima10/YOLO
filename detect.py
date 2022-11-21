import cv2
import time
import sys
import numpy as np

def build_model(is_cuda):
    net = cv2.dnn.readNet("/content/drive/MyDrive/FaceSwap-Kevin/final.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = -10

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    #outputs = net.forward()
    #

    return outputs



def load_classes():
    
    return ["person","head"]

class_list = load_classes()

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data[0].shape[1]
    print("here",rows)
    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row=output_data[0][0][r]
        #print(row.shape)
        confidence = row[4]
        #print(confidence)
        if confidence >= 0.1:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > 0.1):

                confidences.append(confidence)
              
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, left+width, top+height])
                
                
                boxes.append(box)
    print(len(boxes))

                

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


colors = [(255, 255, 0), (0, 255, 0)]

is_cuda = False

net = build_model(is_cuda)


def y_inference(img_arr):
    
    yolo_res = []
    for frame in img_arr:
      inputImage = format_yolov5(frame)
      outs = detect(inputImage, net)

      class_ids, confidences, boxes = wrap_detection(inputImage, outs)

      for (classid, confidence, box) in zip(class_ids, confidences, boxes):
          if classid==1:
              color = [0,0,255]
              xmin,ymin,xmax,ymax=box
              w=xmax-xmin
              h=ymax-ymin
              xmin -= abs(int(0.15 * (xmax - xmin)))
              xmax += abs(int(0.15 * (xmax - xmin)))
              ymin -= abs(int(0.15 * (ymax - ymin)))
              ymax += abs(int(0.15 * (ymax - ymin)))
              xmin, xmax, ymin, ymax = abs(xmin), abs(xmax), abs(ymin), abs(ymax)
              
              x_center = np.average([xmin, xmax])
              y_center = np.average([ymin, ymax])
              size = max(abs(xmax-xmin), abs(ymax-ymin))
                
              xmin, xmax = x_center-size/2, x_center+size/2
              ymin, ymax = y_center-size/2, y_center+size/2
              
              #cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)), color, 2)
              cropped_img = frame[int(ymin):int(ymax),int(xmin):int(xmax)]
              
      yolo_res.append(cropped_img)
    return yolo_res
