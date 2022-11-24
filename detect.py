import cv2
import time
import sys
import numpy as np

def build_model(is_cuda):
    net = cv2.dnn.readNet("/content/drive/MyDrive/FaceSwap-Kevin/final.onnx")
    if is_cuda:
        print("Attempt to use CUDA")
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
NMS_THRESHOLD = 0.1
CONFIDENCE_THRESHOLD = 0.6

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
    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    max_area=-1
    area_box=[]
    for r in range(rows):
        row=output_data[0][0][r]
        confidence = row[4]
        
        
        if confidence >= CONFIDENCE_THRESHOLD:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]

            if (classes_scores[class_id] > 0.3) and class_id==1:
                confidences.append(confidence)

                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                if width*height>max_area:
                    max_area=width*height
                    area_box=np.array([left, top, left+width, top+height])
                #box = np.array([left, top, left+width, top+height])
                
                
    boxes.append(area_box)

    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD) 
    #print(confidences)
    #result_class_ids = []
    #result_confidences = []
    #result_boxes = []

    #for i in indexes:
    #    result_confidences.append(confidences[i])
    #    result_class_ids.append(class_ids[i])
     #   result_boxes.append(boxes[i])

    return boxes#result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


colors = [(255, 255, 0), (0, 255, 0)]

is_cuda = True #set to false for CPU

net = build_model(is_cuda)

def y_inference(img_arr):
    
    yolo_res = []
    for frame in img_arr:
      #The following lines add a black border on all sides of the image. (For pics that are zoomed into head)
      # h,w,_=frame.shape
      # img_dummy=np.zeros((2*h,2*w,3),dtype=np.uint8)
      # img_dummy[int(0.5*h):int(1.5*h),int(0.5*w):int(1.5*w),::]=frame
      # frame=img_dummy
      inputImage = format_yolov5(frame)
      outs = detect(inputImage, net)
      plt.imshow(frame)
      plt.show()
      boxes = wrap_detection(inputImage, outs)
      
      for  box in boxes:
          if  len(list(box))==0:
            print("--123")
            continue
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
          
          cropped_img = inputImage[int(ymin):int(ymax),int(xmin):int(xmax)]
          yolo_res.append(cropped_img)
          
    return yolo_res
