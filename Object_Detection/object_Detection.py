# ---------------------------    YOLO Object Detection      -----------------------------------

# Importing Modules
import cv2
import numpy as np

#Variables
conf_thres = 0.45
nms_thres = 0.2

classNames = []

# Available object classes 
namesPath = 'Object_Detection_Files/coco.names'
f = open(namesPath,'rt')
classNames = f.read().rstrip('\n').split('\n')
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classNames)))

# Resource files path
configPath = 'Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'Object_Detection_Files/frozen_inference_graph.pb'

#Pre_built Model
net = cv2.dnn_DetectionModel(weightsPath,configPath)
# Play with the values in order to observe change in Output.
net.setInputSize(240,240)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


# Reading Video
def readVideo(path):
    cap = cv2.VideoCapture(path)

    while(True):
        success,img = cap.read()
        classIds,confidences, bound_boxes = net.detect(img,confThreshold = conf_thres)
        print(classIds,confidences,bound_boxes)
        bound_boxes = list(bound_boxes)
        confidences = list(np.array(confidences).reshape(1,-1)[0])
        confidences = list(map(float,confidences))

        #Applying Non-Maximum Suppression to get rid off multiple bounding boxes.
        out_indices = cv2.dnn.NMSBoxes(bound_boxes,confidences,conf_thres,nms_thres)

        for i in out_indices:
            i = i[0]
            box = bound_boxes[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            #Drawing Rectangle and name on object detected
            cv2.rectangle(img,(x,y),(x+w,y+h),color=colors[i],thickness=2)
            cv2.putText(img,classNames[classIds[i][0]-1].upper()+" "+str(round(confidences[i]*100,2)),(x+10,y+30),cv2.FONT_HERSHEY_COMPLEX,1,colors[i],2)        
        cv2.imshow("Output",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


#Reading Image
def readImage(path):
    img = cv2.imread(path)
    # img = cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
    classIds,confidences, bound_boxes = net.detect(img,confThreshold = conf_thres)
    print(classIds,confidences,bound_boxes)
    bound_boxes = list(bound_boxes)
    confidences = list(np.array(confidences).reshape(1,-1)[0])
    confidences = list(map(float,confidences))

    #Applying Non-Maximum Suppression to get rid off multiple bounding boxes.
    out_indices = cv2.dnn.NMSBoxes(bound_boxes,confidences,conf_thres,nms_thres)

    for i in out_indices:
        i = i[0]
        box = bound_boxes[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        #Drawing Rectangle and name on object detected
        cv2.rectangle(img,(x,y),(x+w,y+h),color=colors[i],thickness=2)
        cv2.putText(img,classNames[classIds[i][0]-1].upper()+" "+str(round(confidences[i]*100,2)),(x+10,y+30),cv2.FONT_HERSHEY_COMPLEX,1,colors[i],2)    
    cv2.imshow("Output",img)
    cv2.waitKey(0)

readVideo("Object_Detection_Files/MUMBAI TRAFFIC _ INDIA.mp4")

readImage("Object_Detection_Files/rush_Place2.jpg")

cv2.destroyAllWindows()