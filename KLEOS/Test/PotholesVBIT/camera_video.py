# importing necessary libraries
import cv2 as cv
import time
import geocoder
import os
import re
from picamera.array import PiRGBArray
from picamera import PiCamera

#function for deleting all files within a folder
def delete_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
            
            
#delete any pre-existing file within the folders pothole_coordinates and pothole_size
delete_files('/home/pi/Documents/KLEOS/Test/PotholesVBIT/pothole_coordinates')
delete_files('/home/pi/Documents/KLEOS/Test/PotholesVBIT/pothole_size')



# reading label name from obj.names file
class_name = []
with open(os.path.join('/home/pi/Documents/KLEOS/Test/PotholesVBIT/project_files/obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# importing model weights and config file
# defining the model parameters
net1 = cv.dnn.readNet('/home/pi/Documents/KLEOS/Test/PotholesVBIT/project_files/yolov4_tiny.weights',
                      '/home/pi/Documents/KLEOS/Test/PotholesVBIT/project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# defining the video source (0 for camera or file name for video)
"""cap = cv.VideoCapture("WhatsApp Video 2023-02-24 at 19.23.15.mp4")
width = cap.get(3)
height = cap.get(4)"""

camera = PiCamera()
camera.resolution = (640, 480)
width = 640
height = 480
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)
"""result = cv.VideoWriter('result.avi',
                        cv.VideoWriter_fourcc(*'MJPG'),
                        10, (int(width), int(height)))"""

# defining parameters for result saving and get coordinates
# defining initial values for some parameters in the script
g = geocoder.ip('me')
result_path = "/home/pi/Documents/KLEOS/Test/PotholesVBIT/pothole_coordinates"
size_path = "/home/pi/Documents/KLEOS/Test/PotholesVBIT/pothole_size"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    classes, scores, boxes = model1.detect(image, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w*h
        area = width*height
        # drawing detection boxes on frame for detected potholes and saving coordinates txt and photo
        if (len(scores) != 0 and scores[0] >= 0.7):
            if ((recarea/area) <= 0.1 and box[1] < 600):
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                dim = (w, h)
                acc = str(round(scores[0]*100, 2))
                k = (dim, acc)
                with open(os.path.join(size_path, 'pothole' + str(i)+'.txt'), 'w') as f:
                    str_k = str(k)
                    str_k = str_k.replace('(', '')
                    str_k = str_k.replace(')', '')
                    str_k = str_k.replace("'", '')
                    f.write(str_k)
                cv.putText(image, "%" + acc + " " + label,
                           (box[0], box[1]-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                if (i == 0):
                    cv.imwrite(os.path.join(
                        result_path, 'pothole' + str(i)+'.jpg'), image)
                    with open(os.path.join(result_path, 'pothole' + str(i)+'.txt'), 'w') as f:
                        f.write(str(g.latlng))
                        i = i+1
                if (i != 0):
                    if ((time.time()-b) >= 2):
                        cv.imwrite(os.path.join(
                            result_path, 'pothole'+str(i)+'.jpg'), image)
                        with open(os.path.join(result_path, 'pothole'+str(i)+'.txt'), 'w') as f:
                            f.write(str(g.latlng))
                            b = time.time()
                            i = i+1
    # writing fps on frame
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    cv.putText(image, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    # showing and saving result
    cv.imshow('frame', image)
    #result.write(image)
    key = cv.waitKey(1)
    rawCapture.truncate(0)
    if key == ord('q'):
        break
    
        
# detection loop
"""while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break"""

    # analysis the stream with detection model
    

#result.release()
#cv.destroyAllWindows()
