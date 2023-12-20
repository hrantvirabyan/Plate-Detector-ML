from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import string
import easyocr
motion_tracker=Sort()
results={}
#import models
car_model=YOLO("yolov8n.pt")
license_plate_detector=YOLO("license_plate_detector.pt")

cap = cv2.VideoCapture('./sample.mp4')
vehicles=[2,3,5,7]


frame_num=-1
ret = True
while ret:
    frame_num+=1    
    ret, frame = cap.read()
    if ret:
        results[frame_num]={}
        print(frame_num)
        detections = car_model(frame)[0]
        actual_detections=[]
        for i in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=i
            if int(class_id) in vehicles:
                actual_detections.append([x1,y1,x2,y2,score])


#track the vehicles

        track_ids=motion_tracker.update(np.asarray(actual_detections))

#detect license plate
        license_plates=license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate


#assign license to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
        #crop license plate
        # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plate_text,license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_num][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# write results
write_csv(results, './csv_output.csv')