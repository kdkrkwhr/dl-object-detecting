"""Use yolo v3
"""
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO

def process_image(img):   # 3) Frame 크기 재설정

    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

def draw(image, boxes, scores, classes, all_classes):   # 4) Object Detecting 및 처리

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        print('cl : ', cl)
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1, cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

def detect_image(image, yolo, all_classes):   # 2) 하나의 Frame 처리

    pimage = process_image(image)
    boxes, classes, scores = yolo.predict(pimage, image.shape)

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image

def detect_video(video, yolo, all_classes):  # 1) Video 재생

    cap = cv2.VideoCapture(video)

    while True:
        res, frame = cap.read()
        if res == True:

            image = detect_image(frame, yolo, all_classes)

            cv2.imshow("KERAS", image)

            # cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elif res == False:
            cap = cv2.VideoCapture(video)
            continue
    
if __name__ == '__main__':
    yolo = YOLO(0.6, 0.5)

    with open('data/coco_classes.txt') as f:
        class_names = f.readlines()

    all_classes = [c.strip() for c in class_names]

    video = 'videos/test/library1.mp4'
    detect_video(video, yolo, all_classes)
