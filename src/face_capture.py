# Usage: face_capture.py -e emotion-name -n number-of-images
# #1 emotion-name -- emotion-class-name that you want these set of
# images to be labelled as.
# #2 number-of-images -- number of images to capture with the given
# emotion-class.

# It generates the face crops for creating the dataset.
# It captures the frame from the video-feed from your cam
# and detects the faces in it and saves cropped face as a
# png file.

import time
import sys
import os
import json
import logging
import argparse
import numpy as np
import cv2


sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from src.__init__ import *


def generate_faces(saving_path, num):

    face_cascade = cv2.CascadeClassifier(config_parser['OPEN_CV']['cascade_classifier_path'])

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    i = 0

    while i < num:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Operations on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect the faces, bounding boxes
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # draw the rectangle (bounding-boxes)
        for (x,y,w,h) in faces:

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

            img_path = os.path.join(saving_path, '_' + str(time.time()) + '.png')
            cv2.imwrite(img_path, frame[y:y+h, x:x+w, :])

            logger.info('{i} path: {path} created'.format(i=i, path=img_path))
            i += 1

        cv2.imshow('faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    logger = logging.getLogger('emojifier.face_capture')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-e', '--emotion_name',
        type=str,
        choices=EMOTION_MAP.keys(),
        help='Emotion class name for which to capture face-expression.',
        required=True
    )
    parser.add_argument(
        '-n', '--number_of_images',
        type=int,
        help='Number of images to capture for the given emotion.',
        required=True
    )

    args = parser.parse_args()

    EMOTION_CLASS = args.emotion_name
    N = args.number_of_images

    cap = cv2.VideoCapture(0)

    PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'images', EMOTION_CLASS)

    generate_faces(PATH, N)
