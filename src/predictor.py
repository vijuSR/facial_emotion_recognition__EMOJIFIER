import os
import sys
import cv2
import time
import logging
import json
import tensorflow as tf
import numpy as np
import glob
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from src.trainer import model
from src.__init__ import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def inference(sess, gray_img_input):
    
    img = gray_img_input.reshape(1, 48, 48, 1).astype(float) / 255
    
    y_c = sess.run(y_conv, feed_dict={X:img, keep_prob:1.0})
    
    y_c = softmax(y_c)
    p = np.argmax(y_c, axis=1)
    score = np.max(y_c)
    logger.debug('''
        softmax-out: {},
        predicted-index: {},
        predicted-emotion: {},
        confidence: {}'''.format(y_c, p[0], index_emo[p[0]], score))
    return p[0], score
        

def from_cam(sess):
    
    face_cascade = cv2.CascadeClassifier(config_parser['OPEN_CV']['cascade_classifier_path'])
    cap = cv2.VideoCapture(0)

    font               = cv2.FONT_HERSHEY_SIMPLEX
    fontScale          = 1
    fontColor          = (255,255,255)
    lineType           = 2

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Operations on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect the faces, bounding boxes
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # draw the rectangle (bounding-boxes)
        for (x,y,w,h) in faces:
            cv2.rectangle(gray, (x,y), (x+w, y+h), (255,0,0), 2)
            bottomLeftCornerOfText = (x+10,y+h+10)

            face_img_gray = gray[y:y+h, x:x+w]
            face_img_gray = cv2.resize(face_img_gray, (48, 48))
            s = time.time()
            p, confidence = inference(sess, face_img_gray)
            logger.critical('model inference time: {}'.format(time.time() - s))
            
            if confidence > 0.5:
            
                img2 = emoji_to_pic[index_emo[p]]
                img2 = cv2.resize(img2, (w, h))

                alpha = img2[:,:,3]/255.0

                frame[y:y+h, x:x+w, 0] = frame[y:y+h, x:x+w, 0] * (1-alpha) + alpha * img2[:,:,0]
                frame[y:y+h, x:x+w, 1] = frame[y:y+h, x:x+w, 1] * (1-alpha) + alpha * img2[:,:,1]
                frame[y:y+h, x:x+w, 2] = frame[y:y+h, x:x+w, 2] * (1-alpha) + alpha * img2[:,:,2]

                cv2.putText(frame,f'Confidence: {round(confidence, 2)}',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

            else: 
                cv2.putText(frame,'NEUTRAL', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)

        # Display the resulting frame
        cv2.imshow('gray-scale', gray)
        cv2.imshow('faces', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    logger = logging.getLogger('emojifier.predictor')
    CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'model_checkpoints')
    EMOJI_FILE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'emoji')
    tf.reset_default_graph()

    # used to map the output from the prediction to the emotion class
    index_emo = {v:k for k,v in EMOTION_MAP.items()}
    
    # dictionary of emoji name and the corresponding read image
    emoji_to_pic = {k: None for k in EMOTION_MAP.keys()}

    emoji_png_files_path = os.path.join(EMOJI_FILE_PATH, '*.png')
    files = glob.glob(emoji_png_files_path)

    logger.info('loading the emoji png files in memory ...')

    import platform

    if platform.system() == 'Windows':
        split_string = '\\'
    else:
        split_string = '/'

    for file in tqdm.tqdm(files):
        logger.debug('file path: {}'.format(file))
        emoji_to_pic[file.split(split_string)[-1].split('.')[0]] = cv2.imread(file, -1)

    X = tf.placeholder(
        tf.float32, shape=[None, 48, 48, 1]
    )
    
    keep_prob = tf.placeholder(tf.float32)

    y_conv = model(X, keep_prob)
    
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        saver.restore(sess, os.path.join(CHECKPOINT_SAVE_PATH, 'model.ckpt'))

        logger.info('Opening the camera for getting the video feed ...')
        logger.info('PRESS "q" AT ANY TIME TO EXIT!')
        from_cam(sess)
