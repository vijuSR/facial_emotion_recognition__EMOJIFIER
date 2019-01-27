import logging
import pickle
import os
import sys
import json
import cv2
import numpy as np
import glob
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import src
from src.__init__ import *


def image_reader(image_path_list):
    
    image = cv2.imread(image_path_list[0], 0)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)
    
    for img_path in image_path_list[1:]:
        image = np.concatenate(
            (
                image, 
                np.expand_dims(
                    cv2.resize(cv2.imread(img_path, 0), (48, 48)), 
                    axis=0
                )
            ), 
            axis=0
        )
        
    return image


def image_label_generator(emotion_map):
    labels = []
    
    _i = 0

    image_lists = []
    for k, v in tqdm.tqdm(emotion_map.items()):

        path = os.path.join(FACE_IMAGES_PATH, k)
        logger.debug('reading images at path: {}'.format(path))
        image_list = glob.glob(path+'/*.png')
        logger.debug('length images list: {}'.format(len(image_list)))
        image_lists.append(image_list)
        labels.extend([v]*len(image_list))
        
    images = np.vstack((image_reader(image_list) for image_list in image_lists))

    return images, labels


def train_test_splitter(images, labels):
    dataset = [(image, label) for image, label in zip(images, labels)]

    dataset_size = len(dataset)
    trainset_size = int(.8 * dataset_size)
    testset_size = dataset_size - trainset_size
    logger.debug('Dataset size: {}'.format(dataset_size))
    
    np.random.shuffle(dataset)
    
    # PAY ATTENTION HERE: YOU CAN ALSO ADD DEV-SET :)
    trainset, testset = dataset[:trainset_size], dataset[trainset_size:]
    
    logger.debug('Trainset size: {}, Testset size: {}'.format(
        len(trainset), len(testset)
    ))
    
    logger.debug('concatinating the train images on axis 0')
    train_image = np.vstack((tr[0] for tr in tqdm.tqdm(trainset[:])))
    logger.debug('concatinating the train labels on axis 0')
    train_label = [tr[1] for tr in tqdm.tqdm(trainset[:])]

    logger.info('concatinating the test images on axis 0')
    test_image = np.vstack((te[0] for te in tqdm.tqdm(testset[:])))
    logger.debug('concatinating the test labels on axis 0')
    test_label = [te[1] for te in tqdm.tqdm(testset[:])]
    
    logger.debug('train-images-shape: {}, test-images-shape: {}'.format(
        train_image.shape, test_image.shape
    ))
        
    return (train_image, train_label), (test_image, test_label)


def create_dataset(images, labels):
    
    images = np.reshape(images, (-1, 48*48))
    logger.debug('images-shape: {}, length-labels: {}'.format(
        images.shape, len(labels)
    ))
    
    train, test = train_test_splitter(images, labels)
    
    
    train_dict = {
        'data': train[0],
        'labels': train[1]
    }
    test_dict = {
        'data': test[0],
        'labels': test[1]
    }
    
    with open(os.path.join(DATASET_SAVE_PATH, 'train_batch_0'), 'wb') as file:
        pickle.dump(train_dict, file)
        logger.info('dataset: trainset-dict pickled and saved at {}'.format(DATASET_SAVE_PATH))
        
    with open(os.path.join(DATASET_SAVE_PATH, 'test_batch_0'), 'wb') as file:
        pickle.dump(test_dict, file)
        logger.info('dataset: testset-dict pickled and saved at {}'.format(DATASET_SAVE_PATH))
        
    logger.info('dataset created :)')


def condition_satisfied(emotion_map):
    for emotion_class in emotion_map.keys():
        path = os.path.join(FACE_IMAGES_PATH, emotion_class)

        if not os.path.exists(path):
            logger.error('Please capture images for "{}" emotion-class as well'.format(
                emotion_class
            ))
            logger.error('FAIL.')
            return False

    return True


if __name__ == '__main__':

    logger = logging.getLogger('emojifier.dataset_creator')
    FACE_IMAGES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'images')
    DATASET_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'dataset')

    if not os.path.exists(DATASET_SAVE_PATH):
        os.makedirs(DATASET_SAVE_PATH)

    if condition_satisfied(EMOTION_MAP):
        _images, _labels = image_label_generator(EMOTION_MAP)
        create_dataset(_images, _labels)
