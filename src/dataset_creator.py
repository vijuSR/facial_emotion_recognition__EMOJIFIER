import logging
import pickle
import os
import sys
import cv2
import numpy as np
import glob
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import src


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


def image_label_generator():
    labels = []
    
    _i = 0

    for k, v in tqdm.tqdm(emoji_dict.items()):

        path = os.path.join(FACE_IMAGES_PATH, k)

        if _i == 0:
            image_list = glob.glob(path+'/*.png')
            logger.debug('length images list: {}'.format(len(image_list)))
            images = image_reader(image_list)
            _i += 1
        else:
            image_list = glob.glob(path+'/*.png')
            images = np.concatenate(
                (
                    images, 
                    image_reader(image_list)
                ), 
                axis=0
            )
            
        labels.extend([v]*len(image_list))
        
    return images, labels


def train_test_splitter(images, labels):
    data = [(image, label) for image, label in zip(images, labels)]

    logger.debug('length data: {}'.format(len(data)))
    
    np.random.shuffle(data)
    
    # PAY ATTENTION HERE: CHOOSE THIS NUMBER TO SPLIT YOUR DATA SET, 
    # YOU CAN ALSO ADD DEV-SET :)
    train, test = data[:1200], data[1200:]
    
    logger.debug('length-train: {}, length-test: {}'.format(
        len(train), len(test)
    ))
    
    train_image, train_label = train[0][0], [train[0][1]]
    test_image, test_label = test[0][0], [test[0][1]]
        
    train_image = np.expand_dims(train_image, axis=0)
    test_image = np.expand_dims(test_image, axis=0)
    
    logger.debug('train-images-shape: {}, test-images-shape: {}'.format(
        train_image.shape, test_image.shape
    ))
    
    logger.info('concatinating the train images on axis 0')
    for tr in tqdm.tqdm(train[1:]):
        train_image = np.concatenate((train_image, np.expand_dims(tr[0], axis=0)), axis=0)
        train_label.append(tr[1])

    logger.info('concatinating the test images on axis 0')
    for te in tqdm.tqdm(test[1:]):
        test_image = np.concatenate((test_image, np.expand_dims(te[0], axis=0)), axis=0)
        test_label.append(te[1])
        
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
        logger.info('dataset: train-dict pickled and saved !')
        
    with open(os.path.join(DATASET_SAVE_PATH, 'test_batch_0'), 'wb') as file:
        pickle.dump(test_dict, file)
        logger.info('dataset: test-dict pickled and saved !')
        
    logger.info('dataset created :)')


if __name__ == '__main__':

    logger = logging.getLogger('emojifier.dataset_creator')
    FACE_IMAGES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'images')
    DATASET_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'dataset')

    if not os.path.exists(DATASET_SAVE_PATH):
        os.makedirs(DATASET_SAVE_PATH)
    
    # CHANGE THIS DICT AND THE "image_label_generator" function
    # if you have different class of emotions
    emoji_dict = {
        'smile': 0,
        'kiss': 1,
        'tease': 2,
        'angry': 3,
        'glass': 4
    }
    
    i, l = image_label_generator()
    create_dataset(i, l)
