import os
import logging
import json


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] (%(message)s)',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info('Loggers ready !')

# CHANGE THIS DICT AND THE "image_label_generator" function
# if you have different class of emotions
EMOTION_MAP_PATH = os.path.join(
    os.path.dirname(__file__), 
    os.pardir, 
    'emotion_map.json'
)
with open(EMOTION_MAP_PATH) as json_file:
    EMOTION_MAP = json.load(json_file)
    logging.debug('Emotion Map: {}'.format(EMOTION_MAP))
