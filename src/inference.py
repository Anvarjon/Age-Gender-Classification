from model_utils import AgeGenderDetector
from argparse import ArgumentParser
import logging
logging.disable(logging.WARNING)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_path', default='models/', help='Model path')
    parser.add_argument('--audio_file', default='data/CommonVoice/Audio/cv-valid-test/sample-000001.mp3', help='Audio file path')
    parser.add_argument('--cls_task', default='age', choices=['age', 'gender', 'gender_age'], help="Classification task")
    args = parser.parse_args()
    return args

def detect(args):
    # Evaluating the best model on the test dataset
    age_gender_detector = AgeGenderDetector(args.model_path, args.cls_task)
    detection = age_gender_detector.getPrediction(args.audio_file)
    print('Model prediction ==> ', detection)

if __name__ == '__main__':
    args = parse_args()
    detect(args)