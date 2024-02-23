from model_utils import ModelEvaluator
from argparse import ArgumentParser
import logging
logging.disable(logging.WARNING)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/CommonVoice', help='Dataset directory')
    parser.add_argument('--model_path', default='models/', help='Model path')
    parser.add_argument('--cls_task', default='age', choices=['age', 'gender', 'gender_age'], help="Classification task")
    args = parser.parse_args()
    return args

def test_model(args):
    # Evaluating the best model on the test dataset
    model_evaluator = ModelEvaluator(args.dataset_dir, args.cls_task, args.model_path)
    model_evaluator.evaluateBestModel()

if __name__ == '__main__':
    args = parse_args()
    test_model(args)
    