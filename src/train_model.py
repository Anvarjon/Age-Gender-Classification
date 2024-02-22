from model_utils import ModelTrainer, ModelEvaluator
from argparse import ArgumentParser
import logging
logging.disable(logging.WARNING)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/CommonVoice', help='Dataset directory')
    parser.add_argument('--cls_task', default='age', choices=['age', 'gender', 'gender_age'], help="Classification task")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs')
    parser.add_argument('--show_summary', default=False, type=bool, help='Show model summary')
    parser.add_argument('--show_live_plot', default=False, type=bool, help='Show live plot for loss and accuracy')
    args = parser.parse_args()
    return args

def train_model(args):
    # Training model for a given classification task
    model_trainer = ModelTrainer(args.dataset_dir, args.cls_task)
    model_trainer.setTrainingParameters(args.batch_size, args.learning_rate, args.num_epochs)
    model_trainer.startTraining(args.show_summary, args.show_live_plot)
    
    # Evaluating the best model on the test dataset
    best_model_filepath = model_trainer.getBestModelFilepath()
    model_evaluator = ModelEvaluator(args.dataset_dir, args.cls_task, best_model_filepath)
    model_evaluator.evaluateBestModel()

if __name__ == '__main__':
    args = parse_args()
    train_model(args)
    