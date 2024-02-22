import os
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from data_utils import TFRecordDatasetProcessor
from model_builder import ModelBuilder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from seaborn import heatmap
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
import logging
plt.rcParams.update({'font.size': 10})
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)

class ModelTrainer:
    def __init__(self, dataset_dir, cls_task):
        self.dataset_dir = dataset_dir
        self.cls_task = cls_task
        self.reg_value = 0.01
        self._max_symbols = 150
        self._print_symbol = '='
    
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        print(num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol)
        
    def _setTrainingMonitoringSettings(self):
        self.log_file = f'results/{self.cls_task}/train_logs_{self.cls_task}.csv'
        self.logger = CSVLogger(self.log_file)
        self.best_model_filepath = 'models/' + self.cls_task + '/best_model_' + self.cls_task + '.h5'
        self.best_model_saver = ModelCheckpoint(self.best_model_filepath, monitor='val_accuracy', save_best_only=True)
        self._callbacks = [self.logger, self.best_model_saver]
    
    def getBestModelFilepath(self):
        return self.best_model_filepath
    
    def setTrainingParameters(self, batch_size, learning_rate, num_epochs):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.opt = Adam(learning_rate=learning_rate)
    
    def _buildDataset(self):
        self.dataset_processor = TFRecordDatasetProcessor(self.dataset_dir)
        self.tr_dataset = self.dataset_processor.getTrainTFRecordDataset(self.cls_task, self.batch_size)
        self.val_dataset = self.dataset_processor.getValidationTFRecordDataset(self.cls_task, self.batch_size)
        self.num_classes = self.dataset_processor.getNumberOfClasses(self.cls_task)
        self.category_names = self.dataset_processor.getCategoryNames(self.cls_task)
    
    def _buildModel(self, show_summary=False):
        h, w = self.dataset_processor.getInputImageSize()
        inp_shape = (h, w, 3)
        self.model_builder = ModelBuilder(inp_shape, self.num_classes, self.reg_value)
        self.model = self.model_builder.build_model(show_summary=show_summary)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=self.opt, metrics=["accuracy"])
    
    def startTraining(self, show_model_summary=False, show_live_plot=False):
        self._buildDataset()
        self._buildModel(show_summary=show_model_summary)
        self._setTrainingMonitoringSettings()
        if show_live_plot:
            self.model_monitoring = ModelTrainingMonitoring(self.log_file, self.cls_task)
            self._callbacks.append(self.model_monitoring)
        self.model.fit(self.tr_dataset, epochs=self.num_epochs, validation_data=self.val_dataset, callbacks=self._callbacks, shuffle=False)

class ModelEvaluator:
    def __init__(self, dataset_dir, cls_task, model_path):
        self.dataset_dir = dataset_dir
        self.cls_task = cls_task
        self.model_path = model_path
        self.dataset_processor = TFRecordDatasetProcessor(self.dataset_dir)
        self.category_names = self.dataset_processor.getCategoryNames(self.cls_task)
        self.cm_plot_file = f'results/{self.cls_task}/confusion_matrix_{self.cls_task}.png'
        self.test_log_file = f'results/{self.cls_task}/test_log_{self.cls_task}.txt'
        self.batch_size = 128
        self._max_symbols = 150
        self._print_symbol = '='
    
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        message = num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol
        print(message)
        return message

    def evaluateBestModel(self):
        message = self.printTitle(f" Evaluating the best model on the test dataset for {self.cls_task} classification ")
        self.test_dataset = self.dataset_processor.getTestTFRecordDataset(self.cls_task, self.batch_size)
        self.best_model = load_model(self.model_path)
        labels = []
        predictions = []
        for img, lab in self.test_dataset:
            labels.extend(lab.numpy())
            pred = self.best_model.predict(img, verbose=0)
            predictions.extend(np.argmax(pred, axis=1))

        acc = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)
        clr = classification_report(labels, predictions, target_names=self.category_names)
        
        with open(self.test_log_file, 'w') as f: 
            message = self.printTitle(" Accuracy score ")
            f.write(message + '\n')
            
            message = "{:.2f}".format(acc)
            f.write(message + '\n')
            print(message)
            
            message = self.printTitle(" Confusion matrix ")
            f.write(message + '\n')
            
            print(cm)
            for row in cm:
                # Convert each row to a string and join them with a tab separator
                row_str = '\t'.join(map(str, row))
                # Write the row string followed by a newline character
                f.write(row_str + '\n')
            
            message = self.printTitle(" Classification report ")
            f.write(message + '\n')
            
            print(clr)
            f.write(clr)
        
        plot_confusion_matrix(cm, target_names=self.category_names, filename=self.cm_plot_file)


class ModelTrainingMonitoring(tf.keras.callbacks.Callback):
    def __init__(self, log_file, cls_task):
        plt.rcParams['figure.figsize'] = (18, 4)
        plt.ion()
        self.cls_task = cls_task
        self._loss_title = f'Loss monitoring for {self.cls_task} classification'
        self._acc_title = f'Accuracy monitoring for {self.cls_task} classification'
        self._fig_title = f"Training metrics monitoring for {cls_task} classification"
        self.main_fig = plt.figure(self._fig_title, constrained_layout=True)
        self.gridspec = GridSpec(1, 2, figure=self.main_fig)
        self.ax_acc = self.get_subplot(self.gridspec[0, 0], title=self._acc_title)
        self.ax_loss = self.get_subplot(self.gridspec[0, 1], title=self._loss_title)
        self.log_file = log_file
        self.wait_time = 0.001

    def get_subplot(self, grid, title):
        ax = self.main_fig.add_subplot(grid)
        # ax.set_axis_off()
        ax.set_title(title)
        return ax

    def on_epoch_end(self, epoch, logs=None):
        df = pd.read_csv(self.log_file)
        epochs = df['epoch'].values
        tr_loss = df['loss'].values
        tr_acc = df['accuracy'].values
        val_loss = df['val_loss'].values
        val_acc = df['val_accuracy'].values
        self.plotAccuracy(epochs, tr_acc, val_acc)
        self.plotLoss(epochs, tr_loss, val_loss)
        plt.draw()
        plt.pause(self.wait_time)

    def plotLoss(self, epochs, tr_loss, val_loss):
        self.ax_loss.cla()
        self.ax_loss.set_title(self._loss_title)
        self.ax_loss.plot(epochs, tr_loss, label='Train loss')
        self.ax_loss.scatter(epochs[-1], tr_loss[-1], s=10)
        self.ax_loss.plot(epochs, val_loss, label='Validation loss')
        self.ax_loss.scatter(epochs[-1], val_loss[-1], s=10)
        self.ax_loss.set_xlabel('Number of epochs')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.set_aspect('auto')
        self.ax_loss.grid()

    def plotAccuracy(self, epochs, tr_acc, val_acc):
        self.ax_acc.cla()
        self.ax_acc.set_title(self._acc_title)
        self.ax_acc.plot(epochs, tr_acc, label='Train accuracy')
        self.ax_acc.scatter(epochs[-1], tr_acc[-1], s=10)
        self.ax_acc.plot(epochs, val_acc, label='Validation accuracy')
        self.ax_acc.scatter(epochs[-1], val_acc[-1], s=10)
        self.ax_acc.set_xlabel('Number of epochs')
        self.ax_acc.set_ylabel('Accuracy')
        self.ax_acc.legend(loc='lower right')
        self.ax_acc.set_aspect('auto')
        self.ax_acc.grid()

    def on_train_end(self, logs=None):
        plt.savefig(f"results/{self.cls_task}/training_metrics_{self.cls_task}.png", bbox_inches='tight')
        plt.close()


def plot_confusion_matrix(cm, target_names, filename, title='Confusion matrix of Common Voice dataset', normalize=True):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm *= 100

    ax = heatmap(cm, annot=True, fmt='.1f', cmap='Blues', xticklabels=target_names, yticklabels=target_names, linecolor='white', linewidths=.5, cbar_kws={'pad': 0.02})
    ax.set(xlabel='Predicted label', ylabel='True label')
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.savefig(filename, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
