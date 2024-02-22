import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os.path as osp
import tensorflow as tf
from keras.utils import load_img, img_to_array
from shutil import move, rmtree
import pandas as pd
from tabulate import tabulate
from glob import glob
import librosa
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

class CommonVoiceDatasetPreprocessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self._invalid_gender = 'other'
        self._major_gender_age_groups = ['male-teens', 'male-twenties', 'male-thirties', 'male-fourties', 'male-fifties', 'male-sixties', 
                                         'female-teens', 'female-twenties', 'female-thirties', 'female-fourties', 'female-fifties', 'female-sixties']
        self._minor_gender_age_groups = ['male-seventies', 'male-eighties', 'female-seventies', 'female-eighties']
        self._modes = ['train', 'dev', 'test']
        self._csv_filenames = ['cv-valid-train.csv', 'cv-valid-dev.csv', 'cv-valid-test.csv']
        self._keep_keyword = '-valid-'
        self._max_symbols = 150
        self._print_symbol = '='
    
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        print(num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol)
    
    def printPrettyTable(self, dataframe, only_head=True):
        if only_head:
            print(tabulate(dataframe.head(n=3), headers='keys', tablefmt='fancy_grid', showindex=False, stralign='left', numalign='center', maxcolwidths=35))
        else:
            print(tabulate(dataframe, headers='keys', tablefmt='fancy_grid', showindex=False, stralign='left', numalign='center', maxcolwidths=35))
    
    def removeUnnecessaryFolderAndFiles(self):
        self.printTitle('Removing unnecessary folder and files')
        all_folders = [fol for fol in os.listdir(self.dataset_dir) if osp.isdir(osp.join(self.dataset_dir, fol))]
        all_csv_files = glob(self.dataset_dir + '/*.csv')
        print('Folder names: ')
        for fol in all_folders:
            if self._keep_keyword not in fol:
                print(osp.basename(fol))
                path = osp.join(osp.join(self.dataset_dir, fol), fol)
                self.deleteFiles(path)
        print('\nFile names: ')
        for fn in all_csv_files:
            if self._keep_keyword not in fn:
                print(osp.basename(fn))
                os.remove(fn)
    
    def deleteFiles(self, path):
        filenames = os.listdir(path)
        n = len(filenames)
        with alive_bar(n) as bar:
            for fn in filenames:
                os.remove(osp.join(path, fn))
                bar()
        os.removedirs(path)
    
    def loadDataset(self, mode):
        if mode in self._modes:
            if mode == 'train':
                dataframe = pd.read_csv(osp.join(self.dataset_dir, self._csv_filenames[0]))
                return dataframe
            elif mode == 'dev':
                dataframe = pd.read_csv(osp.join(self.dataset_dir, self._csv_filenames[1]))
                return dataframe
            elif mode == 'test':
                dataframe = pd.read_csv(osp.join(self.dataset_dir, self._csv_filenames[2]))
                return dataframe
        else:
            print(f'Invalid mode is given. Mode should be one of {self._modes} modes')
    
    def cleanDataset(self, dataframe):
        self.printTitle('Raw dataset information')
        self.printPrettyTable(dataframe)
        print(f'Number of files ===> {len(dataframe)}')
        ag_df = dataframe[['filename', 'age', 'gender']].copy()
        self.printTitle('Dataset information after removing unnecessary columns')
        self.printPrettyTable(ag_df)
        print(f'Number of files ===> {len(ag_df)}')
        ag_df.dropna(axis=0, inplace=True)
        print(f'Number of files after removing rows with NaN values ===> {len(ag_df)}')
        print(f"Age groups ===> {list(ag_df['age'].unique())}")
        print(f"Gender groups ===> {list(ag_df['gender'].unique())}")
        ag_df = ag_df[ag_df['gender'] != self._invalid_gender]
        print(f'Number of files after removing rows with `{self._invalid_gender}` gender group values ===> {len(ag_df)}')
        ag_df['gender_age'] = ag_df['gender'] + '-' + ag_df['age']
        # Calculate total entries for each gender
        total_males = len(ag_df[ag_df['gender'] == 'male'])
        total_females = len(ag_df[ag_df['gender'] == 'female'])
        # Calculate value counts for each 'gender_age' group
        value_counts = ag_df['gender_age'].value_counts()
        # Calculate percentages based on total entries for each gender
        percentages = {}
        for group, count in value_counts.items():
            gender = group.split('-')[0]
            if gender == 'male':
                percentages[group] = (count / total_males) * 100
            elif gender == 'female':
                percentages[group] = (count / total_females) * 100

        # Create a DataFrame to display the results
        result_df = pd.DataFrame({'Gender-Age group': value_counts.index,'Count': value_counts.values, 'Percentage': percentages.values()})
        result_df = result_df.sort_values(by='Percentage', ascending=False)
        self.printTitle('Dataset statistical information for gender-age groups')
        self.printPrettyTable(result_df, only_head=False)
        # Filter the DataFrame based on valid 'gender_age' groups
        ag_df = ag_df[ag_df['gender_age'].isin(self._major_gender_age_groups)]
        self.printTitle(f'Removing minority {self._minor_gender_age_groups} group files')
        print(f'Final number of files: {len(ag_df)}')
        return ag_df
    
    def writeCleanedDataset(self, mode):
        self.printTitle(f'Started writing clean {mode} dataset')
        self._audio_dir = osp.join(self.dataset_dir, 'Audio')
        folder_name = 'cv-valid-' + mode
        src_dir = osp.join(self.dataset_dir, folder_name)
        des_dir = osp.join(self._audio_dir, folder_name)
        if not osp.exists(des_dir):
            os.makedirs(des_dir)
        
        filenames = self._dataframe['filename'].to_list()
        n = len(filenames)
        with alive_bar(n) as bar:
            for fn in filenames:
                move(osp.join(src_dir, fn), des_dir)
                bar()
        self._dataframe.to_csv(osp.join(self._audio_dir, folder_name + '.csv'), index=False)
        self.printTitle(f'Deleting audio files with invalid labels for {mode} dataset')
        self.deleteFiles(osp.join(src_dir, folder_name))
        os.remove(osp.join(self.dataset_dir, folder_name + '.csv'))
    
    def startPreprocessing(self):
        self.removeUnnecessaryFolderAndFiles()
        for i, mode in enumerate(self._modes[1:]):
            self.printTitle(f'Preprocessing started for {mode} files')
            print(f'Filename ===> {self._csv_filenames[i]}')
            self._dataframe = self.loadDataset(mode)
            self._dataframe = self.cleanDataset(self._dataframe)
            self.writeCleanedDataset(mode)
        move(osp.join(self.dataset_dir, 'LICENSE.txt'), self._audio_dir)
        move(osp.join(self.dataset_dir, 'README.txt'), self._audio_dir)
        

class SpectrogramGenerator:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.audio_dir = osp.join(dataset_dir, 'Audio')
        self.spectrogram_dir = osp.join(dataset_dir, 'Spectrograms')
        os.makedirs(self.spectrogram_dir, exist_ok=True)
        self._modes = ['train', 'dev', 'test']
        # setting the default parameters for Spectrogram generation
        self.sampling_rate = 16000
        self.n_fft = 256
        self.num_overlap = 128
        self._max_symbols = 150
        self._print_symbol = '='
        
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        print(num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol)
    
    def scale_minmax(self, X, min=0, max=255):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def saveSpectrogram(self, data, fn):
        plt.axis('off')
        fig = plt.imshow(data, aspect='auto', origin='lower', interpolation='none')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fn, bbox_inches='tight', pad_inches=0)
        plt.close()

    def startEtractingSpectrograms(self):
        for mode in self._modes:
            self.printTitle(f'Spectrogram generation started for {mode} dataset')
            folder_name = 'cv-valid-' + mode
            self.out_dir = osp.join(self.spectrogram_dir, folder_name)
            os.makedirs(self.out_dir, exist_ok=True)
            audio_files_dir = osp.join(self.audio_dir, folder_name)
            audio_files = glob(audio_files_dir + "\\*.mp3")
            n = len(audio_files)
            print(f'Reading audio files from ==> {audio_files_dir}')
            print(f'Number of audio files ==> {n}')
            print(f'Saving extracted spectrograms to ==> {self.out_dir}\n')
            with alive_bar(n) as bar:
                for audio_file in audio_files:
                    f_name = osp.basename(audio_file).replace('.mp3', '.png')
                    y, sr = librosa.load(audio_file, sr=self.sampling_rate)
                    spec = librosa.stft(y, n_fft=self.n_fft, hop_length=self.num_overlap)
                    spec = librosa.amplitude_to_db(np.abs(spec))
                    # min-max scale to fit inside 8-bit range
                    img = self.scale_minmax(spec).astype(np.uint8)
                    out_file = osp.join(self.out_dir, f_name)
                    self.saveSpectrogram(img, out_file)
                    bar()


class TFRecordDatasetProcessor:
    def __init__(self, dataset_dir):
        self.tfrecord_dir = osp.join(dataset_dir, 'TFRecord')
        self._audio_dir = osp.join(dataset_dir, 'Audio')
        self._spec_dir = osp.join(dataset_dir, 'Spectrograms')
        os.makedirs(self.tfrecord_dir, exist_ok=True)
        self._modes = ['train', 'dev', 'test']
        self._tfrecord_size = 300
        self._spec_img_size = (64, 64)
        self._max_symbols = 150
        self._print_symbol = '='
        self._initializeLabelEncoderAndDecoder()
        self._feature_description = {
                'fpath': tf.io.FixedLenFeature([], tf.string),
                'img': tf.io.FixedLenFeature([], tf.string),
                'img_h': tf.io.FixedLenFeature([], tf.int64),
                'img_w': tf.io.FixedLenFeature([], tf.int64),
                'chan': tf.io.FixedLenFeature([], tf.int64),
                'lab_age': tf.io.FixedLenFeature([], tf.int64),
                'lab_gender': tf.io.FixedLenFeature([], tf.int64),
                'lab_gender_age': tf.io.FixedLenFeature([], tf.int64)}
        
    def printTitle(self, title):
        print('\n')
        text_length = len(title)
        num_symbols = (self._max_symbols - text_length)//2
        print(num_symbols * self._print_symbol + "> " + title + " <" + num_symbols * self._print_symbol)
        
    def _initializeLabelEncoderAndDecoder(self):
        self._encode_gender_age = {'female-teens': 0, 'female-twenties': 1, 'female-thirties': 2, 'female-fourties': 3, 'female-fifties': 4, 'female-sixties': 5,  
                                'male-teens': 6, 'male-twenties': 7, 'male-thirties': 8, 'male-fourties': 9, 'male-fifties': 10, 'male-sixties': 11}
        self._encode_gender = {'male': 0, 'female': 1}
        self._encode_age = {'teens': 0, 'twenties': 1, 'thirties': 2, 'fourties': 3, 'fifties': 4, 'sixties': 5}
        
        self._decode_gender_age = {0:'female-teens', 1:'female-twenties', 2:'female-thirties', 3:'female-fourties', 4:'female-fifties', 5:'female-sixties',  
                                6:'male-teens', 7:'male-twenties', 8:'male-thirties', 9:'male-fourties', 10:'male-fifties', 11:'male-sixties'}
        self._decode_gender = {0:'male', 1:'female'}
        self._decode_age = {0:'teens', 1:'twenties', 2:'thirties', 3:'fourties', 4:'fifties', 5:'sixties'}
    
    def getDecodedGender(self, encod_lab):
        return self._decode_gender.get(encod_lab)
    
    def getDecodedAge(self, encod_lab):
        return self._decode_age.get(encod_lab)
    
    def getDecodedGenderAge(self, encod_lab):
        return self._decode_gender_age.get(encod_lab)
    
    def getCategoryNames(self, cls_task):
        if cls_task=='age':
            return self._encode_age.keys()
        elif cls_task=='gender':
            return self._encode_gender.keys()
        elif cls_task=='gender_age':
            return self._encode_gender_age.keys()
    
    def getNumberOfClasses(self, category):
        if category=='age':
            return len(self._encode_age)
        elif category=='gender':
            return len(self._encode_gender)
        elif category=='gender_age':
            return len(self._encode_gender_age)
    
    def getInputImageSize(self):
        return self._spec_img_size
    
    @staticmethod
    def _bytesFeature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    @staticmethod
    def _floatFeature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    @staticmethod
    def _int64Feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    @classmethod
    def serializeExample(self, fpath, img, img_h, img_w, chan, lab_age, lab_gender, lab_gender_age):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
        feature = {
            'fpath': self._bytesFeature(fpath),
            'img': self._bytesFeature(img),
            'img_h': self._int64Feature(img_h),
            'img_w': self._int64Feature(img_w),
            'chan': self._int64Feature(chan),
            'lab_age': self._int64Feature(lab_age),
            'lab_gender': self._int64Feature(lab_gender),
            'lab_gender_age': self._int64Feature(lab_gender_age)}

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def startWritingTFRecordFiles(self):
        for mode in self._modes:
            self.printTitle(f'Writing TFRecord files started for {mode} dataset')
            folder_name = 'cv-valid-' + mode
            out_dir = osp.join(self.tfrecord_dir, folder_name)
            os.makedirs(out_dir, exist_ok=True)
            dataframe = pd.read_csv(osp.join(self._audio_dir, folder_name + '.csv'))
            sliced_dataframes = [dataframe[i:i+self._tfrecord_size] for i in range(0, len(dataframe), self._tfrecord_size)]
            n = len(sliced_dataframes)
            with alive_bar(n) as bar:
                for i, df in enumerate(sliced_dataframes):
                    out_file = osp.join(out_dir, f'spec_data_{i:03d}.tfrecord')
                    with tf.io.TFRecordWriter(out_file) as writer:
                        for row_idx, data in df.iterrows():
                            spec_filepath = osp.join(self._spec_dir, data['filename'].replace('.mp3', '.png'))
                            age = self._encode_age.get(data['age'])
                            gender = self._encode_gender.get(data['gender'])
                            gender_age = self._encode_gender_age.get(data['gender_age'])
                            img = img_to_array(load_img(spec_filepath, target_size=self._spec_img_size))
                            img_h = img.shape[0]
                            img_w = img.shape[1]
                            chan = img.shape[2]
                            img_bytes = tf.io.serialize_tensor(img)
                            fpath = bytes(spec_filepath, 'utf-8')
                            example = self.serializeExample(fpath, img_bytes, img_h, img_w, chan, age, gender, gender_age)
                            writer.write(example)
                    bar()
    
    def _decodeAge(self, serialized_example):
        example = tf.io.parse_single_example(serialized_example, self._feature_description)
        img = tf.io.parse_tensor(example['img'], out_type=tf.float32)
        img_h = example['img_h']
        img_w = example['img_w']
        chan = example['chan']
        img = tf.reshape(img, [img_h, img_w, chan])
        img = img/255.0
        label = example['lab_age']
        return (img, label)
    
    def _decodeGender(self, serialized_example):
        example = tf.io.parse_single_example(serialized_example, self._feature_description)
        img = tf.io.parse_tensor(example['img'], out_type=tf.float32)
        img_h = example['img_h']
        img_w = example['img_w']
        chan = example['chan']
        img = tf.reshape(img, [img_h, img_w, chan])
        img = img/255.0
        label = example['lab_gender']
        return (img, label)
    
    def _decodeGenderAge(self, serialized_example):
        example = tf.io.parse_single_example(serialized_example, self._feature_description)
        img = tf.io.parse_tensor(example['img'], out_type=tf.float32)
        img_h = example['img_h']
        img_w = example['img_w']
        chan = example['chan']
        img = tf.reshape(img, [img_h, img_w, chan])
        img = img/255.0
        label = example['lab_gender_age']
        return (img, label)
    
    def _isTaskCorrect(self, label: str) -> bool:
        correct_labels = ['age', 'gender', 'gender_age']
        if label in correct_labels:
            return True
        else:
            print("Given label argument should be one of ['age', 'gender', 'gender_age']")
            return False
    
    def _setLabelDecoder(self, dataset: tf.data.TFRecordDataset, label: str) -> tf.data.TFRecordDataset:
        if label=='age':
            return dataset.map(self._decodeAge, num_parallel_calls=tf.data.AUTOTUNE)
        elif label=='gender':
            return dataset.map(self._decodeGender, num_parallel_calls=tf.data.AUTOTUNE)
        elif label=='gender_age':
            return dataset.map(self._decodeGenderAge, num_parallel_calls=tf.data.AUTOTUNE)
    
    def getTrainTFRecordDataset(self, cls_task, batch_size):
        if self._isTaskCorrect(cls_task):
            train_tfrecord_dir = osp.join(self.tfrecord_dir, 'cv-valid-train')
            tfrecord_files = tf.data.Dataset.list_files(train_tfrecord_dir + "\\*.tfrecord")
            tr_dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
            tr_dataset = tr_dataset.shuffle(10000)
            tr_dataset = self._setLabelDecoder(tr_dataset, cls_task)
            tr_dataset = tr_dataset.batch(batch_size)
            tr_dataset = tr_dataset.prefetch(tf.data.AUTOTUNE)
            return tr_dataset
    
    def getValidationTFRecordDataset(self, cls_task, batch_size):
        if self._isTaskCorrect(cls_task):
            valid_tfrecord_dir = osp.join(self.tfrecord_dir, 'cv-valid-dev')
            tfrecord_files = tf.data.Dataset.list_files(valid_tfrecord_dir + "\\*.tfrecord")
            valid_dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
            valid_dataset = valid_dataset.shuffle(2000)
            valid_dataset = self._setLabelDecoder(valid_dataset, cls_task)
            valid_dataset = valid_dataset.batch(batch_size)
            valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
            return valid_dataset
    
    def getTestTFRecordDataset(self, cls_task, batch_size):
        if self._isTaskCorrect(cls_task):
            test_tfrecord_dir = osp.join(self.tfrecord_dir, 'cv-valid-test')
            tfrecord_files = tf.data.Dataset.list_files(test_tfrecord_dir + "\\*.tfrecord")
            test_dataset = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE)
            test_dataset = test_dataset.shuffle(2000)
            test_dataset = self._setLabelDecoder(test_dataset, cls_task)
            test_dataset = test_dataset.batch(batch_size)
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
            return test_dataset