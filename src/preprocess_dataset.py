from data_utils import CommonVoiceDatasetPreprocessor, SpectrogramGenerator, TFRecordDatasetProcessor
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', default='data/CommonVoice', help='Dataset directory')
    parser.add_argument('--dataset_type', default='clean', choices=['raw', 'clean'], help="Dataset type. It can be `raw` or `clean`")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.dataset_type == 'raw':
        dataset_preprocessor = CommonVoiceDatasetPreprocessor(args.dataset_dir)
        dataset_preprocessor.startPreprocessing()
    
    # generating spectrograms from audio files
    spectrogram_generator = SpectrogramGenerator(args.dataset_dir)
    spectrogram_generator.startEtractingSpectrograms()
    
    # Reading generated spectrograms and writing them into tfrecord files  
    tfrecord_processor = TFRecordDatasetProcessor(args.dataset_dir)
    tfrecord_processor.startWritingTFRecordFiles()
