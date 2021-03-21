from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")

import os
import glob
import ddsp.training
from matplotlib import pyplot as plt
import IPython.display as ipd
import gin
import numpy as np
import utils
import tensorflow as tf

sample_rate = 16000

DRIVE_DIR = './Pretrained_ Models_for_T2/training_solo'  
assert os.path.exists(DRIVE_DIR)

# create all directories leading up to the given directory that do not exist already. 
# If the given directory already exists, ignore the error.
DATA_DIR = os.path.join(DRIVE_DIR, 'data')
get_ipython().system('mkdir -p "$DATA_DIR"')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio')
AUDIO_FILEPATTERN = AUDIO_DIR + '/*'
get_ipython().system('mkdir -p "$AUDIO_DIR"')
# folder to save the model
SAVE_DIR = os.path.join(DRIVE_DIR, 'ddsp-solo-instrument')
get_ipython().system('mkdir -p "$SAVE_DIR"')

# Prepare Dataset
mp3_files = glob.glob(os.path.join(DRIVE_DIR, '*.mp3'))
wav_files = glob.glob(os.path.join(DRIVE_DIR, '*.wav'))
audio_files = mp3_files + wav_files

for fname in audio_files:
  target_name = os.path.join(AUDIO_DIR, 
                             os.path.basename(fname).replace(' ', '_'))
  print('Copying {} to {}'.format(fname, target_name))
  get_ipython().system('cp "$fname" "$target_name"')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
get_ipython().system('mkdir -p "$TRAIN_DIR"')
TRAIN_TFRECORD = TRAIN_DIR + '/train.tfrecord'

get_ipython().system('ddsp_prepare_tfrecord --input_audio_filepatterns="$AUDIO_FILEPATTERN "--output_tfrecord_path="$TRAIN_TFRECORD" --num_shards=10 --alsologtostderr')


TRAIN_TFRECORD_FILEPATTERN = TRAIN_DIR + '/train.tfrecord*'
#print(TRAIN_TFRECORD_FILEPATTERN)

data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
dataset = data_provider.get_dataset(shuffle=False)
PICKLE_FILE_PATH = os.path.join(SAVE_DIR, 'dataset_statistics.pkl')
utils.save_dataset_statistics(data_provider, PICKLE_FILE_PATH, batch_size=1)

# train a model
get_ipython().run_line_magic('reload_ext', 'tensorboard')
import tensorboard as tb
tb.notebook.start('--logdir "{}"'.format(SAVE_DIR))
get_ipython().system('ddsp_run   --mode=train   --alsologtostderr   --save_dir="$SAVE_DIR"   --gin_file=models/solo_instrument.gin   --gin_file=datasets/tfrecord.gin   --gin_param="FRecordProvider.file_pattern=./Pretrained_ Models_for_T2/training_solo/data/train/train.tfrecord*"  --gin_param="batch_size=16"   --gin_param="train_util.train.num_steps=10"   --gin_param="train_util.train.steps_per_save=2"   --gin_param="trainers.Trainer.checkpoints_to_keep=1"')


data_provider = ddsp.training.data.TFRecordProvider(TRAIN_TFRECORD_FILEPATTERN)
dataset = data_provider.get_batch(batch_size=1, shuffle=False)

try:
  batch = next(iter(dataset))
except OutOfRangeError:
  raise ValueError(
      'TFRecord contains no examples. Please try re-running the pipeline with '
      'different audio file(s).')

# Parse the gin config.
gin_file = os.path.join(SAVE_DIR, 'operative_config-0.gin')
gin.parse_config_file(gin_file)

# Load model
model = ddsp.training.models.Autoencoder()
model.restore(SAVE_DIR)

# Resynthesize audio.
outputs = model(batch, training=False)
audio_gen = model.get_audio_from_outputs(outputs)
audio = batch['audio']

print('Original Audio')
utils.specplot(audio)

print('Resynthesis')
utils.specplot(audio_gen)

CHECKPOINT_ZIP = 'my_solo_instrument.zip'
latest_checkpoint_fname = os.path.basename(tf.train.latest_checkpoint(SAVE_DIR))
get_ipython().system('cd "$SAVE_DIR" && zip $CHECKPOINT_ZIP $latest_checkpoint_fname* operative_config-0.gin dataset_statistics.pkl')
get_ipython().system('cp "$SAVE_DIR/$CHECKPOINT_ZIP" ./ #copy')


