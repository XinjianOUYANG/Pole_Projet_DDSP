import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import ddsp

import base64
import io
import os
import ddsp
import ddsp.training
import gin
import pickle
import numpy as np
from scipy import stats
from scipy.io import wavfile

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import librosa
import librosa.display
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds  


sample_rate = 16000
EFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE

_play_count = 0  # Used for ephemeral play().

# Alias these for backwards compatibility and ease.
specplot = ddsp.training.plotting.specplot
plot_impulse_responses = ddsp.training.plotting.plot_impulse_responses
transfer_function = ddsp.training.plotting.transfer_function


def specplot(audio,
            vmin=-5,
            vmax=1,
            rotate=True,
            size=512 + 256,
            **matshow_kwargs):
    """Plot the log magnitude spectrogram of audio."""
    # If batched, take first element.
    if len(audio.shape) == 2:
        audio = audio[0]

    logmag = ddsp.spectral_ops.compute_logmag(ddsp.core.tf_float32(audio), size=size)
    if rotate:
        logmag = np.rot90(logmag)
    # Plotting.
    plt.matshow(logmag,
                vmin=vmin,
                vmax=vmax,
                cmap=plt.cm.magma,
                aspect='auto',
                **matshow_kwargs)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Time')
    plt.ylabel('Frequency')

def model_loading(audio, audio_features, model_dir, training = False):
    # dataset_statistics.pkl in .model folder
    dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')

    # operative_config-0.gin in model folder
    gin_file = os.path.join(model_dir, 'operative_config-0.gin')

    # Load the dataset statistics.
    print(f'Loading dataset statistics from {dataset_stats_file}')
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print('Loading dataset statistics from pickle failed: {}.'.format(err))


    # Parse gin config,
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(model_dir, ckpt_name)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio.shape[0] / hop_size)
    n_samples = time_steps * hop_size

    # print("===Trained model===")
    # print("Time Steps", time_steps_train)
    # print("Samples", n_samples_train)
    # print("Hop Size", hop_size)
    # print("\n===Resynthesis===")
    # print("Time Steps", time_steps)
    # print("Samples", n_samples)
    # print('')

    gin_params = [
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
        'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors.
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)


    # Trim all input vectors to correct lengths 
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:n_samples]

    # Set up the model just to predict audio given new conditioning
    model = ddsp.training.models.Autoencoder()
    model.restore(ckpt)

    # Resynthesize audio.
    outputs = model(audio_features, training) # Run the forward pass, add losses, and create a dictionary of outputs.
    # print(outputs.keys())
    # dict_keys(['inputs', 'audio', 'f0_hz', 'f0_confidence', 'loundness_db', 
    #           'f0_condience', 'loudness_db', 'f0_scaled', 'ld_scaled', 'amps', 
    #           'harmonic_distribution', 'noise_magnitudes', 'harmonic', 'filtered_noise', 
    #           'add', 'reverb', 'out', 'audio_synth'])

    return outputs



# def play(array_of_floats,
#          sample_rate=DEFAULT_SAMPLE_RATE,
#          ephemeral=True,
#          autoplay=False):
#   """Creates an HTML5 audio widget to play a sound in Colab.
#   This function should only be called from a Colab notebook.
#   Args:
#     array_of_floats: A 1D or 2D array-like container of float sound samples.
#       Values outside of the range [-1, 1] will be clipped.
#     sample_rate: Sample rate in samples per second.
#     ephemeral: If set to True, the widget will be ephemeral, and disappear on
#       reload (and it won't be counted against realtime document size).
#     autoplay: If True, automatically start playing the sound when the widget is
#       rendered.
#   """
#   # If batched, take first element.
#   if len(array_of_floats.shape) == 2:
#     array_of_floats = array_of_floats[0]

#   normalizer = float(np.iinfo(np.int16).max)
#   array_of_ints = np.array(
#       np.asarray(array_of_floats) * normalizer, dtype=np.int16)
#   memfile = io.BytesIO()
#   wavfile.write(memfile, sample_rate, array_of_ints)
#   html = """<audio controls {autoplay}>
#               <source controls src="data:audio/wav;base64,{base64_wavfile}"
#               type="audio/wav" />
#               Your browser does not support the audio element.
#             </audio>"""
#   html = html.format(
#       autoplay='autoplay' if autoplay else '',
#       base64_wavfile=base64.b64encode(memfile.getvalue()).decode('ascii'))
#   memfile.close()
#   global _play_count
#   _play_count += 1
#   if ephemeral:
#     element = 'id_%s' % _play_count
#     display.display(display.HTML('<div id="%s"> </div>' % element))
#     js = output._js_builder  # pylint:disable=protected-access
#     js.Js('document', mode=js.EVAL).getElementById(element).innerHTML = html
#   else:
#     display.display(display.HTML(html))