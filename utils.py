import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import ddsp

import math
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

def adding_AWGN_noise(signal_file_path,SNR=20):
    '''
    Signal to noise ratio (SNR) can be defined as 
    SNR = 20*log(RMS_signal/RMS_noise)
    where RMS_signal is the RMS value of signal and RMS_noise is that of noise.
        log is the logarithm of 10
    *****additive white gausian noise (AWGN)****
    - This kind of noise can be added (arithmatic element-wise addition) to the signal
    - mean value is zero (randomly sampled from a gausian distribution with mean value of zero. standard daviation can varry)
    - contains all the frequency components in an equal manner (hence "white" noise) 
    '''

    #SNR in dB
    #given a signal and desired SNR, this gives the required AWGN what should be added to the signal to get the desired SNR
    def get_white_noise(signal,SNR) :
        #RMS value of signal
        RMS_s=math.sqrt(np.mean(signal**2))
        #RMS values of noise
        RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
        #Additive white gausian noise. Thereore mean=0
        #Because sample length is large (typically > 40000)
        #we can use the population formula for standard daviation.
        #because mean=0 STD=RMS
        STD_n=RMS_n
        noise=np.random.normal(0, STD_n, signal.shape[0])
        return noise

    #given a signal, noise (audio) and desired SNR, this gives the noise (scaled version of noise input) that gives the desired SNR
    def get_noise_from_sound(signal,noise,SNR):
        RMS_s=math.sqrt(np.mean(signal**2))
        #required RMS of noise
        RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
        
        #current RMS of noise
        RMS_n_current=math.sqrt(np.mean(noise**2))
        noise=noise*(RMS_n/RMS_n_current)
        
        return noise

    #***convert complex np array to polar arrays (2 apprays; abs and angle)
    def to_polar(complex_ar):
        return np.abs(complex_ar),np.angle(complex_ar)



    #**********************************
    #*************add AWGN noise******
    #**********************************
    signal, sr = librosa.load(signal_file_path)
    signal=np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    noise=get_white_noise(signal,SNR=20)
    #analyze the frequency components in the signal
    X=np.fft.rfft(noise)
    radius,angle=to_polar(X)
    # plt.plot(radius)
    # plt.xlabel("FFT coefficient")
    # plt.ylabel("Magnitude")
    # plt.show()
    # signal_noise=signal+noise
    # plt.plot(signal_noise)
    # plt.xlabel("Sample number")
    # plt.ylabel("Amplitude")
    # plt.show()

    #**********************************
    #*************add real world noise******
    #**********************************

    # signal, sr = librosa.load(signal_file)
    # signal=np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    # plt.plot(signal)
    # plt.xlabel("Sample number")
    # plt.ylabel("Signal amplitude")
    # plt.show()

    # noise_file='/home/sleek_eagle/research/emotion/code/audio_processing/noise.wav'
    # noise, sr = librosa.load(noise_file)
    # noise=np.interp(noise, (noise.min(), noise.max()), (-1, 1))


    #crop noise if its longer than signal
    #for this code len(noise) shold be greater than len(signal)
    #it will not work otherwise!
    if(len(noise)>len(signal)):
        noise=noise[0:len(signal)]

    noise=get_noise_from_sound(signal,noise,SNR)

    signal_noise=signal+noise


    print("SNR = " + str(20*np.log10(math.sqrt(np.mean(signal**2))/math.sqrt(np.mean(noise**2)))))

    plt.plot(signal_noise)
    plt.xlabel("Sample number")
    plt.ylabel("Amplitude")
    plt.show()

    return signal_noise



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