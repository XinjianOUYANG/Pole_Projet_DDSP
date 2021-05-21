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
from scipy import stats
from scipy.io import wavfile

import matplotlib.pyplot as plt
import scipy as sp
import librosa
import librosa.display
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds  

# 'argon2:$argon2id$v=19$m=10240,t=10,p=8$jstZWD53O9lFFYm490tyFg$pnwYfl+KN8UTteed8hIGOQ'
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

# ------------------------------------------------------------------------------
# Loudness Normalization
# ------------------------------------------------------------------------------
def smooth(x, filter_size=3):
  """Smooth 1-d signal with a box filter."""
  x = tf.convert_to_tensor(x, tf.float32)
  is_2d = len(x.shape) == 2
  x = x[:, :, tf.newaxis] if is_2d else x[tf.newaxis, :, tf.newaxis]
  w = tf.ones([filter_size])[:, tf.newaxis, tf.newaxis] / float(filter_size)
  y = tf.nn.conv1d(x, w, stride=1, padding='SAME')
  y = y[:, :, 0] if is_2d else y[0, :, 0]
  return y.numpy()

def detect_notes(loudness_db,
                 f0_confidence,
                 note_threshold=1.0,
                 exponent=2.0,
                 smoothing=40,
                 f0_confidence_threshold=0.7,
                 min_db=-120.):
  """Detect note on-off using loudness and smoothed f0_confidence."""
  mean_db = np.mean(loudness_db)
  db = smooth(f0_confidence**exponent, smoothing) * (loudness_db - min_db)
  db_threshold = (mean_db - min_db) * f0_confidence_threshold**exponent
  note_on_ratio = db / db_threshold
  mask_on = note_on_ratio >= note_threshold
  return mask_on, note_on_ratio

class QuantileTransformer:
  """Transform features using quantiles information.
  Stripped down version of sklearn.preprocessing.QuantileTransformer.
  https://github.com/scikit-learn/scikit-learn/blob/
  863e58fcd5ce960b4af60362b44d4f33f08c0f97/sklearn/preprocessing/_data.py
  Putting directly in ddsp library to avoid dependency on sklearn that breaks
  when pickling and unpickling from different versions of sklearn.
  """

  def __init__(self,
               n_quantiles=1000,
               output_distribution='uniform',
               subsample=int(1e5)):
    """Constructor.
    Args:
      n_quantiles: int, default=1000 or n_samples Number of quantiles to be
        computed. It corresponds to the number of landmarks used to discretize
        the cumulative distribution function. If n_quantiles is larger than the
        number of samples, n_quantiles is set to the number of samples as a
        larger number of quantiles does not give a better approximation of the
        cumulative distribution function estimator.
      output_distribution: {'uniform', 'normal'}, default='uniform' Marginal
        distribution for the transformed data. The choices are 'uniform'
        (default) or 'normal'.
      subsample: int, default=1e5 Maximum number of samples used to estimate
        the quantiles for computational efficiency. Note that the subsampling
        procedure may differ for value-identical sparse and dense matrices.
    """
    self.n_quantiles = n_quantiles
    self.output_distribution = output_distribution
    self.subsample = subsample
    self.random_state = np.random.mtrand._rand

  def _dense_fit(self, x, random_state):
    """Compute percentiles for dense matrices.
    Args:
      x: ndarray of shape (n_samples, n_features)
        The data used to scale along the features axis.
      random_state: Numpy random number generator.
    """
    n_samples, _ = x.shape
    references = self.references_ * 100

    self.quantiles_ = []
    for col in x.T:
      if self.subsample < n_samples:
        subsample_idx = random_state.choice(
            n_samples, size=self.subsample, replace=False)
        col = col.take(subsample_idx, mode='clip')
      self.quantiles_.append(np.nanpercentile(col, references))
    self.quantiles_ = np.transpose(self.quantiles_)
    # Due to floating-point precision error in `np.nanpercentile`,
    # make sure that quantiles are monotonically increasing.
    # Upstream issue in numpy:
    # https://github.com/numpy/numpy/issues/14685
    self.quantiles_ = np.maximum.accumulate(self.quantiles_)

  def fit(self, x):
    """Compute the quantiles used for transforming.
    Parameters
    ----------
    Args:
      x: {array-like, sparse matrix} of shape (n_samples, n_features)
        The data used to scale along the features axis. If a sparse
        matrix is provided, it will be converted into a sparse
        ``csc_matrix``. Additionally, the sparse matrix needs to be
        nonnegative if `ignore_implicit_zeros` is False.
    Returns:
      self: object
         Fitted transformer.
    """
    if self.n_quantiles <= 0:
      raise ValueError("Invalid value for 'n_quantiles': %d. "
                       'The number of quantiles must be at least one.' %
                       self.n_quantiles)
    n_samples = x.shape[0]
    self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

    # Create the quantiles of reference
    self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
    self._dense_fit(x, self.random_state)
    return self

  def _transform_col(self, x_col, quantiles, inverse):
    """Private function to transform a single feature."""
    output_distribution = self.output_distribution
    bounds_threshold = 1e-7

    if not inverse:
      lower_bound_x = quantiles[0]
      upper_bound_x = quantiles[-1]
      lower_bound_y = 0
      upper_bound_y = 1
    else:
      lower_bound_x = 0
      upper_bound_x = 1
      lower_bound_y = quantiles[0]
      upper_bound_y = quantiles[-1]
      # for inverse transform, match a uniform distribution
      with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
        if output_distribution == 'normal':
          x_col = stats.norm.cdf(x_col)
        # else output distribution is already a uniform distribution

    # find index for lower and higher bounds
    with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
      if output_distribution == 'normal':
        lower_bounds_idx = (x_col - bounds_threshold < lower_bound_x)
        upper_bounds_idx = (x_col + bounds_threshold > upper_bound_x)
      if output_distribution == 'uniform':
        lower_bounds_idx = (x_col == lower_bound_x)
        upper_bounds_idx = (x_col == upper_bound_x)

    isfinite_mask = ~np.isnan(x_col)
    x_col_finite = x_col[isfinite_mask]
    if not inverse:
      # Interpolate in one direction and in the other and take the
      # mean. This is in case of repeated values in the features
      # and hence repeated quantiles
      #
      # If we don't do this, only one extreme of the duplicated is
      # used (the upper when we do ascending, and the
      # lower for descending). We take the mean of these two
      x_col[isfinite_mask] = .5 * (
          np.interp(x_col_finite, quantiles, self.references_) -
          np.interp(-x_col_finite, -quantiles[::-1], -self.references_[::-1]))
    else:
      x_col[isfinite_mask] = np.interp(x_col_finite, self.references_,
                                       quantiles)

    x_col[upper_bounds_idx] = upper_bound_y
    x_col[lower_bounds_idx] = lower_bound_y
    # for forward transform, match the output distribution
    if not inverse:
      with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
        if output_distribution == 'normal':
          x_col = stats.norm.ppf(x_col)
          # find the value to clip the data to avoid mapping to
          # infinity. Clip such that the inverse transform will be
          # consistent
          clip_min = stats.norm.ppf(bounds_threshold - np.spacing(1))
          clip_max = stats.norm.ppf(1 - (bounds_threshold - np.spacing(1)))
          x_col = np.clip(x_col, clip_min, clip_max)
        # else output distribution is uniform and the ppf is the
        # identity function so we let x_col unchanged

    return x_col

  def _transform(self, x, inverse=False):
    """Forward and inverse transform.
    Args:
      x : ndarray of shape (n_samples, n_features)
        The data used to scale along the features axis.
      inverse : bool, default=False
        If False, apply forward transform. If True, apply
        inverse transform.
    Returns:
      x : ndarray of shape (n_samples, n_features)
        Projected data
    """
    x = np.array(x)  # Explicit copy.
    for feature_idx in range(x.shape[1]):
      x[:, feature_idx] = self._transform_col(
          x[:, feature_idx], self.quantiles_[:, feature_idx], inverse)
    return x

  def transform(self, x):
    """Feature-wise transformation of the data."""
    return self._transform(x, inverse=False)

  def inverse_transform(self, x):
    """Back-projection to the original space."""
    return self._transform(x, inverse=True)

  def fit_transform(self, x):
    """Fit and transform."""
    return self.fit(x).transform(x)


def fit_quantile_transform(loudness_db, mask_on, inv_quantile=None):
    """Fits quantile normalization, given a note_on mask.
    Optionally, performs the inverse transformation given a pre-fitted transform.
    Args:
        loudness_db: Decibels, shape [batch, time]
        mask_on: A binary mask for when a note is present, shape [batch, time].
        inv_quantile: Optional pretrained QuantileTransformer to perform the inverse
        transformation.
    Returns:
        Trained quantile transform. Also returns the renormalized loudnesses if
        inv_quantile is provided.
    """
    quantile_transform = QuantileTransformer()
    loudness_flat = np.ravel(loudness_db[mask_on])[:, np.newaxis]
    loudness_flat_q = quantile_transform.fit_transform(loudness_flat)

    if inv_quantile is None:
        return quantile_transform
    else:
        loudness_flat_norm = inv_quantile.inverse_transform(loudness_flat_q)
        loudness_norm = np.ravel(loudness_db.copy())[:, np.newaxis]
        loudness_norm[mask_on] = loudness_flat_norm
        return quantile_transform, loudness_norm


def save_dataset_statistics(data_provider, file_path, batch_size=1):
  """Calculate dataset stats and save in a pickle file."""
  print('Calculating dataset statistics for', data_provider)
  data_iter = iter(data_provider.get_batch(batch_size, repeats=1))

  # Unpack dataset.
  i = 0
  loudness = []
  f0 = []
  f0_conf = []
  audio = []

  for batch in data_iter:
    loudness.append(batch['loudness_db'])
    f0.append(batch['f0_hz'])
    f0_conf.append(batch['f0_confidence'])
    audio.append(batch['audio'])
    i += 1

  print(f'Computing statistics for {i * batch_size} examples.')

  loudness = np.vstack(loudness)
  f0 = np.vstack(f0)
  f0_conf = np.vstack(f0_conf)
  audio = np.vstack(audio)

  # Fit the transform.
  trim_end = 20
  f0_trimmed = f0[:, :-trim_end]
  l_trimmed = loudness[:, :-trim_end]
  f0_conf_trimmed = f0_conf[:, :-trim_end]
  mask_on, _ = detect_notes(l_trimmed, f0_conf_trimmed)
  quantile_transform = fit_quantile_transform(l_trimmed, mask_on)

  # Average values.
  mean_pitch = np.mean(ddsp.core.hz_to_midi(f0_trimmed[mask_on]))
  mean_loudness = np.mean(l_trimmed)
  mean_max_loudness = np.mean(np.max(l_trimmed, axis=0))

  # Object to pickle all the statistics together.
  ds = {'mean_pitch': mean_pitch,
        'mean_loudness': mean_loudness,
        'mean_max_loudness': mean_max_loudness,
        'quantile_transform': quantile_transform}

  # Save.
  with tf.io.gfile.GFile(file_path, 'wb') as f:
    pickle.dump(ds, f)
  print(f'Done! Saved to: {file_path}')

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