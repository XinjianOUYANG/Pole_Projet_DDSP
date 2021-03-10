import numpy as np
import matplotlib.pyplot as plt
import ddsp

import base64
import io
import pickle

import ddsp
import ddsp.training
from IPython import display
import note_seq
import numpy as np
from scipy import stats
from scipy.io import wavfile
import tensorflow.compat.v2 as tf

DEFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE

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