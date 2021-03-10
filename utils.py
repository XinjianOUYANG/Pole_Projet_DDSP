import numpy as np
import matplotlib.pyplot as plt
import ddsp

# def specplot(audio,sample_rate):
#     wlen_sec=32e-3
#     hop_percent=.5
#     wlen = int(wlen_sec*sample_rate) # window length of 64 ms
#     wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2 practically
#     nfft = wlen
#     hop = np.int(hop_percent*wlen) # hop size
#     win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

#     X = librosa.stft(audio, n_fft=nfft, hop_length=hop, win_length=wlen, window=win) # STFT

#     plt.figure(figsize=(10,7))
#     librosa.display.specshow(librosa.power_to_db(np.abs(X)**2), sr=sample_rate, hop_length=hop, x_axis='time', y_axis='hz')

#     plt.set_cmap('gray_r')
#     plt.colorbar()
#     plt.clim(vmin=-30)

#     plt.ylabel('frequency (Hz)')
#     plt.xlabel('time (s)')
#     plt.tight_layout()

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