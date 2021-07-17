import librosa as lr
import numpy as np
import soundfile as sf


def to_spectral(sig, res):
    spec_complex = np.transpose(lr.stft(sig, n_fft=res * 2))[:, :-1]
    spec = np.zeros(shape=[spec_complex.shape[0], spec_complex.shape[1], 2])
    spec[:, :, 0] = np.abs(spec_complex)
    spec[:, :, 1] = np.angle(spec_complex)
    return spec


def to_signal(spec):
    a = spec[:, :, 0] * np.cos(spec[:, :, 1])
    b = spec[:, :, 0] * np.sin(spec[:, :, 1])
    spec_complex = a + 1j * b
    spec_complex = np.append(spec_complex, np.zeros(shape=[spec.shape[0], 1]), axis=-1)
    return lr.istft(np.transpose(spec_complex))


def slice_spectral(path, res):

    sig, sr = lr.load(path=path)
    total_duration = sig.shape[0] / sr

    print('Loaded ' + path + ': duration ' + str(total_duration) + 's')

    spec = to_spectral(sig, res)
    print(spec.shape)

    n_slices = int(spec.shape[0] / res * 2)

    x = np.zeros(shape=[n_slices, res, res, 2])
    for i in range(n_slices):
        start = np.random.randint(0, spec.shape[0] - res)
        x[i, :, :] = spec[start:start + res]
    
    return x
