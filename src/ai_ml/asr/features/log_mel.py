import librosa
import torch

SAMPLE_RATE = 16000


N_FFT = 400
WIN_LENGTH = 400
HOP_LENGTH = 160

N_MEL = 80

def extract_log_mel(audio: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(WIN_LENGTH)

    stft = torch.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=window, return_complex=True)

    power_spectrogram = stft.abs()**2

    mel_filter = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MEL, fmin=0, fmax=SAMPLE_RATE / 2)

    mel_filter = torch.tensor(mel_filter, dtype=torch.float32)

    mel_spectrogram = mel_filter @ power_spectrogram

    log_mel = torch.log(mel_spectrogram.clamp(min=1e-10))

    return log_mel.transpose(0,1)