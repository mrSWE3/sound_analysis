import sounddevice as sd
from typing import Protocol, Any, Callable
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import spectrogram
from pydub import AudioSegment


class InputStream(Protocol):
    def start(self, callback: Callable[[NDArray], None],
              block_length: float,
              samplerate: int):
        ...



class PyaudioIS(InputStream):
    def start(self, callback: Callable[[NDArray], None],
              block_length: float,
              samplerate: int):
        self.inputStream = sd.InputStream(callback=(lambda block,a,b,c: callback(block[:,0])),
                                           channels=1,
                                           samplerate=samplerate,
                                           blocksize=int(block_length * samplerate))
        self.inputStream.start()


class Mp3InputStream(InputStream):
    def __init__(self, mp3_path) -> None:
        # Load MP3 file
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Extract raw audio data and sample rate
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        
        # Ensure that the audio data is in the right shape (1D array)
        if len(samples.shape) > 1:
            samples = samples[:, 0]

        self.samples = samples
        self.sample_rate = sample_rate

    def start(self, callback: Callable[[NDArray[np.int16]], None],
              block_length: float,
              samplerate: int = -1):
        samples = self.samples
        if samplerate > self.sample_rate:
            samples = resample_data(self.samples, self.sample_rate, samplerate)
        else:
            samplerate = self.sample_rate
        blocksize = int(samplerate * block_length)
        for index in range(0, samples.shape[0], blocksize):
            block = samples[index:min(index+blocksize, samples.shape[0])]
            callback(block)

def resample_data(data, original_rate, target_rate):
    # Calculate the resampling factor
    resample_factor = target_rate / original_rate
    
    # Calculate the number of samples in the resampled data
    num_samples_original = data.shape[0]
    num_samples_resampled = int(num_samples_original * resample_factor)
    
    # Create the time indices for the original and resampled data
    original_time_indices = np.arange(num_samples_original)
    resampled_time_indices = np.linspace(0, num_samples_original - 1, num_samples_resampled)
    
    # Interpolate the data to get the resampled data
    resampled_data = np.interp(resampled_time_indices, original_time_indices, data)
    
    return resampled_data


def specto_rate(N, sample_rate, nperseg, noverlap) -> float:
    step = nperseg - noverlap
    M = (N - noverlap) // step
    time_step = step / sample_rate
    time_span = (M - 1) * time_step
    if time_span == 0:
        raise Exception("Time span is 0")
    ratio = M / time_span
    return ratio


class BeatFinder:
    def __init__(self,
                beat_callback: Callable[[float], None],
                samplerate: int,  
                nperseg: int = 800,
                noverlap: int | None = None,
                base_frequency_cutoff: int = 300,
                mean_window_length: float = 0.1,
                lookback_window_length: float = 1) -> None:
        
        self.beat_callback = beat_callback
        if noverlap == None:
            noverlap = int(0.8 * nperseg)
        self.noverlap = noverlap
        self.samplerate = samplerate
        self.nperseg = nperseg
        self.base_frequency_cutoff = base_frequency_cutoff
        self.mean_window_length = mean_window_length
        self.lookback_window_size = lookback_window_length


        self.lookback_spectogram: NDArray = np.array([])
        
    def process_block(self, block: NDArray):
        spectrogram_rate = specto_rate(block.shape[0], self.samplerate, self.nperseg, self.noverlap)
        lookback_window_size = int(self.lookback_window_size * spectrogram_rate)
       
        frequencies, times, Sxx =  spectrogram(block, self.samplerate, nperseg=self.nperseg, noverlap=self.noverlap)
        volume = 10 * np.log10(Sxx)

        freq_mask = frequencies < self.base_frequency_cutoff
        frequencies = frequencies[freq_mask]
        volume = volume[freq_mask]

        y = np.sum(volume, axis=0) / volume.shape[0]
        smoothed_data = pd.Series(y).rolling(
            window=int(self.mean_window_length * spectrogram_rate),
            center=True, min_periods=0).mean()
        
        self.lookback_spectogram = np.concat((self.lookback_spectogram, smoothed_data))[-lookback_window_size:]
        
        m = smoothed_data.mean()
        prob = (self.lookback_spectogram < m).mean()
        self.beat_callback(prob)


class ISBF:
    def __init__(self,
                input_stream: InputStream,
                beatFidner: BeatFinder,
                block_length: float,
                samplerate: int) -> None:
        self.bf = beatFidner

        self.input_stream = input_stream
        self.block_length = block_length
        self.samplerate = samplerate


    def start(self):
        self.input_stream.start(self.bf.process_block, self.block_length, self.samplerate)

class Mp3BF(ISBF):
    def __init__(self, mp3_path: str, beat_callback: Callable[[float], None], block_length: float, nperseg: int = 800, noverlap: int | None = None, base_frequency_cutoff: int = 300, mean_window_length: float = 0.1, lookback_window_length: float = 1) -> None:

        mp3 = Mp3InputStream(mp3_path)
        bf = BeatFinder(beat_callback, mp3.sample_rate, nperseg, noverlap, base_frequency_cutoff, mean_window_length, lookback_window_length)
        super().__init__(mp3, bf, block_length, mp3.sample_rate)

class PyAudioBF(ISBF):
    def __init__(self, beatFidner: BeatFinder, block_length: float, samplerate: int) -> None:
        super().__init__(PyaudioIS(), beatFidner,block_length, samplerate)
