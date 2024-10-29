
import numpy as np
import pandas as pd
import random
import torch, torchaudio
import librosa

def process_wav_file(file_path, spec_type):
    y, sr = librosa.load(file_path)
    y = y.astype(np.float32) / np.iinfo(np.int16).max # Normalize
    L = 71000

    if len(y) > L:
        i = np.random.randint(0, len(y) - L)
        y = y[i:(i+L)]  

    elif len(y) < L:
        rem_len = L - len(y)
        silence_part = np.random.randint(-100,100,L).astype(np.float32) / np.iinfo(np.int16).max
        j = np.random.randint(0, rem_len)
        silence_part_left  = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        y = np.concatenate([silence_part_left, y, silence_part_right])

    if spec_type == 'mel':
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, n_mels=128, hop_length=512)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        return np.expand_dims(spec_db, axis=2)

    if spec_type == 'stft':
        eps=1e-10
        threshold_freq = 5500

        freqs, times, spec = stft(y, L, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
        
        if threshold_freq is not None:
            spec = spec[freqs <= threshold_freq,:]
            freqs = freqs[freqs <= threshold_freq]

        amp = np.log(np.abs(spec)+eps)
    
        return np.expand_dims(amp, axis=2)

    if spec_type == 'mfcc':
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
        return np.expand_dims(mfccs, axis=2)

def xlsr53_load_audio(file_path, spec_type):
    from datasets import load_dataset, load_metric, Audio
    common_voice_train = load_dataset("common_voice", "tr", split="train+validation")
    common_voice_test = load_dataset("common_voice", "tr", split="test")
    common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    return common_voice_train, common_voice_test

def load_audio(filepath):
    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate} to 16000 Hz")
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform

class SERDataset(Dataset):
    def __init__(self, audio_files, labels, processor):
        self.audio_files = audio_files
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        waveform = load_audio(self.audio_files[idx])
        inputs = self.processor(waveform, return_tensors="pt", sampling_rate=16000).input_values
        label = self.labels[idx]
        return inputs.squeeze(0), torch.tensor(label, dtype=torch.long)


def pad_sequences(sequences, max_length):
    padded_sequences = torch.zeros(len(sequences), max_length, sequences[0].size(1))
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        padded_sequences[i, -length:] = seq
    return padded_sequences

def generate_masks(src, tgt):
    src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=tgt.device), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

def plot_data(data, features, data2=None, title='Random Sample'):
    '''
    data: DataFrame containing the data channels to plot
    features: List of strings containing the feature names to plot
    data2: Optional, generally for comparing manipulated data
    '''
    import matplotlib.pyplot as plt
    if len(data) == 0:
        print("No samples to plot.")
    X = data.iloc[:, 0]
    X2 = data2.iloc[:, 0] if data2 is not None else None

    fig, axs = plt.subplots(len(features), 1, figsize=(12, 15), sharex=True)
    for i, label in enumerate(features):
        axs[i].plot(X, data.iloc[:, i+1], label=label)
        if data2 is not None:
            axs[i].plot(X2, data2.iloc[:, i+1], label='(Data2)', linestyle='--', color='red', linewidth=1)
        axs[i].set_ylabel(label)
        axs[i].grid(True)
        axs[i].legend(loc='upper right')
    axs[-1].set_xlabel(features[0])
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return

def scale_data(train_data, val_data, train_target, val_target):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    # Combine the lists of DataFrames into single DataFrames
    train_data_combined = pd.concat(train_data, ignore_index=True)
    val_data_combined = pd.concat(val_data, ignore_index=True)
    train_target_combined = pd.concat(train_target, ignore_index=True)
    val_target_combined = pd.concat(val_target, ignore_index=True)

    # Fit and transform the combined data
    Xscaler = StandardScaler()
    Yscaler = StandardScaler()
    train_data_scaled = Xscaler.fit_transform(train_data_combined)
    val_data_scaled = Xscaler.transform(val_data_combined)
    train_target_scaled = Yscaler.fit_transform(train_target_combined)
    val_target_scaled = Yscaler.transform(val_target_combined)

    # Split the scaled data back into the original list of DataFrames
    train_data_scaled_list = []
    val_data_scaled_list = []
    train_target_scaled_list = []
    val_target_scaled_list = []

    train_start = 0
    val_start = 0
    train_target_start = 0
    val_target_start = 0

    for df in train_data:
        train_end = train_start + len(df)
        train_data_scaled_list.append(pd.DataFrame(train_data_scaled[train_start:train_end], columns=df.columns))
        train_start = train_end

    for df in val_data:
        val_end = val_start + len(df)
        val_data_scaled_list.append(pd.DataFrame(val_data_scaled[val_start:val_end], columns=df.columns))
        val_start = val_end

    for df in train_target:
        train_target_end = train_target_start + len(df)
        train_target_scaled_list.append(pd.DataFrame(train_target_scaled[train_target_start:train_target_end], columns=df.columns))
        train_target_start = train_target_end

    for df in val_target:
        val_target_end = val_target_start + len(df)
        val_target_scaled_list.append(pd.DataFrame(val_target_scaled[val_target_start:val_target_end], columns=df.columns))
        val_target_start = val_target_end

    return train_data_scaled_list, val_data_scaled_list, train_target_scaled_list, val_target_scaled_list, Xscaler, Yscaler

def inverse_transform(data, scaler):
    """
    Inverse transform the data using the provided scaler.

    Parameters:
    scaler: The scaler used for the original scaling (e.g., StandardScaler).
    data: A list of numpy arrays or DataFrames representing the batches to inverse transform,
          or a single numpy array or DataFrame.

    Returns:
    inverse_transformed_data: The inverse transformed data.
    """
    def handle_batch(batch):
        if isinstance(batch, pd.DataFrame):
            batch_values = batch.values
        else:
            batch_values = batch
        
        # Save mask of zero values
        zero_mask = (batch_values == 0)

        # Check if batch is 3D
        if batch_values.ndim == 3:
            batch_shape = batch_values.shape
            batch_values = batch_values.reshape(-1, batch_shape[-1])

            inverse_transformed_values = scaler.inverse_transform(batch_values)
            inverse_transformed_values = inverse_transformed_values.reshape(batch_shape)
        else:
            inverse_transformed_values = scaler.inverse_transform(batch_values)
        
        # Apply mask to set padded values back to zero
        inverse_transformed_values[zero_mask] = 0

        if isinstance(batch, pd.DataFrame):
            return pd.DataFrame(inverse_transformed_values, columns=batch.columns)
        else:
            return inverse_transformed_values

    # Check if data is a list (multiple batches)
    if isinstance(data, list):
        inverse_transformed_data = []
        for batch in data:
            inverse_transformed_data.append(handle_batch(batch))
        return inverse_transformed_data
    else:
        # Single sample case
        return handle_batch(data)

def NormalizeData(data, avg=None, stdev=None):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    
    normalized_data = np.copy(data)
    # Replace missing/nan values with column mean
    # mask = np.isnan(normalized_data)
    # for i in range(normalized_data.shape[1]):
    #     col_mask = mask[:, i]
    #     for j in range(len(col_mask)):
    #         if col_mask[j]:
    #             normalized_data[col_mask[j], i] = np.nanmean(normalized_data[:, i])
    
    if avg is None:
        avg = data.mean(axis=0)
    if stdev is None:
        stdev = data.std(axis=0)
    
    # Replace zero stdev with 1
    stdev = np.where(stdev == 0, 1, stdev)
    
    normalized_data = (data - avg) / stdev
    
    return normalized_data, avg, stdev

def DenormalizeData(data, avg, stdev):
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")
    
    denormalized_data = np.copy(data)
    
    denormalized_data = (data * stdev) + avg
    
    return denormalized_data

def getmedian(data):
    data_values = data.values if isinstance(data, pd.DataFrame) else data
    
    med_single = np.median(data_values, axis=0)
    med_single = np.where(med_single == 0, 1, med_single)
    return med_single

def SeriesFilter(data, filter_columns, time=None, filter_type='biquadratic', cutoff_freq=1, order=2):
    """
    Apply a specified filter to each dimension of the input data.
    
    Parameters:
        data (np.array): 2D array where the first column is time in hours and the subsequent columns are the signals to be filtered.
        filter_cols (list): List of column indices to filter.
        filter_type (str): Type of filter to apply ('biquadratic' or 'butter').
        cutoff_freq (float): Cutoff frequency in 1/days (e.g., 0.1 for a 10-day cutoff period).
        order (int): Order of the filter.
    
    Returns:
        np.array: The filtered signals.
    """
    from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi, lfilter
    from scipy.interpolate import interp1d

    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def biquad_lowpass(cutoff, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        sos = butter(order, normal_cutoff, btype='low', output='sos')
        return sos

    def apply_filter(data, filter_coefficients, filter_type='biquad'):
        if filter_type == 'biquad':
            sos = filter_coefficients
            zi = sosfilt_zi(sos)
            filtered_data, _ = sosfilt(sos, data, zi=zi*data[0])
            return filtered_data
        else:
            b, a = filter_coefficients
            filtered_data = lfilter(b, a, data)
            return filtered_data

    # Extract the time and signal columns
    if time is None:
        time = data[:, 0]
        signals = data[:, 1:]
    else:
        signals = data

    # Determine the sampling frequency (fs) in Hz (1/seconds)
    # Calculate the sampling rate based on the time increments in hours
    time_diff = np.diff(time)
    # if not np.allclose(time_diff, time_diff[0]):
    #     # Interpolate data to a uniform time grid
    #     print("Interpolating data to a uniform time grid.")
    #     time_uniform = np.linspace(time.min(), time.max(), len(data))
    #     interpolator = interp1d(time, signals, kind='linear', fill_value='extrapolate')
    #     signals = interpolator(time_uniform)
    #     time_diff = np.diff(time_uniform)
    #     fs = 1.0 / (time[1] - time[0])  # Uniform sampling frequency in 1/hours
    # else:
    fs = 1.0 / time_diff[0]

    # Convert cutoff frequency from 1/days to 1/hours
    cutoff_freq_per_hour = cutoff_freq / 24.0

    # Initialize the filtered signals array
    filtered_signals = signals.copy()

    if filter_type == 'biquadratic':
        sos = biquad_lowpass(cutoff_freq_per_hour, fs, order=order)
        for col in filter_columns:
            filtered_signals[:, col-1] = apply_filter(data[:, col], sos, filter_type='biquad')
    elif filter_type == 'butter':
        b, a = butter_lowpass(cutoff_freq_per_hour, fs, order=order)
        for col in filter_columns:
            filtered_signals[:, col-1] = apply_filter(data[:, col], (b, a), filter_type='butter')
    elif filter_type == 'filtfilt':
        b, a = butter_lowpass(cutoff_freq_per_hour, fs, order=order)
        for col in filter_columns:
            filtered_signals[:, col-1] = filtfilt(b, a, data[:, col])
        
    else:
        raise ValueError("Unsupported filter type: {}".format(filter_type))

    return np.column_stack((time, filtered_signals))
