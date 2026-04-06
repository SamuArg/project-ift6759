import h5py
import numpy as np
import seisbench.data as sbd
from tqdm import tqdm
from scipy.signal import decimate
import pywt

def cwt_transform(traces, sf, down_sample_factor, cwt_widths=None):
    """
    Converts 3-component seismic traces to 3-channel time-frequency representations.
    
    Args:
        traces: numpy array of shape (3, num_samples)
        sf: sampling frequency in Hz
        cwt_widths: Array of wavelet scales to use
        
    Returns:
        amplitude_spectrogram: numpy array of shape (3, freq_bins, time_steps)
        log_spectrogram: log-scaled amplitude spectrogram
        (frequencies, times): Tuples of frequency and time axes
    """

    traces = decimate(traces, q=down_sample_factor, axis=-1)
    sf = sf / down_sample_factor
    num_channels, num_samples = traces.shape
        
    # 'mexh' is the PyWavelets identifier for the Mexican Hat (Ricker) wavelet
    wavelet = 'cmor1.5-1.0'
    # Default widths (scales) if none are provided. 
    # In CWT, scale is inversely proportional to frequency.
    if cwt_widths is None:
        cwt_widths = np.arange(1, 129) 
        
    cwt_channels = []
    frequencies = None
    
    
    for i in range(num_channels):
        # cwt calculates both the coefficients and the proper frequencies.
        # Passing sampling_period=1/sf ensures frequencies are returned in Hz.
        cwt_mat, freqs = pywt.cwt(traces[i], scales=cwt_widths, wavelet=wavelet, sampling_period=1/sf)
        cwt_channels.append(cwt_mat)
        
        # We only need to grab the frequency array once since it's identical for all 3 channels
        if frequencies is None:
            frequencies = freqs
            
    # Stack back into shape: (3, len(cwt_widths), num_samples)
    amplitude_spectrogram = np.abs(np.stack(cwt_channels))
    
    # For CWT, the time resolution perfectly matches the input trace length
    times = np.arange(num_samples) / sf

    # Log scaling to compress the dynamic range
    log_spectrogram = np.log1p(amplitude_spectrogram) 
    
    return amplitude_spectrogram, log_spectrogram, (frequencies, times)

def build_cwt_dataset(dataset_name="STEAD", output_path="stead_cwt.h5"):
    # 1. Load the raw dataset
    if dataset_name == "STEAD":
        dataset = sbd.STEAD(component_order="ZNE")
    elif dataset_name == "INSTANCE":
        dataset = sbd.InstanceCounts(component_order="ZNE")

    # We iterate over the metadata to track trace names/IDs
    metadata = dataset.metadata
    sampling_rate = getattr(dataset, 'sampling_rate', 100.0)

    # --- Config for Optimizations ---
    down_sample_factor = 2
    target_sf = sampling_rate / down_sample_factor
    
    # Calculate exactly 32 scales mapping to the 1 Hz -> 45 Hz range logarithmically
    # We use a geometric space because wavelets require logarithmic scale spacing
    wavelet = 'cmor1.5-1.0'
    desired_freqs = np.geomspace(1, 45, num=32)

    # pywt formula to convert Hz to CWT scales for a given wavelet and sampling period
    cwt_widths = pywt.central_frequency(wavelet) / (desired_freqs * (1 / target_sf))

    with h5py.File(output_path, 'w') as h5f:
        # Create a dataset to hold the CWTs. 
        # Shape: (Num_traces, Channels, Freqs, Time)
        # Using float16 can save massive amounts of disk space with negligible loss in accuracy
        cwt_dset = h5f.create_dataset(
            "spectrograms", 
            shape=(len(metadata), 3, 64, 3000), 
            dtype=np.float16,
            compression="lzf"
        )
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            # 2. Get raw waveform
            raw_trace = dataset.get_waveforms(row['trace_name'])
            
            # 3. Apply CWT (assuming your function now decimates to 50 Hz and pools frequencies)
            amp_spec, log_spec, _ = cwt_transform(raw_trace, sampling_rate, down_sample_factor=down_sample_factor, cwt_widths=cwt_widths)
            
            # 4. Save to HDF5
            cwt_dset[idx] = log_spec.astype(np.float16)

    # Save the metadata CSV so you can easily link index to trace later
    metadata.to_csv(output_path.replace(".h5", "_metadata.csv"))
    print("Done!")