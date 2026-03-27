import obspy
from obspy.signal.trigger import ar_pick
from obspy import read

import seisbench
import os
import seisbench.data as sbd
from types import SimpleNamespace

import numpy as np

class Sta_Lta:
  def __init__(self, sta : float, lta : float, sampling_rate : float,
            low_band_pass_freq=1.0,
            upper_band_pass_freq=45.0,
            num_ar_p=2,
            num_ar_s=2,
            var_window_length_p=0.2,
            var_window_length_s=0.2,):
    """
    Parameters:
        sta: short term avergae window size (seconds)
        lta: longterm averge window size (seconds)
        sampling_rate: trace sampling rate (Hz)
    """
    self.sta_window_sec_p = sta
    self.sta_window_sec_s = sta
    self.lta_window_sec_p = lta
    self.lta_window_sec_s = lta
    self.sampling_rate = sampling_rate
    self.low_band_pass_freq = low_band_pass_freq
    self.upper_band_pass_freq = upper_band_pass_freq
    self.num_ar_p = num_ar_p
    self.num_ar_s = num_ar_s
    self.var_window_length_p = var_window_length_p
    self.var_window_length_s = var_window_length_s

    
  def pick_trace(self, trace, metadata, s_pick, verbose):

    order = metadata["trace_component_order"]
    z_idx, n_idx, e_idx = order.index("Z"), order.index("N"), order.index("E") 
    trace_z = trace[z_idx]
    trace_n = trace[n_idx]
    trace_e = trace[e_idx]

    p_arrival, s_arrival = ar_pick(
      trace_z, trace_n, trace_e, 
      samp_rate=self.sampling_rate, 
      f1=self.low_band_pass_freq, 
      f2=self.upper_band_pass_freq, 
      lta_p=self.lta_window_sec_p,
      sta_p=self.sta_window_sec_p,
      lta_s=self.lta_window_sec_s,
      sta_s=self.sta_window_sec_s,
      m_p=self.num_ar_p,
      m_s=self.num_ar_s,
      l_p=self.var_window_length_p,
      l_s=self.var_window_length_s,
      s_pick=s_pick)
    
    if verbose: 
        print(f"Detected P-wave at: {p_arrival}, and S-wave at : {s_arrival}")    
    return p_arrival, s_arrival

def test_picker():
    data = sbd.DummyDataset()
    print(data)

    print("Cache root:", seisbench.cache_root)
    print("Contents:", os.listdir(seisbench.cache_root))
    print("datasets:", os.listdir(seisbench.cache_root / "datasets"))
    print("dummydataset:", os.listdir(seisbench.cache_root / "datasets" / "dummydataset"))

    
    sample_idx = 3
    waveforms = data.get_waveforms(sample_idx)
    print("waveforms.shape:", waveforms.shape)
    metadata = data.metadata.iloc[sample_idx]
    sampling_rate = metadata['trace_sampling_rate_hz']
    
    print(f"Testing on trace {sample_idx} | Sampling Rate: {sampling_rate}Hz")

    picker = Sta_Lta(sta=0.2, lta=2.0, sampling_rate=sampling_rate,
            low_band_pass_freq=1.0,
            upper_band_pass_freq=45.0,
            num_ar_p=2,
            num_ar_s=2,
            var_window_length_p=0.2,
            var_window_length_s=0.2)

    try:
        p_pick, s_pick = picker.pick_trace(trace=waveforms, metadata=metadata, s_pick=True, verbose=True)
        
        print(f"\nResults:")
        print(f"P-arrival (seconds from start): {p_pick:.3f}")
        print(f"S-arrival (seconds from start): {s_pick:.3f}")
        #print(f"Actual P-arrival (seconds fomr start): {}")
        #print(f"Actual S-arrival (seconds fomr start): {}")
        
    except Exception as e:
        print(f"Error during picking: {e}")

if __name__ == "__main__":
   test_picker()
        


