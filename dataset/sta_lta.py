import obspy
from obspy.signal.trigger import ar_pick
from obspy import read

class Sta_Lta:

    def __init__(self, sta : float, lta : float, sampling_rate : float):
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
    
    def pick_trace(self, trace,
                   low_band_pass_freq,
                   upper_band_pass_freq,
                   num_ar_p, num_ar_s,
                   var_window_length_p, var_window_length_s, 
                   s_pick, verbose):

        p_arrival, s_arrival = ar_pick(
            trace.Z, trace.N, trace.E, 
            samp_rate=self.sampling_rate, 
            f1=low_band_pass_freq, 
            f2=upper_band_pass_freq, 
            lta_p=self.lta_window_sec_p,
            sta_p=self.sta_window_sec_p,
            lta_s=self.lta_window_sec_s,
            sta_s=self.sta_window_sec_s,
            m_p=num_ar_p,
            m_s=num_ar_s,
            l_p=var_window_length_p,
            l_s=var_window_length_s,
            s_pick=s_pick)  

        if verbose: 
            print(f"Detected P-wave at: {p_arrival}, and S-wave at : {s_arrival}")    
        return p_arrival, s_arrival
        


