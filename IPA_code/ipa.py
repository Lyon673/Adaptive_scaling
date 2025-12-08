import math
import pywt
import numpy as np

class Pupil:
    def __init__(self, data, timestamp):
        self.data = data
        self.timestamp = timestamp
    
    def get_timestamp(self):
        return self.timestamp

    def get_data(self):
        return self.data

def ipa_cal(d,threshold=None, flag=True, timestamp=None):
    """
    Calculate the Inter-Pupillary Amplitude (IPA) of a pupil diameter signal.
    
    Parameters:
        d (list): A list of signal data with timestamp information.
    
    Returns:
        float: The calculated IPA value.


    flag : False for using the threshold calculated from the total preceding data
           True for using the threshold calculated from the recent data

    timestamp : None means calculating the whole-time ipa
                else means calculating the realtime ipa
        
    """
    pupil_data = np.array([i.get_data() for i in d])

    # calculate the threshold based on the total preceding data
    cA2, cD2, cD1 = pywt.wavedec(pupil_data[:timestamp], 'sym16', 'per', level=2)
    cD2[:] = [x / math.sqrt(40) for x in cD2]
    cD2m= modmax(cD2)
    threshold = np.std(cD2m) * math.sqrt(2.0 * np.log2(len(cD2m)))*0.7

    if timestamp is not None:
        pupil_data = pupil_data[timestamp-64:timestamp+64]

    try:
        # Obtain 2-level Discrete Wavelet Transform (DWT) of the pupil diameter signal
        cA2, cD2, cD1 = pywt.wavedec(pupil_data, 'sym16', 'per', level=2)
    except ValueError:
        return None
    
    # Get signal duration (in seconds)
    tt = d[-1].get_timestamp() - d[0].get_timestamp()
    
    # Normalize by 1/2 for 2-level DWT
    cA2[:] = [x / math.sqrt(4.0) for x in cA2]
    cD1[:] = [x / math.sqrt(20) for x in cD1]
    cD2[:] = [x / math.sqrt(40) for x in cD2]
    #print(f"<LYON> len(cA2): {len(cA2)}, len(cD1): {len(cD1)}, len(cD2): {len(cD2)}")
    #print(f"<LYON> cD2: {cD2}")
    


    # Detect modulus maxima
    #cD2m= modmax(cD2[-128:])
    cD2m= modmax(cD2)
    
    # Threshold using universal threshold λuniv = σˆ * sqrt(2 * log2(n))
    if flag:
        threshold = np.std(cD2m) * math.sqrt(2.0 * np.log2(len(cD2m)))*0.7
    cD2t = pywt.threshold(cD2m, threshold, mode="hard")
    print(f"<LYON> threshold: {threshold}")
    
    position = []
    # Compute IPA
    ctr = 0
    for i in range(len(cD2t)-1):
        if math.fabs(cD2t[i]) > 0:
            ctr += 1
            position.append(4*i+2)
    
    IPA = float(ctr) / tt
    #print(f"<LYON> ctr: {ctr}, tt: {tt}, IPA: {IPA}")
    return IPA, position, threshold



def modmax(d):
    """
    Compute the modulus maxima (local maxima) of a signal.
    
    Parameters:
        d (list): A list of signal data.
    
    Returns:
        list: A list where local maxima are marked with the magnitude, others are 0.
    """
    # Compute signal modulus (absolute value of the signal)
    m = [math.fabs(x) for x in d]
    
    # Initialize the result list with 0.0
    t = [0.0] * len(d)
    

    for i in range(len(d)):
        # Get the neighboring values, ensuring correct boundaries
        ll = m[i - 1] if i >= 1 else m[i] + 1  # Left neighbor
        oo = m[i]  # Current value
        rr = m[i + 1] if i < len(d) - 1 else m[i]  # Right neighbor
        
        # Check if the current value is larger than both neighbors and strictly larger than either
        if (ll <= oo and oo >= rr) and (ll < oo or oo > rr):
            t[i] = math.fabs(d[i])  # Assign the magnitude at the local maximum
            
        else:
            t[i] = 0.0  # Otherwise set it to 0
    
    return t
