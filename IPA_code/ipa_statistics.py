import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ipa



if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pupil_data = np.load(os.path.join(current_dir, os.pardir, 'data', '0_data_12-01', 'Lpupil.npy'), allow_pickle=True)
    ipa_data, position, threshold = ipa.ipa_cal(pupil_data)
    
    plt.scatter(range(len(ipa_data)), ipa_data, c='red', alpha=0.5, s=2)
    plt.show()