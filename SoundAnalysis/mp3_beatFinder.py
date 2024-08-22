from beatFinder import Mp3BF
import numpy as np

import matplotlib.pyplot as plt

peaks = []

def call_back(prob: float):
    global peaks
    peaks.append(prob)


block_length = 0.05
bf = Mp3BF("mellow-future-bass-bounce-on-it-184234.mp3",call_back, block_length)

bf.start()

print(peaks[0].dtype)
plt.plot( list(np.arange(0,len(peaks) * block_length, block_length)), peaks)
plt.xlim(40,49)
plt.show()
