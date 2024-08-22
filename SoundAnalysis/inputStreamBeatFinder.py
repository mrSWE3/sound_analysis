from beatFinder import PyAudioBF, BeatFinder
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation




block_length = 0.1
SAMPLE_RATE = 44100


BUFFER_SECONDS = 1

num_chunks = int(BUFFER_SECONDS / block_length)
volume_buffer = np.zeros(num_chunks)  # Initialize the volume buffer

# Time axis for the volume plot
x = np.linspace(0, BUFFER_SECONDS, num_chunks)

# Create a figure and axis for plotting
fig, ax = plt.subplots()
line, = ax.plot(x, volume_buffer)

# Set the limits of the y-axis and labels
ax.set_ylim(0.5, 1)  # Volume (RMS) ranges between 0 and 1
ax.set_xlim(0, BUFFER_SECONDS)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Volume (RMS)')

def audio_callback(indata):
    
    global volume_buffer
    
    volume_buffer = np.roll(volume_buffer, -1)
    volume_buffer[-1] = indata

bf = BeatFinder(audio_callback,
                 SAMPLE_RATE,
                 lookback_window_length=3)

isbf = PyAudioBF(bf, block_length, SAMPLE_RATE)
isbf.start()


# Animation function for updating the plot
def update_plot(frame):
    line.set_ydata(volume_buffer)
    return line,

ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True)
plt.show()


