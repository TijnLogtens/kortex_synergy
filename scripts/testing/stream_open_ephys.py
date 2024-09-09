import zmq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
host = '127.0.0.1'
port = 5555
num_channels = 128
samples_per_channel = 5

# Set up ZMQ
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(f"tcp://{host}:{port}")
socket.setsockopt_string(zmq.SUBSCRIBE, '')

# Initialize plot
fig, ax = plt.subplots()
lines = [ax.plot([], [], label=f'Channel {i}')[0] for i in range(10)]
ax.set_xlim(0, samples_per_channel*200)
ax.set_ylim(-1, 1)  # Adjust based on expected signal range
ax.legend(loc='upper right')

# Data storage
data = np.zeros((num_channels, samples_per_channel))

def update(frame):
    global data

    try:
        # Receive the message
        message = socket.recv(zmq.NOBLOCK)
        
        # Ensure the message length is correct
        element_size = np.dtype(np.float32).itemsize
        if len(message) % element_size != 0:
            print(f"Received message size {len(message)} is not a multiple of element size {element_size}")
            return lines
        
        # Unpack the data
        samples = np.frombuffer(message, dtype=np.float32)
        
        # Check if the length matches the expected size for 128 channels and 5 samples/channel
        if len(samples) == num_channels * samples_per_channel:
            # Reshape the data to (5 samples, 128 channels)
            reshaped_samples = samples.reshape((samples_per_channel, num_channels)).T
            
            # Update the data
            data = reshaped_samples
        
            # Update the plot
            for i, line in enumerate(lines):
                line.set_ydata(data[i])
                line.set_xdata(np.arange(samples_per_channel))
    
    except zmq.Again:
        pass  # No message received

    return lines

# Animation
ani = FuncAnimation(fig, update, blit=True, interval=100)  # Update every 100 ms

plt.show()