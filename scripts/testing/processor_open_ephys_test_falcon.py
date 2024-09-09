import zmq
import flatbuffers
import numpy as np
import time
import matplotlib.pyplot as plt
from ContinuousData import ContinuousData

# Parameters
address = "127.0.0.1" # can be replaced by 'localhost'
port = 5556

# Step 1: Create your ZMQ socket
context = zmq.Context()
tcp_address = f"tcp://{address}:{port}"
socket = context.socket(zmq.SUB)
socket.setsockopt_string(zmq.SUBSCRIBE, "")
socket.connect(tcp_address)

# Collect data for plotting
channel_data = []
time.sleep(1) # second for warmup

# Step 2: Loop to receive packets
time_in = time.time()
while True:
    try:
        # Non-blocking wait to receive a message
        message = socket.recv(flags=zmq.NOBLOCK)
        
        # Step 3: Decode the message
        try:
            buf = bytearray(message)
            data = ContinuousData.GetRootAsContinuousData(buf, 0)
        except Exception as e:
            print(f"Impossible to parse the packet received - skipping to the next. Error: {e}")
            continue

        # Access fields based on the schema
        message_id = data.MessageId()
        stream = data.Stream().decode('utf-8')
        sample_num = data.SampleNum()
        num_samples = data.NSamples()
        num_channels = data.NChannels()
        timestamp = data.Timestamp()
        sample_rate = data.SampleRate()
        samples_flat = data.SamplesAsNumpy()

        # Print the received metadata
        print(f"Received packet number: {message_id}")
        print(f"Stream: {stream}")
        print(f"Sample Number: {sample_num}")
        print(f"Samples per Channel: {num_samples}")
        print(f"Number of Channels: {num_channels}")
        print(f"Timestamp: {timestamp}")
        print(f"Sample Rate: {sample_rate}")

        # Check if the total size matches
        total_elements = samples_flat.size
        expected_elements = num_samples * num_channels
        
        if total_elements != expected_elements:
            print(f"Error: Expected {expected_elements} elements but got {total_elements}.")
        else:
            # Reshape the flat samples array
            samples_reshaped = samples_flat.reshape((num_channels, num_samples))
            channel_data.extend(samples_reshaped[54, :])  # Collect data from channel

            # Plot the data if we've collected enough samples
            if len(channel_data) >= 1000000: # 33.33333... samples at 30kHz
                plt.figure(figsize=(10, 4))
                plt.plot(channel_data)
                plt.title(f"Channel 54 Data, with time lag: {time.time() - time_in - 33.33:.4f}s")
                plt.xlabel("Sample Number")
                plt.ylabel("Amplitude")
                plt.show()
                channel_data = []  # Clear the data after plotting

    except zmq.Again:
        # No message received
        pass