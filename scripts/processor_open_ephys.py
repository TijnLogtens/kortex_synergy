import zmq
import numpy as np
import json
import uuid
import time
import os
import pickle
import rospy
import xgboost as xgb
import flatbuffers
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.ContinuousData import ContinuousData
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from scipy.signal import iirnotch, filtfilt, resample
from kortex_synergy.msg import MultiLabelActionNotification

class RollingBuffer:
    def __init__(self, num_channels, sampling_rate, buffer_duration):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.buffer_duration = buffer_duration
        
        # Calculate the total number of samples for the buffer
        self.total_samples = int(sampling_rate * buffer_duration)
        self.buffer = np.zeros((num_channels, self.total_samples))
        
        # Initialize pointers
        self.write_index = 0
        self.lock = Lock()
        
    def add_data(self, data):
        with self.lock:
            num_samples = data.shape[1]
            
            end_index = self.write_index + num_samples
            if end_index < self.total_samples:
                self.buffer[:, self.write_index:end_index] = data
            else:
                first_part = self.total_samples - self.write_index
                second_part = num_samples - first_part
                self.buffer[:, self.write_index:] = data[:, :first_part]
                self.buffer[:, :second_part] = data[:, first_part:]
            
            self.write_index = (self.write_index + num_samples) % self.total_samples

    def get_buffer(self):
        with self.lock:
            return np.copy(self.buffer)

class DataProcessor:
    def __init__(self):
        self.context = zmq.Context()
        self.data_socket = None
        self.event_socket = None
        self.poller = zmq.Poller()
        self.message_num = -1
        self.socket_waits_reply = False
        self.app_name = 'Data Processor Falcon'
        self.uuid = str(uuid.uuid4())
        self.last_heartbeat_time = 0
        self.last_reply_time = time.time()
        self.channel_num = 128 
        self.sample_rate = 10000  
        self.target_rate = 2048 # The rate we need for the classifier
        self.batch_size = int(0.150 * self.sample_rate)  # 150 ms batch size of streamed data
        self.buffer_size = 0.151
        self.rolling_buffer = RollingBuffer(self.channel_num, self.sample_rate, self.buffer_size)
        self.channel_data_received = [False] * self.channel_num
        self.running = True
        self.lock = Lock()  # To synchronize access to shared resources
        self.model = None   # Classifier
        self.feat = False    # Does the model need T4 features?
        self.publisher = None   # ROS publisher for classification publishing
        self.multi_labels_actual = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])#,[1,1,0,0,0,0,0,0],[1,0,1,0,0,0,0,0],[1,0,0,1,0,0,0,0],[0,1,1,0,0,0,0,0],[0,1,0,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,0,1,0],[0,0,0,0,1,0,0,1],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,0,1],[0,0,0,0,0,0,1,1],[1,0,0,0,0,1,0,0],[1,0,0,0,0,0,1,0],[1,0,0,0,0,0,0,1],[0,1,0,0,1,0,0,0],[0,0,1,0,1,0,0,0],[0,0,0,1,1,0,0,0]])
        self.amplitude_threshold = 0
        self.class_mapping = { # Bit of a hack... modify per classifier
            1: 1,
            2: 2,
            3: 2,
            4: 4, 
            5: 1,
            6: 5,
            7: 6,
            8: 8 
        }
        self.total_count = 0
        self.correct = 0

    def init_socket(self):
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.connect("tcp://localhost:5555")
        self.data_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.poller.register(self.data_socket, zmq.POLLIN)

    def start(self):
        self.init_socket()
        try:
            # Get node params
            robot_name = rospy.get_param('~robot_name', "my_gen3")

            # Define publisher
            self.publisher = rospy.Publisher('/'+robot_name+'/desired_pose', MultiLabelActionNotification, queue_size=10)
            rospy.loginfo("Pose publisher is live.")
        except Exception as e:
            print(f"ROS node or publisher failed to instantiate.")
            rospy.logerr("Publisher failed to instantiate.")

        try:
            # Define resource path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, r"../rsc/")
            self.model = pickle.load(open(path + "S1_sesh_4.pkl", "rb"))
            self.feat = False
            rospy.loginfo("Model loaded.")
        except Exception as e:
            print(f"Model loading error: {e}")
            rospy.logerr("Model loading error.")
        print("Acquisition started.")
        Thread(target=self.data_acquisition).start()
        time.sleep(0.160)
        print("Processing started.")
        Thread(target=self.data_processing).start()
    
    # Feature extraction functions
    def mean_absolute_value(self, emg_signal):
        return np.mean(np.abs(emg_signal))

    def zero_crossing(self, emg_signal, threshold=0):
        zc_count = 0
        for i in range(1, len(emg_signal)):
            if ((emg_signal[i-1] > threshold and emg_signal[i] < -threshold) or
                (emg_signal[i-1] < -threshold and emg_signal[i] > threshold)):
                zc_count += 1
        return zc_count

    def slope_sign_changes(self, emg_signal, threshold=0):
        ssc_count = 0
        for i in range(1, len(emg_signal) - 1):
            if ((emg_signal[i] > emg_signal[i-1] and emg_signal[i] > emg_signal[i+1]) or
                (emg_signal[i] < emg_signal[i-1] and emg_signal[i] < emg_signal[i+1])):
                if (abs(emg_signal[i] - emg_signal[i-1]) > threshold or
                    abs(emg_signal[i] - emg_signal[i+1]) > threshold):
                    ssc_count += 1
        return ssc_count

    def waveform_length(self, emg_signal):
        return np.sum(np.abs(np.diff(emg_signal)))
    
    # Notch filter function
    def notchsignals(self, signal, fsamp, notch_freq=50.0, Q=30.0):
        # Applies a notch filter to each channel of a multi-channel signal array.
        b, a = iirnotch(notch_freq, Q, fsamp)
        filteredsignal = filtfilt(b, a, signal, axis=1)
        return filteredsignal
    
    def standardize_by_sample(self, data):
        # Compute the mean and standard deviation for each sample
        mean_vals = data.mean(axis=1, keepdims=True)
        std_vals = data.std(axis=1, keepdims=True)
        
        # Apply standard scaling
        standardized_data = (data - mean_vals) / std_vals
    
        return standardized_data

    def process(self, data_batch): #57ms +/- 5ms
        #start = time.time()        
        # Data transformation using vectorized operations
        num_channels, num_samples = data_batch.shape # 128,1500
        # Calculate the mean rectified amplitude across channels
        mean_rectified_amplitude = np.mean(np.mean(np.abs(data_batch), axis=1))

        # Check if the mean rectified amplitude is above the threshold for any channel
        if (mean_rectified_amplitude < self.amplitude_threshold):
            #print(f"baby too smoll: {data_batch}")
            return
        
        # Downsample to CNN rate
        num_resamp = int(num_samples * (self.target_rate / self.sample_rate))
        downsampled_data = resample(data_batch, num_resamp, axis=1)
        num_samples = downsampled_data.shape[1]

        # Reshape data for vectorized double differentiation
        reshaped_data = downsampled_data.reshape(16, 8, num_samples)

        # Calculate the differences
        diff1 = reshaped_data[:, :6, :] - reshaped_data[:, 1:7, :]
        diff2 = reshaped_data[:, 2:8, :] - reshaped_data[:, 1:7, :]

        # Sum the differences
        transformed_data = diff1 + diff2

        # Reshape the transformed data to the expected shape
        transformed_data = transformed_data.reshape(96, num_samples)
        # Apply Notch filter
        filtered_data = self.notchsignals(transformed_data, self.target_rate) 
 
        # Standardize the data by sample
        standardized_data = self.standardize_by_sample(filtered_data).T
        
        if self.feat == True:
            # Calculate the time domain features per channel
            features = np.zeros(shape=(standardized_data.shape[0], 4))
            for i in range(standardized_data.shape[0]):
                data = standardized_data[i,:]  # Get current channel
                features[i][0] = self.mean_absolute_value(data)
                features[i][1] = self.zero_crossing(data)
                features[i][2] = self.slope_sign_changes(data)
                features[i][3] = self.waveform_length(data)

            # Classify the data
            prediction = list(map(int, self.model.predict([features.flatten()]).flatten()))
        else:
            
            # Classify using CNN
            prediction = self.model.predict(np.array([standardized_data]),verbose=0)

        # Map the predicted class to the correct class
        predicted_class = np.argmax(prediction)
        corrected_class = self.class_mapping.get(predicted_class + 1, predicted_class + 1) - 1

        # Get the closest matching label
        closest_match = self.multi_labels_actual[corrected_class]
        print( closest_match)
        if list(closest_match) == [0,0,0,0,0,0,0,1]:
            self.correct += 1
        
        self.total_count += 1

        rospy.loginfo(f"Predicted pose - {closest_match}")
        print(f"Accuracy - {self.correct/self.total_count}")
        msg = MultiLabelActionNotification()
        msg.pose_array = list(map(int,closest_match.flatten()))
        #self.publisher.publish(msg)
        
        
    def data_acquisition(self):
        while self.running:
            socks = dict(self.poller.poll(1))
            if not socks:
                continue

            if self.data_socket in socks:
                try:
                    message = self.data_socket.recv(zmq.NOBLOCK)
                except zmq.ZMQError as err:
                    print(f"got error: {err}")
                    continue

                # Step 3: Decode the message
                try:
                    buf = bytearray(message)
                    data = ContinuousData.GetRootAsContinuousData(buf, 0)
                except Exception as e:
                    print(f"Impossible to parse the packet received - skipping to the next. Error: {e}")
                    continue
                
                # Extract data and parameters
                samples_flat = data.SamplesAsNumpy()
                num_samples = data.NSamples()
                num_channels = data.NChannels()

                # Check if the total size matches
                total_elements = samples_flat.size
                expected_elements = num_samples * num_channels
                
                if total_elements != expected_elements:
                    print(f"Error: Expected {expected_elements} elements but got {total_elements}.")

                elif num_channels == 0:
                    print("Channels from Falcon = 0")
                else:
                    # Reshape the flat samples array
                    samples_reshaped = samples_flat.reshape((num_channels, num_samples))
                    self.rolling_buffer.add_data(samples_reshaped)

    def data_processing(self):
        while self.running:
            buffer_data = self.rolling_buffer.get_buffer()
            if buffer_data.shape[1] >= self.batch_size:
                batch_data = buffer_data[:, -self.batch_size:]
                self.process(batch_data)
            time.sleep(0.160)  # Wait for the next batch interval

    def stop(self):
        self.running = False

if __name__ == '__main__':
    processor = DataProcessor()
    try:
        processor.start()
    except KeyboardInterrupt:
        processor.stop()
