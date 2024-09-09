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
from ContinuousData import ContinuousData
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from scipy.signal import iirnotch, filtfilt, resample

class RollingBuffer:
    def __init__(self, num_channels, sampling_rate, buffer_duration, batch_size):
        self.num_channels = num_channels
        self.sampling_rate = sampling_rate
        self.buffer_duration = buffer_duration
        self.batch_size = batch_size
        
        # Calculate the total number of samples for the buffer
        self.total_samples = int(sampling_rate * buffer_duration)
        self.buffer = np.zeros((num_channels, self.total_samples))
        
        # Initialize pointers
        self.write_index = 0
        self.lock = Lock()
        
    def add_data(self, data):
        with self.lock:
            num_samples = data.shape[1]
            if num_samples != self.batch_size:
                raise ValueError("Incoming batch size does not match the specified batch size.")
            
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
        self.num_samples_per_message = 640
        self.batch_size = int(0.150 * self.sample_rate)  # 150 ms batch size of streamed data
        self.buffer_size = 10  # Buffer to hold 10 seconds of data
        self.rolling_buffer = RollingBuffer(self.channel_num, self.sample_rate, self.buffer_size, self.num_samples_per_message)
        self.channel_data_received = [False] * self.channel_num
        self.running = True
        self.lock = Lock()  # To synchronize access to shared resources
        self.model = None   # Classifier
        self.feat = False    # Does the model need T4 features?
        self.publisher = None   # ROS publisher for classification publishing

    def init_socket(self):
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.connect("tcp://localhost:5555")
        self.data_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.poller.register(self.data_socket, zmq.POLLIN)

    def start(self):
        self.init_socket()
        try:
            # Get node params
            #robot_name = rospy.get_param('~robot_name', "my_gen3")

            # Define publisher
            #self.publisher = rospy.Publisher('/'+robot_name+'/desired_pose', MultiLabelActionNotification, queue_size=10)
            rospy.loginfo("Pose publisher is live.")
        except Exception as e:
            print(f"ROS node or publisher failed to instantiate.")
            rospy.logerr("Publisher failed to instantiate.")

        try:
            # Define resource path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, r"../rsc/")
            self.model = pickle.load(open(path + "S2_combo_actual.pkl", "rb"))
            self.feat = False
            rospy.loginfo("Model loaded.")
            print("model loaded")
        except Exception as e:
            print(f"Model loading error: {e}")
            rospy.logerr("Model loading error.")
        Thread(target=self.data_acquisition).start()
        time.sleep(0.150)
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
    def notch_filter(self, signal, fsamp, notch_freq=50.0, Q=30.0):
        b, a = iirnotch(notch_freq, Q, fsamp)
        return filtfilt(b, a, signal)
    
    # Notch filter function
    def notchsignals(self, signal, fsamp, notch_freq=50.0, Q=30.0):
        # Applies a notch filter to each channel of a multi-channel signal array.
        b, a = iirnotch(notch_freq, Q, fsamp)
        filteredsignal = filtfilt(b, a, signal, axis=1)
        return filteredsignal
    
    def notchsignals1(self, signal, fsamp, notch_freq=50.0, Q=30.0):
        # Applies a notch filter to each channel of a multi-channel signal array.
        filteredsignal = np.zeros(signal.shape)
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.notch_filter, [signal[i, :] for i in range(signal.shape[0])], [fsamp]*signal.shape[0], [notch_freq]*signal.shape[0], [Q]*signal.shape[0]))
        
        for i, result in enumerate(results):
            filteredsignal[i, :] = result
        return filteredsignal

    def process(self, data_batch):
        process_start_time = time.time()
        
        # Data transformation using vectorized operations
        transform_start_time = time.time()
        num_channels, num_samples = data_batch.shape # 128,1500

        # Reshape data for vectorized double differentiation
        reshaped_data = data_batch.reshape(16, 8, num_samples)

        # Calculate the differences
        diff1 = reshaped_data[:, :6, :] - reshaped_data[:, 1:7, :]
        diff2 = reshaped_data[:, 2:8, :] - reshaped_data[:, 1:7, :]

        # Sum the differences
        transformed_data = diff1 + diff2

        # Reshape the transformed data to the expected shape
        transformed_data = transformed_data.reshape(96, num_samples)
        transform_end_time = time.time()
        print(f"Data transformation took {transform_end_time - transform_start_time:.4f} seconds")
    
        # Apply Notch filter
        notch_start_time = time.time()
        filtered_data = self.notchsignals(transformed_data, self.sample_rate) #slow
        notch_end_time = time.time()
        print(f"Notch filtering took {notch_end_time - notch_start_time:.4f} seconds")

        if self.feat == True:
            # Calculate the time domain features per channel
            features = np.zeros(shape=(filtered_data.shape[0], 4))
            for i in range(filtered_data.shape[0]):
                data = filtered_data[i,:]  # Get current channel
                features[i][0] = self.mean_absolute_value(data)
                features[i][1] = self.zero_crossing(data)
                features[i][2] = self.slope_sign_changes(data)
                features[i][3] = self.waveform_length(data)

            # Classify the data
            prediction = list(map(int, self.model.predict([features.flatten()]).flatten()))
        else:
            # Downsample to CNN rate
            ds_start_time = time.time()
            num_resamp = int(num_samples * (self.target_rate / self.sample_rate))
            filtered_data = resample(filtered_data, num_resamp, axis=1).T
            ds_end_time = time.time()
            print(f"Down sampling took {ds_end_time - ds_start_time:.4f} seconds")

            # Classify using CNN
            prediction = self.model.predict(np.array([filtered_data]))

        rospy.loginfo(f"Predicted pose - {prediction}")
        #msg = MultiLabelActionNotification()
        #msg.pose_array = prediction
        #self.publisher.publish(msg)
        
        process_end_time = time.time()
        print(f"Total process method took {process_end_time - process_start_time:.4f} seconds")

    def data_acquisition(self):
        batch_progress = 0 
        acquisition_start_time = time.time()
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
                else:
                    # Reshape the flat samples array
                    samples_reshaped = samples_flat.reshape((num_channels, num_samples))
                    
                    self.rolling_buffer.add_data(samples_reshaped)
                
                batch_progress+=num_samples

            if batch_progress >= self.batch_size:
                acquisition_end_time = time.time()
                print(f"Data acquisition took {acquisition_end_time - acquisition_start_time:.4f} seconds")
                batch_progress = 0
                acquisition_start_time = time.time()

    def data_processing(self):
        while self.running:
            buffer_data = self.rolling_buffer.get_buffer()
            if buffer_data.shape[1] >= self.batch_size:
                batch_start_time = time.time()
                batch_data = buffer_data[:, -self.batch_size:]
                self.process(batch_data)
                batch_end_time = time.time()
                print(f"Batch processing took {batch_end_time - batch_start_time:.4f} seconds \n")
            time.sleep(0.150)  # Wait for the next batch interval

    def stop(self):
        self.running = False

if __name__ == '__main__':
    script_start_time = time.time()
    processor = DataProcessor()
    try:
        processor.start()
    except KeyboardInterrupt:
        processor.stop()
    script_end_time = time.time()
    print(f"Script startup took {script_end_time - script_start_time:.4f} seconds")
