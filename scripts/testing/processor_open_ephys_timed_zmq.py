import zmq
import numpy as np
import json
import uuid
import time
import os
import pickle
import rospy
import xgboost as xgb
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from scipy.signal import iirnotch, filtfilt

class DataProcessor:
    def __init__(self):
        self.context = zmq.Context()
        self.data_socket = None
        self.event_socket = None
        self.poller = zmq.Poller()
        self.message_num = -1
        self.socket_waits_reply = False
        self.app_name = 'Data Processor'
        self.uuid = str(uuid.uuid4())
        self.last_heartbeat_time = 0
        self.last_reply_time = time.time()
        self.channel_num = 128  # Assuming 128 channels for EMG
        self.sample_rate = 30000  # Assuming a sample rate of 30000 Hz
        self.num_samples_per_message = 640
        self.batch_size = int(0.150 * self.sample_rate)  # 150 ms batch size
        self.buffer_size = 10 * self.sample_rate  # Buffer to hold 10 seconds of data
        self.data_buffer = np.zeros((self.channel_num, self.buffer_size))
        self.buffer_index = 0
        self.channel_data_received = [False] * self.channel_num
        self.running = True
        self.lock = Lock()  # To synchronize access to shared resources
        self.model = None   # Classifier
        self.publisher = None   # ROS publisher for classification publishing

    def init_socket(self):
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.connect("tcp://localhost:5556")
        self.data_socket.setsockopt(zmq.SUBSCRIBE, b'')
        self.poller.register(self.data_socket, zmq.POLLIN)

        self.event_socket = self.context.socket(zmq.REQ)
        self.event_socket.connect("tcp://localhost:5557")
        self.poller.register(self.event_socket, zmq.POLLIN)

    def send_heartbeat(self):
        while self.running:
            try:
                if (time.time() - self.last_heartbeat_time) > 2.0:
                    d = {'application': self.app_name, 'uuid': self.uuid, 'type': 'heartbeat'}
                    j_msg = json.dumps(d)
                    self.event_socket.send(j_msg.encode('utf-8'))
                    self.last_heartbeat_time = time.time()
                    self.socket_waits_reply = True

                if self.socket_waits_reply:
                    socks = dict(self.poller.poll(1000))
                    if self.event_socket in socks:
                        message = self.event_socket.recv()
                        self.socket_waits_reply = False
            except zmq.ZMQError as err:
                print(f"Heartbeat error: {err}")
                time.sleep(1)

    def start(self):
        self.init_socket()
        Thread(target=self.send_heartbeat).start()
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
            self.model = pickle.load(open(path + "S2_xgb_nm_30.pkl", "rb"))
            rospy.loginfo("Model loaded.")
        except Exception as e:
            print(f"Model loading error: {e}")
            rospy.logerr("Model loading error.")
        Thread(target=self.data_acquisition).start()
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
        num_channels, num_samples = data_batch.shape

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
        #filtered_data = self.notch_filter(transformed_data, self.sample_rate)
        notch_end_time = time.time()
        print(f"Notch filtering took {notch_end_time - notch_start_time:.4f} seconds")

        # Calculate the time domain features per channel
        features_start_time = time.time()
        features = np.zeros(shape=(filtered_data.shape[0], 4))
        for i in range(filtered_data.shape[0]):
            data = filtered_data[i,:]  # Get current channel
            features[i][0] = self.mean_absolute_value(data)
            features[i][1] = self.zero_crossing(data)
            features[i][2] = self.slope_sign_changes(data)
            features[i][3] = self.waveform_length(data)
        features_end_time = time.time()
        print(f"Feature extraction took {features_end_time - features_start_time:.4f} seconds")

        # Classify the data
        classification_start_time = time.time()
        prediction = list(map(int, self.model.predict([features.flatten()]).flatten()))
        classification_end_time = time.time()
        print(f"Classification took {classification_end_time - classification_start_time:.4f} seconds")
        
        rospy.loginfo(f"Predicted pose - {prediction}")
        #msg = MultiLabelActionNotification()
        #msg.pose_array = prediction
        #self.publisher.publish(msg)
        
        process_end_time = time.time()
        print(f"Total process method took {process_end_time - process_start_time:.4f} seconds")

    def data_acquisition(self):
        total_samples_acquired = 0  # Variable to keep track of the total number of samples acquired

        while self.running:
            acquisition_start_time = time.time()
            socks = dict(self.poller.poll(1))
            if not socks:
                continue

            if self.data_socket in socks:
                try:
                    message = self.data_socket.recv_multipart(zmq.NOBLOCK)
                except zmq.ZMQError as err:
                    print(f"got error: {err}")
                    continue

                if message:
                    try:
                        header = json.loads(message[1].decode('utf-8'))
                    except ValueError as e:
                        print(f"ValueError: {e}")
                        print(message[1])
                        continue

                    if header['type'] == 'data':
                        c = header['content']
                        channel_num = c['channel_num']
                        num_samples = c['num_samples']

                        if num_samples != self.num_samples_per_message:
                            print(f"Unexpected number of samples: {num_samples}")
                            continue

                        try:
                            data = np.frombuffer(message[2], dtype=np.float32)
                            with self.lock:
                                start_index = self.buffer_index % self.buffer_size
                                end_index = (self.buffer_index + num_samples) % self.buffer_size

                                if start_index < end_index:
                                    self.data_buffer[channel_num, start_index:end_index] = data
                                else:
                                    self.data_buffer[channel_num, start_index:] = data[:self.buffer_size - start_index]
                                    self.data_buffer[channel_num, :end_index] = data[self.buffer_size - start_index:]

                                self.buffer_index += num_samples
                                self.channel_data_received[channel_num] = True
                                
                                # Update the total number of samples acquired
                                total_samples_acquired += num_samples

                                # Print the number of samples acquired in this iteration
                                print(f"Samples acquired in this iteration: {num_samples}, {data.shape}")
                                #print(f"Total samples acquired so far: {total_samples_acquired}")

                        except ValueError as e:
                            print(e)
                            print(header)
                            print(message[1])

            acquisition_end_time = time.time()
            print(f"Data acquisition took {acquisition_end_time - acquisition_start_time:.4f} seconds")

    def data_processing(self):
        while self.running:
            with self.lock:
                if self.buffer_index >= self.batch_size:
                    batch_start_time = time.time()
                    batch_data = self.data_buffer[:, self.buffer_index - self.batch_size:self.buffer_index]
                    self.process(batch_data)
                    self.buffer_index = 0  # Reset buffer index for the next batch
                    batch_end_time = time.time()
                    print(f"Batch processing took {batch_end_time - batch_start_time:.4f} seconds \n")
            #time.sleep(0.150)  # Wait for the next batch interval

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
    print(f"Entire script took {script_end_time - script_start_time:.4f} seconds")
