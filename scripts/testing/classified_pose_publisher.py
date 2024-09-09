#!/usr/bin/env python

import rospy
import numpy as np
import time
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from kortex_synergy.msg import CommandActionNotification
#from kortex_synergy.rsc import trained_cnn_model, sim_data


# Function to simulate real-time classification
def simulate_realtime_classification(batches, classifier_model, publisher):
    start_time = time.time()  # Get the start time
    previous_class = None
    sampling_rate = 2000  # Sampling rate in Hz
    batch_duration = 1 / sampling_rate  # Duration of each batch in seconds
    num_samples_per_batch = 250
    num_channels = 36

    for batch_idx, batch in enumerate(batches):
        current_time = time.time() - start_time
        if current_time < 30:  # Simulate for 30 seconds
            # Classifier.predict() takes a batch of shape (batch_size, 250, 36)
            batch_reshaped = np.reshape(batch, (-1, num_samples_per_batch, num_channels))
            predictions = classifier_model.predict(batch_reshaped, verbose=0)
            # Extracting the class with the highest probability
            predicted_class = np.argmax(predictions)+1
            if previous_class != predicted_class:
                rospy.loginfo(f"Time: {current_time:.2f}s, Batch {batch_idx + 1}: Predicted pose - {predicted_class}")
                msg = CommandActionNotification()
                msg.pose_value = predicted_class
                publisher.publish(msg)
                previous_class = predicted_class
            time.sleep(batch_duration/5)  # Wait for the duration of each batch
        else:
            rospy.loginfo("Simulation complete. Shutting down...")
            rospy.signal_shutdown("Simulation complete")
            break

            
def main():
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 (default), 1 (INFO), 2 (WARNING), 3 (ERROR)
    
    # Define resource path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "../rsc/")

    rospy.init_node('classified_pose_publisher', anonymous = True)
    # Get node params
    robot_name = rospy.get_param('~robot_name', "my_gen3")

    # Define publisher
    pub = rospy.Publisher('/'+robot_name+'/desired_pose', CommandActionNotification, queue_size=10)
    rate = rospy.Rate(10)

    # Load pre-trained model
    rospy.loginfo("Loading model...")
    model = load_model(path+"trained_cnn_model.keras")

    # Load data
    rospy.loginfo("Loading data...")
    data = np.load(path+"sim_data.npy")
    
    simulate_realtime_classification(data, model, pub)
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
