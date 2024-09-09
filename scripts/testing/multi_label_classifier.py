#!/usr/bin/env python

import rospy
import numpy as np
import time
import os
import pickle
import xgboost as xgb
from kortex_synergy.msg import MultiLabelActionNotification
#from kortex_synergy.rsc import trained_cnn_model, sim_data


# Function to simulate real-time classification
def simulate_realtime_classification(batches, labels, classifier_model, publisher):
    start_time = time.time()  # Get the start time
    previous_class = np.array([0,0,0,0,0,0,0,0])
    sampling_rate = 2048  # Sampling rate in Hz
    batch_duration = 1 / sampling_rate  # Duration of each batch in seconds

    for batch_idx, batch in enumerate(batches):
        current_time = time.time() - start_time
        if current_time < 30:  # Simulate for 30 seconds
            # Classifier.predict() takes a batch of shape (batch_size, 250, 36)
            prediction = list(map(int,classifier_model.predict([batch]).flatten()))
            #if previous_class.all() != prediction.all():
            rospy.loginfo(f"Time: {current_time:.2f}s, Batch {batch_idx + 1}: Predicted pose - {prediction}")
            msg = MultiLabelActionNotification()
            msg.pose_array = prediction
            publisher.publish(msg)
            #previous_class = prediction
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
    pub = rospy.Publisher('/'+robot_name+'/desired_pose', MultiLabelActionNotification, queue_size=10)
    rate = rospy.Rate(10)

    # Load pre-trained model
    rospy.loginfo("Loading model...")
    model = pickle.load(open(path+f"S2_xgb_nm_30.pkl", "rb"))

    # Load data
    rospy.loginfo("Loading data...")
    data = np.load(path+"S2_prepped_data.npy")

    # Load data
    rospy.loginfo("Loading labels...")
    labels = np.load(path+"S2_y_test_multi_nm_30.npy")
    
    simulate_realtime_classification(data, labels, model, pub)
    

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
