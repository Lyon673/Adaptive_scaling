import cv2
import socket
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
import os
from os.path import join
import time
import sys

pupil = []
start_time = time.time()

def get_screen_coor(x, y, w, h):
    return int(x * w), int(y * h)

def init_publisher():
    pub = rospy.Publisher('gaze_data', Float32MultiArray, queue_size=10)
    pubtest = rospy.Publisher('pointtest', Float32MultiArray, queue_size=10)
    rospy.init_node('gaze_tracker', anonymous=True)
    return pub,pubtest


def create_gaze_message(left_valid, left_x, left_y, right_valid, right_x, right_y, left_pupil, right_pupil,timestamp):
    msg = Float32MultiArray()
    
    msg.layout.dim = [MultiArrayDimension()]
    msg.layout.dim[0].size = 9 
    msg.layout.dim[0].stride =9
    msg.layout.dim[0].label = "gaze_data"
    
    
    msg.data = [float(left_valid), float(left_x), float(left_y), 
                float(right_valid), float(right_x), float(right_y),
                float(left_pupil), float(right_pupil), timestamp]
    return msg

def main():
    # Connect to gaze tracker
    server_ip = '192.168.44.33'
    server_port = 7979

    print(f"<LYON> Connecting to gaze tracker at {server_ip}:{server_port}")

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))
    print(f"<LYON> Successfully connected to gaze tracker at {server_ip}:{server_port}")

    # ROS publisher
    gaze_pub, pubtest = init_publisher()
    

    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        # Read gaze point
        data = client_socket.recv(1024)
        sys.stdout.write('\r-- Time past: %02.1f' % float(time.time() - start_time))
        sys.stdout.flush()

        try:

            data_str = str(data)
            print(f"\n<LYON> data: {data_str}")
            
            data_parts = data_str.split(',')
            left_valid = str2bool(data_parts[4])  # false -> 0
            left_x = float(data_parts[5])
            left_y = float(data_parts[6])
            right_valid = str2bool(data_parts[7])  # false -> 0
            right_x = float(data_parts[8])
            right_y = float(data_parts[9])
            left_pupil = float(data_parts[10])
            right_pupil = float(data_parts[11])
            timestamp = int(data_parts[0].split('{')[1])
            
            pupil.append([left_pupil, right_pupil,left_valid,right_valid,timestamp])

            gaze_msg = create_gaze_message(left_valid, left_x, left_y, 
                                        right_valid, right_x, right_y,
                                        left_pupil, right_pupil,timestamp)
            gaze_pub.publish(gaze_msg)


            msg_test = Float32MultiArray()
            msg_test.layout.dim = [MultiArrayDimension()]
            msg_test.layout.dim[0].size = 2
            msg_test.layout.dim[0].stride = 2
            msg_test.layout.dim[0].label = "test"

            msg_test.data = [np.round(left_x * 640) , np.round(left_y * 360)] 
            pubtest.publish(msg_test)
            

        except (IndexError, ValueError) as e:
            print(f"Error parsing data: {e}")
            continue

        rate.sleep()

    client_socket.close()

def str2bool(str):
    if str == "false":
        return False
    return True

if __name__ == '__main__':
    try:
        main()
        # pupilnp = np.array(pupil)
        # if not os.path.exists('gazedata'):
        #     os.mkdir('gazedata')
        # print("saving data...")
        # np.save(join('gazedata', 'gaze04.npy'), pupilnp)
        # print("done")
    except rospy.ROSInterruptException:
        pass

    