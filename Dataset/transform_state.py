import numpy as np
import os

def transform_state():
    current_dir = os.path.dirname(__file__) 
    state_path = os.path.join(current_dir, os.pardir, 'data')
    for demo_id in os.listdir(state_path):
        demo_dir = os.path.join(state_path, demo_id)
        left_pose = np.load(os.path.join(demo_dir, 'Lpsm_pose.npy'))
        right_pose = np.load(os.path.join(demo_dir, 'Rpsm_pose.npy'))
        left_gripper_state = np.load(os.path.join(demo_dir, 'Lgripper_state.npy'))
        right_gripper_state = np.load(os.path.join(demo_dir, 'Rgripper_state.npy'))
        
        min_len = np.min([len(left_pose), len(right_pose), len(left_gripper_state), len(right_gripper_state)])
        left_pose = left_pose[:min_len]
        right_pose = right_pose[:min_len]
        left_gripper_state = left_gripper_state[:min_len]
        right_gripper_state = right_gripper_state[:min_len]
        
        # deal with orientation
        left_position = left_pose[:, :3]
        right_position = right_pose[:, :3]
        left_orientation = left_pose[:, 3:7]
        right_orientation = right_pose[:, 3:7]
        
        demo_state = []

        for i in range(min_len):
            LQuaternion_correction = np.dot(left_orientation[i], left_orientation[np.max(i-1, 0)])
            RQuaternion_correction = np.dot(right_orientation[i], right_orientation[np.max(i-1, 0)])
            if LQuaternion_correction < 0:
                left_orientation[i] = -left_orientation[i]
            if RQuaternion_correction < 0:
                right_orientation[i] = -right_orientation[i]
            
            demo_state.append(np.hstack((left_position[i], left_orientation[i], left_gripper_state[i], right_position[i], right_orientation[i], right_gripper_state[i])))
            
        demo_state = np.array(demo_state)
        # create and save to txt

        demo_id_num = demo_id.split('_')[0]
        with open(os.path.join(current_dir, 'state', f'{demo_id_num}.txt'), 'w') as f:
            for state in demo_state:
                f.write(' '.join(map(str, state)) + '\n')



    return   

if __name__ == '__main__':
    transformed_state = transform_state()
