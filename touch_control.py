#!/usr/bin/env python
# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2020-2021 Johns Hopkins University (JHU), Worcester Polytechnic Institute (WPI) All Rights Reserved.


#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.


#     \author    <amunawar@jhu.edu>
#     \author    Adnan Munawar
#     \version   1.0
# */
# //==============================================================================
import sys
import os

# 1. 获取当前文件(main.py)的绝对路径
current_file_path = os.path.abspath(__file__)

# 2. 获取当前文件所在的目录 (sub_app/)
current_dir = os.path.dirname(current_file_path)

# 3. 获取上一级目录 (my_project/)
parent_dir = os.path.dirname(current_dir)

# 4. 将上一级目录添加到 sys.path
#    （把它放在列表的开头，以便优先搜索）
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.psm_arm import PSM
import time
import rospy
from PyKDL import Frame, Rotation, Vector
from argparse import ArgumentParser
from surgical_robotics_challenge.input_devices.geomagic_device import GeomagicDevice
from itertools import cycle
from surgical_robotics_challenge.utils.jnt_control_gui import JointGUI
from surgical_robotics_challenge.utils.utilities import get_boolean_from_opt
from surgical_robotics_challenge.utils import coordinate_frames
import sys
from std_msgs.msg import Float32MultiArray


class ControllerInterface:
    def __init__(self, Lleader, Rleader, psm_arms, ecm):
        self.counter = 0
        self.Lleader = Lleader
        self.Rleader = Rleader
        self.Lpsm = psm_arms[0]
        self.Rpsm = psm_arms[1]
        self.gui = JointGUI('ECM JP', 4, ["ecm j0", "ecm j1", "ecm j2", "ecm j3"], lower_lims=cam.get_lower_limits(),
                            upper_lims=cam.get_upper_limits())
        self.Lcmd_xyz = self.Lpsm.T_t_b_home.p
        self.Rcmd_xyz = self.Rpsm.T_t_b_home.p
        self.Lleader_prev_rpy = None
        self.Rleader_prev_rpy = None
        self.LM_p_c = self.Lleader.measured_cp().M
        self.RM_p_c = self.Rleader.measured_cp().M

        self.initial_Lcmd_xyz = self.Lpsm.T_t_b_home.p
        self.initial_Rcmd_xyz = self.Rpsm.T_t_b_home.p
        self.initial_LM_p_c = self.Lleader.measured_cp().M
        self.initial_RM_p_c = self.Rleader.measured_cp().M
        
        self.Lcmd_rpy = None
        self.Rcmd_rpy = None
        self.LT_IK = None
        self.RT_IK = None
        self.L_IK = None
        self.R_IK = None

        self._ecm = ecm

        self._LT_c_b = None
        self._RT_c_b = None
        self._update_T_c_b = True

        self.received_msg = None

        self.scale = [15,15]
        #self.subscriber = rospy.Subscriber('/scale', Float32MultiArray, self.scale_cb)


    def update_T_c_b(self):
        if self._update_T_c_b or self._ecm.has_pose_changed():
            self._LT_c_b = self.Lpsm.get_T_w_b() * self._ecm.get_T_c_w()
            self._RT_c_b = self.Rpsm.get_T_w_b() * self._ecm.get_T_c_w()
            self._update_T_c_b = False

    def update_camera_pose(self):
        self.gui.App.update()
        self._ecm.servo_jp(self.gui.jnt_cmds)

    def update_arm_pose(self):
        self.update_T_c_b()
        # Left Arm
        Ltwist = self.Lleader.measured_cv()
        
        if not self.Lleader.clutch_button_pressed:
            self.Lcmd_xyz = self.Lpsm.T_t_b_home.p
            #delta_t = self._T_c_b.M * twist.vel * 0.000002 * self.scale
            delta_t = self._LT_c_b.M * Ltwist.vel * 0.0000002 * self.scale[0] # 0.00004
            self.Lcmd_xyz = self.Lcmd_xyz + delta_t
            self.Lpsm.T_t_b_home.p = self.Lcmd_xyz

            if self.Lleader_prev_rpy is not None:
                delta_rpy = self.Lleader_prev_rpy.Inverse() * self.Lleader.measured_cp().M
                self.LM_p_c = self.LM_p_c * delta_rpy
                self.Lcmd_rpy = self._LT_c_b.M * self.LM_p_c* Rotation.RPY(3.14, 0, 3.14 / 2)
                self.Lleader_prev_rpy = self.Lleader.measured_cp().M
            else:
                self.Lleader_prev_rpy = self.Lleader.measured_cp().M
                self.Lcmd_rpy = self._LT_c_b.M * self.LM_p_c * Rotation.RPY(3.14, 0, 3.14 / 2)
        else:
            self.Lleader_prev_rpy = self.Lleader.measured_cp().M
            self.Lcmd_rpy = self._LT_c_b.M * self.LM_p_c * Rotation.RPY(3.14, 0, 3.14 / 2)

        if self.Lleader.double_press_gripper:
            self.LM_p_c = self.Lleader.measured_cp().M
            self.Lleader.double_press_gripper = False

        self.LT_IK = Frame(self.Lcmd_rpy, self.Lcmd_xyz)
        self.Lpsm.servo_cp(self.LT_IK)
        self.Lpsm.set_jaw_angle(self.Lleader.get_jaw_angle())

        # Right Arm
        Rtwist = self.Rleader.measured_cv()
        
        if not self.Rleader.clutch_button_pressed:
            self.Rcmd_xyz = self.Rpsm.T_t_b_home.p
            delta_t = self._RT_c_b.M * Rtwist.vel * 0.0000002 * self.scale[1] # 0.00004
            self.Rcmd_xyz = self.Rcmd_xyz + delta_t
            self.Rpsm.T_t_b_home.p = self.Rcmd_xyz

            if self.Rleader_prev_rpy is not None:
                delta_rpy = self.Rleader_prev_rpy.Inverse() * self.Rleader.measured_cp().M
                self.RM_p_c = self.RM_p_c * delta_rpy
                self.Rcmd_rpy = self._RT_c_b.M * self.RM_p_c* Rotation.RPY(3.14, 0, 3.14 / 2)
                self.Rleader_prev_rpy = self.Rleader.measured_cp().M
            else:
                self.Rleader_prev_rpy = self.Rleader.measured_cp().M
                self.Rcmd_rpy = self._RT_c_b.M * self.RM_p_c * Rotation.RPY(3.14, 0, 3.14 / 2)
        else:
            self.Rleader_prev_rpy = self.Rleader.measured_cp().M
            self.Rcmd_rpy = self._RT_c_b.M * self.RM_p_c * Rotation.RPY(3.14, 0, 3.14 / 2)

        if self.Rleader.double_press_gripper:
            self.RM_p_c = self.Rleader.measured_cp().M
            self.Rleader.double_press_gripper = False

        # if self.Rleader.double_press_gripper or self.Lleader.double_press_gripper:
        #     self.Lcmd_xyz = self.initial_Lcmd_xyz
        #     self.Rcmd_xyz = self.initial_Rcmd_xyz
        #     self.Lpsm.T_t_b_home.p = self.Lcmd_xyz
        #     self.Rpsm.T_t_b_home.p = self.Rcmd_xyz
        #     self.Lleader_prev_rpy = None
        #     self.Rleader_prev_rpy = None
        #     self.LM_p_c = self.initial_LM_p_c
        #     self.RM_p_c = self.initial_RM_p_c
            
        #     self.Rleader.double_press_gripper = False
        #     self.Lleader.double_press_gripper = False

        # If clutch is pressed, Rcmd_rpy remains unchanged (rotation is locked)
        self.RT_IK = Frame(self.Rcmd_rpy, self.Rcmd_xyz)
        self.Rpsm.servo_cp(self.RT_IK)
        self.Rpsm.set_jaw_angle(self.Rleader.get_jaw_angle())

    def update_visual_markers(self):
        # Move the Target Position Based on the GUI
        if self.Lpsm.target_IK is not None:
            T_t_w = self.Lpsm.get_T_b_w() * self.LT_IK
            self.Lpsm.target_IK.set_pose(T_t_w)
        if self.Rpsm.target_IK is not None:
            T_t_w = self.Rpsm.get_T_b_w() * self.RT_IK
            self.Rpsm.target_IK.set_pose(T_t_w)

    def run(self):
        self.update_camera_pose()
        #print("<LYON> Camera Pose Updated")
        self.update_arm_pose()
        #print("<LYON> Arm Pose Updated")
        self.update_visual_markers()
        #print("<LYON> Visual Markers Updated")

    def scale_cb(self, scale):
        self.scale = [scale.data[0]*10, scale.data[1]*10]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', action='store', dest='client_name', help='Client Name', default='geomagic_sim_teleop')
    parser.add_argument('--one', action='store', dest='run_psm_one', help='Control PSM1', default=True)
    parser.add_argument('--two', action='store', dest='run_psm_two', help='Control PSM2', default=True)
    parser.add_argument('--three', action='store', dest='run_psm_three', help='Control PSM3', default=False)

    parsed_args = parser.parse_args()
    print('Specified Arguments')
    print(parsed_args)

    parsed_args.run_psm_one = get_boolean_from_opt(parsed_args.run_psm_one)
    parsed_args.run_psm_two = get_boolean_from_opt(parsed_args.run_psm_two)
    parsed_args.run_psm_three = get_boolean_from_opt(parsed_args.run_psm_three)

    simulation_manager = SimulationManager(parsed_args.client_name)

    cam = ECM(simulation_manager, 'CameraFrame')
    time.sleep(0.5)

    controllers = []
    psm_arms = []

    if parsed_args.run_psm_one is True:
        # Initial Target Offset for PSM1
        # init_xyz = [0.1, -0.85, -0.15]
        arm_name = 'psm1'
        print('LOADING CONTROLLER FOR ', arm_name)
        psm = PSM(simulation_manager, arm_name, add_joint_errors=False)
        if psm.is_present():
            T_psmtip_c = coordinate_frames.PSM1.T_tip_cam
            T_psmtip_b = psm.get_T_w_b() * cam.get_T_c_w() * T_psmtip_c
            psm.set_home_pose(T_psmtip_b)
            psm_arms.append(psm)

    if parsed_args.run_psm_two is True:
        # Initial Target Offset for PSM1
        # init_xyz = [0.1, -0.85, -0.15]
        arm_name = 'psm2'
        print('LOADING CONTROLLER FOR ', arm_name)
        psm = PSM(simulation_manager, arm_name, add_joint_errors=False)
        if psm.is_present():
            T_psmtip_c = coordinate_frames.PSM2.T_tip_cam
            T_psmtip_b = psm.get_T_w_b() * cam.get_T_c_w() * T_psmtip_c
            psm.set_home_pose(T_psmtip_b)
            psm_arms.append(psm)

    if parsed_args.run_psm_three is True:
        # Initial Target Offset for PSM1
        # init_xyz = [0.1, -0.85, -0.15]
        arm_name = 'psm3'
        print('LOADING CONTROLLER FOR ', arm_name)
        psm = PSM(simulation_manager, arm_name, add_joint_errors=False)
        if psm.is_present():
            T_psmtip_c = coordinate_frames.PSM3.T_tip_cam
            T_psmtip_b = psm.get_T_w_b() * cam.get_T_c_w() * T_psmtip_c
            psm.set_home_pose(T_psmtip_b)
            psm_arms.append(psm)

    if len(psm_arms) == 0:
        print('No Valid PSM Arms Specified')
        print('Exiting')

    else:
        Lleader = GeomagicDevice('/Geomagic_Left/')
        Rleader = GeomagicDevice('/Geomagic_Right/')
        theta_base = -0.9
        theta_tip = -theta_base
        Lleader.set_base_frame(Frame(Rotation.RPY(theta_base, 0, 0), Vector(0, 0, 0)))
        Lleader.set_tip_frame(Frame(Rotation.RPY(theta_base + theta_tip, 0, 0), Vector(0, 0, 0)))
        Rleader.set_base_frame(Frame(Rotation.RPY(theta_base, 0, 0), Vector(0, 0, 0)))
        Rleader.set_tip_frame(Frame(Rotation.RPY(theta_base + theta_tip, 0, 0), Vector(0, 0, 0)))
        print("<LYON> Leader Devices Created")
        controller = ControllerInterface(Lleader, Rleader, psm_arms, cam)
        controllers.append(controller)
        print("<LYON> Controller Created")
        rate = rospy.Rate(200)

        
        while not rospy.is_shutdown():
            for cont in controllers:
                    cont.Lleader.set_scale(cont.scale[0])
                    cont.Rleader.set_scale(cont.scale[1])
                    cont.run()
            rate.sleep()


