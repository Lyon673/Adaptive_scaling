#!/bin/bash
SESSION_NAME="Adaptive_Scaling_Project"
SESSION_TARGET_ROS="${SESSION_NAME}:ros"
SESSION_TARGET_MAIN="${SESSION_NAME}:main"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "The session $SESSION_NAME existed,attaching now..."
    tmux attach -t $SESSION_NAME
    exit 0
fi

echo "creating new tmux session: $SESSION_NAME"

tmux new-session -d -s $SESSION_NAME -n "ros"
tmux split-window -h -t "${SESSION_TARGET_ROS}.0"
tmux split-window -v -t "${SESSION_TARGET_ROS}.0"
tmux split-window -v -t "${SESSION_TARGET_ROS}.2"


tmux new-window -t $SESSION_NAME -n "main"

tmux split-window -h -t "${SESSION_TARGET_MAIN}.0"
tmux split-window -v -t "${SESSION_TARGET_MAIN}.0"
tmux split-window -v -t "${SESSION_TARGET_MAIN}.2"


echo "Launching ROS components..."
# tmux send-keys -t "${SESSION_TARGET_ROS}.0" "roscore" C-m
# sleep 1 
tmux send-keys -t "${SESSION_TARGET_ROS}.0" "roslaunch geomagic_control geomagic_headless.launch device_name:=Right prefix:=Geomagic_Right" C-m
sleep 1 
tmux send-keys -t "${SESSION_TARGET_ROS}.1" "roslaunch geomagic_control geomagic_headless.launch device_name:=Left prefix:=Geomagic_Left" C-m
sleep 1 
tmux send-keys -t "${SESSION_TARGET_ROS}.2" "./env.sh" C-m
sleep 1
tmux send-keys -t "${SESSION_TARGET_ROS}.3" "python3 touch_control.py" C-m


echo "Launching main application..."

#tmux send-keys -t "${SESSION_TARGET_MAIN}.0" "./env.sh" C-m
#sleep 1 
#tmux send-keys -t "${SESSION_TARGET_MAIN}.1" "python3 touch_control.py" C-m
#tmux send-keys -t "${SESSION_TARGET_MAIN}.2" "python3 get_gaze.py" C-m
#sleep 1 
#tmux send-keys -t "${SESSION_TARGET_MAIN}.3" "python3 main.py" C-m

tmux select-window -t $SESSION_TARGET_ROS
tmux attach -t $SESSION_NAME
