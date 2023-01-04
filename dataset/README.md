# Data Collection 

## Sensor Types: 
  1. Barometer from [Takkstrip](https://www.labs.righthandrobotics.com/takkstrip)
  2. [ADXL203 accelerometer](https://www.analog.com/media/en/technical-documentation/data-sheets/adxl103_203.pdf)
  
## Data types: 
  1. Pressure sensor data
  2. X and Y axis data from Accelerometer
  3. Gripper state data (opened/closed) 
  4. Number of squeezes
  
## ROS Packages
  1. [Schuck_ezn64](https://github.com/SmartRoboticSystems/schunk_grippers)
  2. [vibrotactile](https://github.com/togjade/Work_bundle/tree/main/granular_objects/ROS_Packages/vibrotactile)
  
## ROS Packages were built using CATKIN BUILD
## Conducting the Experiments 
  * Connect the gripper to the Power supply (24V).
  
1. Command to run on the Centos 7

		1.1. Turn on the computer -> choose advanced options for centos 7 -> choose second one from the top

		1.2. Open the terminal and run following commands in given order
    
				1.2.1. export ROS_MASTER_URI=http://10.1.71.79:11311        %%to connect to the master node that is running on the second Ubuntu PC
        
				1.2.2. export ROS _IP=10.1.70.233
        
				1.2.3. rosrun nidaq Togzhan6221 // to read the accelerometer data from NIDAQ
   
2. On the 2nd computer with Ubuntu ->

		2.1. In 1nd terminal:
    
				2.1.1. sudo –i  // enter the root 
        
				2.1.2. source /home/user/catkin_ws/devel/setup.bash
        
				2.1.3. roslaunch schunck_ezn64 ezn64_usb_control.launch   %%launch the Schunk
        
		2.2. In 2nd terminal:
    
				2.2.1. rosservice call /schunck_ezn64/reference           %%move the Schunk to the reference position (opened position)
        
		2.3. In 3rd terminal:
    
				2.3.1.roslaunch gripper
                
## Saving the data
* Open new terminal:
	* rosbag record /chatter

