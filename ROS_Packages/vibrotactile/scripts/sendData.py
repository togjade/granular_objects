#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32, Int32, Header 
from sensor_msgs.msg import JointState

import numpy as np
import time
from std_msgs.msg import Float32MultiArray

arr = Float32MultiArray()
arr.data =[]
arr2 = Float32MultiArray()
arr2.data =[]
velocityGlobal = np.float32()
positionGlobal = np.float32()
stateGlobal = np.int32()
trialGlobal = np.int32()
velocityGlobal = 0
positionGlobal = 0

def talker():
	global stateGlobal
	global trialGlobal
	i = 60
	j = 0
	while not rospy.is_shutdown():
		if (i%20) == 0 and (i%30) == 0:
			velocityGlobal = 80
			time.sleep(7)

		elif (i%20) == 0 and ((i-20)%30) == 0:
			velocityGlobal = 60

		elif (i%20) == 0 and ((i-40)%30) == 0:
			velocityGlobal = 40
		
		if (i%2) == 0:
			arr.data.insert(0, velocityGlobal) # closes state 1 
			arr.data.insert(1, 0) ### 0
			pub.publish(arr)
			arr.data[:]=[]
			arr2.data.insert(0,1)
			arr2.data.insert(1,j)
			
			pub2.publish(arr2)
			arr2.data[:] = []
			i=i+1
			time.sleep(1)
			 #clear array

		elif(i%2) == 1: 
			arr.data.insert(0, velocityGlobal) # opens state 2
			arr.data.insert(1, 20)
			pub.publish(arr)
			arr.data[:]=[]
			arr2.data.insert(0,2)
			arr2.data.insert(1,j)
			j = j+1
			pub2.publish(arr2)
			arr2.data[:] = []
			i=i+1
			time.sleep(1)
			rate.sleep()
		
if __name__ == '__main__':
    rospy.init_node('publisher', anonymous=True)
    pub = rospy.Publisher('setData', Float32MultiArray, queue_size =100)
    pub2 = rospy.Publisher('state_trial', Float32MultiArray, queue_size =100)
    rate = rospy.Rate(8000) #10hz
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
    
