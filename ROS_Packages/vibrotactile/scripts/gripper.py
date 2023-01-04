#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32, Int32, Header 
from sensor_msgs.msg import JointState
import numpy as np
import time
from std_msgs.msg import Float32MultiArray
from togzhan_m.msg import AP

message = AP()

arr = Float32MultiArray()
arr.data = []
velocityGlobal = np.float32()
positionGlobal = np.float32()
pressureGlobal = np.float32()
accelGlobal=np.float32()
accelGlobal2=np.float32()
state=np.float32()
trial=np.float32()

velocityGlobal = 0
positionGlobal = 0
accelGlobal = 0
accelGlobal2 = 0
pressureGlobal = 0
state = 0
trial = 0

def callbackPressure(pressure_msg):
    global pressureGlobal 
    pressureGlobal=pressure_msg.data

def callbackAccel(accel):
    global accelGlobal 
    accelGlobal=accel.data

def callbackAccel2(accel2):
    global accelGlobal2 
    accelGlobal2=accel2.data

def listener1():
    rospy.Subscriber("pressure", Float32, callbackPressure)

def listener2():
    rospy.Subscriber("nidaqTogzhan6221", Float32, callbackAccel)

def listener5():
    rospy.Subscriber("nidaqTog6221", Float32, callbackAccel2)

def talker():
    i = 60
    j = 0
    while not rospy.is_shutdown():
        global state
        global trial
        if (i%20) == 0 and (i%30) == 0:
            velocityGlobal = 80
            time.sleep(5)
        elif (i%20) == 0 and ((i-20)%30) == 0:
        	velocityGlobal = 60
        elif (i%20) == 0 and ((i-40)%30) == 0:
        	velocityGlobal = 40
        if (i%2) == 0:
            arr.data.insert(0, velocityGlobal) # closes state 1 
            arr.data.insert(1, 0) ### 0
            pub.publish(arr)
            arr.data[:]=[]
            state =1
            trial = j
            j = j+1
            i=i+1
            time.sleep(1)
        elif(i%2) == 1: 
            arr.data.insert(0, velocityGlobal) # opens state 2
            arr.data.insert(1, 20)
            pub.publish(arr)
            arr.data[:]=[]
            state=2
            
            i=i+1
            time.sleep(1)

            
        message.header.stamp = rospy.Time.now()
        message.accel = accelGlobal
        message.accel2 = accelGlobal2
        message.pressure = pressureGlobal
        message.state = state
        message.trial = trial
        pub2.publish(message)

        rate.sleep()
		
if __name__ == '__main__':
    rospy.init_node('listener1', anonymous=True)
    pub = rospy.Publisher('setData', Float32MultiArray, queue_size =100)
    pub2 = rospy.Publisher('chatter', AP, queue_size =100)
    rate = rospy.Rate(8000) #10hz
    listener1()
    listener2() 
    listener5()
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
    
