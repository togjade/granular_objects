#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32, Int32, Header 
from sensor_msgs.msg import JointState
import numpy as np
import time
from std_msgs.msg import Float64MultiArray, Float32MultiArray
from togzhan_m.msg import AP

message = AP()

pressureGlobal = np.float32()
accelGlobal=np.float32()
accelGlobal2=np.float32()
stateGlobal = np.int32()
trialGlobal=np.int32()
accelGlobal = 0
accelGlobal2 = 0
pressureGlobal = 0
stateGlobal = 0
trialGlobal = 0

def callbackPressure(pressure_msg):
    global pressureGlobal 
    pressureGlobal=pressure_msg.data
#    rospy.loginfo(rospy.get_caller_id() + "I heard %s", pressure_msg.data)

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

    while not rospy.is_shutdown():

        #message.header.stamp = rospy.Time.now()
        message.accel = accelGlobal
        message.accel2 = accelGlobal2
        message.pressure = pressureGlobal
        message.trial = trialGlobal
        message.state = stateGlobal
        pub.publish(message)
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('listener1', anonymous=True)
    pub = rospy.Publisher('chatter', AP, queue_size =100)
    rate = rospy.Rate(4000) #10hz
    listener1()
    listener2() 
    listener5() 
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()
    
