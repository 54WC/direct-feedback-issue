
# These IPython-specific commands
# tell the notebook to reload imported
# files every time you rerun code. So
# you can modify inertial_wheel_pendulum.py
# and then rerun this cell to see the changes.
#load_ext autoreload
#autoreload 2
from __future__ import division
from pydrake.all import (BasicVector, DiagramBuilder, FloatingBaseType,
                         LinearQuadraticRegulator, RigidBodyPlant,
                         RigidBodyTree, Simulator)
from inertial_wheel_pendulum import *
import math
import numpy as np
from IPython.display import HTML
from inertial_wheel_pendulum_visualizer import *
import matplotlib.pyplot as plt



# Make numpy printing prettier
np.set_printoptions(precision=3, suppress=True)

# Define the upright fixed point here.
uf = np.array([0.])
xf = np.array([math.pi, 0, 0, 0])
#xf = np.array([0, 0, 0, 0])

# Pendulum params. You're free to play with these,
# of course, but we'll be expecting you to use the
# default values when answering questions, where
# varying these values might make a difference.
m1 = 1. # Default 1
l1 = 1. # Default 1
m2 = 2. # Default 2
l2 = 2. # Default 2
r = 1.0 # Default 1
g = 10  # Default 10
input_max = 10
pendulum_plant = InertialWheelPendulum(
                                       m1 = m1, l1 = l1, m2 = m2, l2 = l2,
                                       r = r, g = g, input_max = input_max)

'''
    Code submission for 3.1:
    Edit this method in `inertial_wheel_pendulum.py`
    and ensure it produces reasonable A and B
    '''
A, B = pendulum_plant.GetLinearizedDynamics(uf, xf)

print('A',A)
print('B',B)



def create_reduced_lqr(A, B):
    '''
        Code submission for 3.3: Fill in the missing
        details of this function to produce a control
        matrix K, and cost-to-go matrix S, for the full
        (4-state) system.
        '''
    Q = np.zeros((3, 3))
    # Not clear what these gains will do,
    # but as long as Q is positive semidefinite
    # this should find a solution.
    Q1 = np.random.random((3, 3))
    Q = (Q1.T + Q1) # make symmetric and thus psd
    print('Q',Q)
    R = [1.]
    A=np.delete(A, 1, 0)
    A=np.delete(A,1,1)
    B=np.delete(B, 1, 0)
    print('A',A)
    print('B',B)
    K, S = LinearQuadraticRegulator(A, B, Q, R)
    
    K=np.insert(K,1,0,axis=1)
    S=np.insert(S,1,0,axis=1)
    S=np.insert(S,1,0,axis=0)
    
    # Refer to create_lqr() to see how invocations
    # to LinearQuadraticRegulator work.
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.delete.html
    # and
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.insert.html
    # might be useful for helping with the state reduction.
    #K = np.zeros((1, 4))
    #S = np.eye(4)
    return (K, S)

K, S = create_reduced_lqr(A, B)
print("K:",K)
print("S:",S)


def lqr_controller(x):
    # This should return a 1x1 u that is bounded
    # between -input_max and input_max.
    # Remember to wrap the angular values back to
    # [-pi, pi].
    # xf [ 3.142  0.     0.     0.   ]
    # uf [ 0.]
    # K [[-109.67     0.     -49.263   -0.233]]
    # xf [ 3.142  0.     0.     0.   ]
    # uf [ 0.]
    # K [[-109.67     0.     -49.263   -0.233]]
    # xf [ 3.142  0.     0.     0.   ]
    
    
    u = np.zeros((1, 1))
    global xf, uf, K,input_max
    u=uf-K.dot((x-xf).T)
    
    if u > input_max:
        u = input_max
    elif u<-input_max:
        u=-input_max
    
    '''
        Code submission for 3.3: fill in the code below
        to use your computed LQR controller (i.e. gain matrix
        K) to stabilize the robot by setting u appropriately.
        '''
    return u

# Run forward simulation from the specified initial condition
duration = 20.
x0 = np.array([3, 0, 0, 0])
input_log, state_log = RunSimulation(pendulum_plant,
                                   lqr_controller,
                                    x0=x0,
                             duration=duration)


# Visualize state and input traces
fig = plt.figure().set_size_inches(6, 6)
for i in range(4):
    plt.subplot(5, 1, i+1)
    plt.plot(state_log.sample_times(), state_log.data()[i, :])
    plt.grid(True)
    plt.ylabel("x[%d]" % i)
plt.subplot(5, 1, 5)
plt.plot(input_log.sample_times(), input_log.data()[0, :])
plt.ylabel("u[0]")
plt.xlabel("t")
plt.grid(True)

# Visualize the simulation
viz = InertialWheelPendulumVisualizer(pendulum_plant)
ani = viz.animate(input_log, state_log, 30, repeat=True)
plt.show()
#plt.close(viz.fig)
#HTML(ani.to_html5_video())

# The swingup controller should accept a state x,
# and return a control input u (a 1x1 numpy array)
# that respects the plant's input limits.


def swingup_controller(x):
    # Here's some useful things...
    global xf,uf,input_max,l1,g,m1,m2
    
    q = x[0:2]
    qd = x[2:4]
    
    (M, C, tauG,dtauG, B) = pendulum_plant.GetManipulatorDynamics(q, qd)
    
    qf = xf[0:2]
    qdf = xf[2:4]
    
    (Mf, Cf, tauGf,dtauGf, Bf) = pendulum_plant.GetManipulatorDynamics(qf, qdf)
    
    k=0.1
    Eq = 4.5*x[2]**2-50*math.cos(x[0])
    Ed=  4.5*xf[2]**2 - 50*math.cos(xf[0])
    
    #Eq = qd.dot(M.dot(qd))+(m1+m2)*g*l1*math.cos(q[0])
    #Ed = qdf.dot(Mf.dot(qdf))+(m1+m2)*g*l1*math.cos(qf[0])
    
    #print('Eq',Eq)
    #print('Ed',Ed)
    
    u=np.array([k*x[2]*(Eq-Ed)])
    
    if u > input_max:
        u = np.array([input_max])
    elif u <-input_max:
        u = np.array([-input_max])

    '''
        Code response for problem 3.8:
        Fill in this swingup controller to control the pendulum
        energy to the homonclinic orbit.
        (You'll probably want to ensure that your controller
        controls the velocity of the inertial wheel, too, using whatever
        means you think are reasonable.)
        '''
    #u = np.array([0.])

    return u

def combined_controller(x):
    '''
        Code response for problem 3.8:
        Fill in this combined controller to dispatch to either the
        swingup or lqr controller by checking the state against your
        calculated region of attraction.
        '''
    
    #gg=np.shape(V_samples)
    #aa=xf[0]-theta_width
    #cc=xf[2]-thetad_width
    #bb=2*theta_width/float(gg[0] )
    #dd=2*thetad_width/float(gg[1])
    #ee=xf[3]-theta2d_width
    #ff=2*theta2d_width/float(gg[2])
    #i=int((x[0]-aa)/float(bb))
    #j=int((x[2]-cc)/float(dd))
    #k=int((x[3]-ee)/float(ff))
    
    
    #if (i<gg[0]) and (j<gg[1]) and (k<gg[2]) and (i>=0) and (j>=0) and (k>=0) :
    #      if V_samples[i,j,k]< rho:
    #            u=lqr_controller(x)
    #u= swingup_controller(x)
    #          print('lqr')
    #else:
    #      u= swingup_controller(x)
    #    print('V', V_samples[i,j,k])
    #else:
    #u= swingup_controller(x)
    
    
    
    if abs(x[0]) >2.7:
        u=lqr_controller(x)
    else:
        if (x[0] <0.01) and (x[2]<0.001):
            u=10
        else:
            u= swingup_controller(x)
    return u

# Simulate!
#x0 = [2, 0, 0, 0]
#x0 = [0, 0, 0.1, 0]
#x0 = [math.pi, 0, 0, 0]
#x0 = [3*math.pi, 0, 0, 0]
#x0 = [0, -100, 0, 0]
x0 = [0, 0, 0, 0]



duration = 30
input_log, state_log = RunSimulation(pendulum_plant,
                                     combined_controller,
                                     x0 = x0,
                                     duration = duration)

# Plot traces of the results
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_size_inches(12,12)
for i in range(4):
    plt.subplot(5, 1, i+1)
    plt.plot(state_log.sample_times(), state_log.data()[i, :])
    plt.grid(True)
    plt.ylabel("x[%d]" % i)
plt.subplot(5, 1, 5)
plt.plot(input_log.sample_times(), input_log.data()[0, :])
plt.ylabel("u[0]")
plt.xlabel("t")
plt.grid(True)



from IPython.display import HTML
from inertial_wheel_pendulum_visualizer import *

viz = InertialWheelPendulumVisualizer(pendulum_plant)
ani = viz.animate(input_log, state_log, 30, repeat=True)
plt.show()
#plt.close(viz.fig)
#HTML(ani.to_html5_video())
