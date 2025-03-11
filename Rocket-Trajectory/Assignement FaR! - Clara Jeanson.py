# -*- coding: utf-8 -*-
"""
Numerical calculation of rocket trajectory with air resistance.
In 2 dimensions, and with a time-dependent mass.

Created on Thu 07 Dec 2023 at 14:50:30.
Last modified [dd.mm.yyyy]: 12.03.2024
@author: bjarne.aadnanes.bergtun
"""

import numpy as np # maths
import matplotlib.pyplot as plt # plotting


# ========================= Constants & parameters ========================== #

# Constants
g = 9.81				# gravitational acceleration [m/s^2]
rho_0 = 1.225			# air density at sea level [kg/m^3]
H = 77000	        	# scale height [m]. Hint to 3a: np.inf = positive infinity
C_D = 0.51				# drag coefficient of the rocket [-]
A = 1.081e-2          	# rocket body frontal area [m^2]. "e-2" is short for "*10**(-2)"
m_0 = 19.765			# wet mass of the rocket [kg]
m_f = 11.269			# dry mass of the rocket [kg]
T_0 = 2501.8		    # average rocket engine thrust [N]
t_b = 6.09		    	# burn time [s]
theta_0 = 75*np.pi/180  # launch angle [rad]. Not used in 1D.


# Simulation parameters
dt = 0.001				# simulation time step [s]
#t_0 = 0                 # simulation start [s]; not needed when we start at 0
t_f = 180				# simulation time end [s]



# ================================ Functions ================================ #

def m(t):
    """
    Rocket mass [kg]
    as a function of time t [s]
    PS! Assumes 0 <= t <= t_b.
    """
    return m_0 - (m_0 - m_f) * t / t_b


def rho(y):
    """
    Air density [kg/m^3]
    as a function of altitude y [m]
    """
    return rho_0 * np.exp(-y/H)



def D_x(y, v, v_x):
    """
    Drag in the y-direction [N]
    as a function of altitude y [m], and velocity v, v_y [m/s]
    """
    return -(0.5 * C_D * A * rho(y) * v * v_x)

def D_y(y, v, v_y):
    """
    Drag in the y-direction [N]
    as a function of altitude y [m], and velocity v, v_y [m/s]
    """
    return -(0.5 * C_D * A * rho(y) * v * v_y)

def T_x(theta):

    return T_0*np.cos(theta)

def T_y(theta):

    return T_0*np.sin(theta)

def theta(xA, xB, yA, yB):
    return np.arctan( (yB-yA)/(xB-xA) )


# ======================== Numerical implementation ========================= #

# Calculate the number of data points in our simulation
N = int(np.ceil(t_f/dt))

#angle versus time
theta_i = np.zeros(N)


# Create data lists
# Except for the time list, all lists are initialized as lists of zeros.
t = np.arange(t_f, step=dt) # runs from 0 to t_f with step length dt
x = np.zeros(N)
y = np.zeros(N)
v_x = np.zeros(N)
v_y = np.zeros(N)
a_x = np.zeros(N)
a_y = np.zeros(N)

Speed = np.zeros(N)


# We will use while loops to iterate over our data lists. For this, we will use
# the auxillary variable n to keep track of which element we're looking at.
# The data points are numbered from 0 to N-1
n = 0
n_max = N - 1


# Burn phase
# ---------------------------------- #
# First, we iterate until the motor has finished burning, or until we reach the lists' end:
while t[n] < t_b and n < n_max:
    # Values needed for Euler's method
    # ---------------------------------- #
    if n==0 :
        theta_i[n] = theta_0
    else:
        theta_i[n] = theta(x[n],x[n-1],y[n],y[n-1])
    
    # Speed
    v = np.sqrt(v_y[n]**2+v_x[n]**2) # Powers, like a^2, is written a**2
    
    # Acceleration
    a_x[n] = ( T_x(theta_i[n]) + D_x(y[n], v, v_x[n]) )/ m(t[n])
    a_y[n] = ( T_y(theta_i[n]) + D_y(y[n], v, v_y[n]) )/ m(t[n]) - g
    
    Speed[n] = np.sqrt(v_y[n]**2+v_x[n]**2)
    
    # Euler's method:
    # ---------------------------------- #
    # Velocity
    v_x[0]=a_x[0]*dt
    v_y[0]=a_y[0]*dt
    
    v_x[n+1] = v_x[n] + a_x[n]*dt
    v_y[n+1] = v_y[n] + a_y[n]*dt
    
    # Position
    x[n+1] = x[n] + v_x[n]*dt
    y[n+1] = y[n] + v_y[n]*dt
    
    
    # Advance n with 1
    n += 1

#Burnout index
n_b = n

# Coasting phase
# ---------------------------------- #
# Then we iterate until the rocket has crashed, or until we reach the lists' end:
while y[n] >= 0 and n < n_max:
    # Values needed for Euler's method
    # ---------------------------------- # 

    theta_i[n] = theta(x[n],x[n-1],y[n],y[n-1])
    
    # Speed
    v = np.sqrt(v_y[n]**2 + v_x[n]**2)
    
    # Acceleration
    a_x[n] = D_x(y[n], v, v_x[n]) / m_f
    a_y[n] = D_y(y[n], v, v_y[n]) / m_f - g
    
    Speed[n] = np.sqrt(v_y[n]**2+v_x[n]**2)
    
    # Euler's method:
    # ---------------------------------- #
    # Position
    x[n+1] = x[n] + v_x[n]*dt
    y[n+1] = y[n] + v_y[n]*dt
    
    # Velocity
    v_x[n+1] = v_x[n] + a_x[n]*dt
    v_y[n+1] = v_y[n] + a_y[n]*dt
    
    
    # Advance n with 1
    n += 1
    
#spashdown index
n_sd = n-1


# When we exit the loops above, our index n has reached a value where the rocket
# has crashed (or it has reached its maximum value). Since we don't need the
# data after n, we redefine our lists to include only the points from 0 to n:
t = t[:n]
x = x[:n]
y = y[:n]
v_x = v_x[:n]
v_y = v_y[:n]
a_x = a_x[:n]
a_y = a_y[:n]
Speed = Speed[:n]


# ============================== Data analysis ============================== #

#Burnout
v_b = np.sqrt(v_y[n_b]**2+v_x[n_b]**2)

# Apogee
n_a = np.argmax(y) # Index at apogee
v_a = np.sqrt(v_y[n_a]**2+v_x[n_a]**2)


#Splashdown
v_sd = np.sqrt(v_y[n_sd]**2+v_x[n_sd]**2)


# =========================== Printing of results =========================== #

print('\n---------------------------------\n')
print('Burnout time:\t', np.round(t[n_b]), 's')
print('Burnout altitude:\t', np.round(y[n_b])/1000, 'km')
print('Burnout speed:   \t',np.round(v_b),'m/s')
print('\n---------------------------------\n')

print('\n---------------------------------\n')
print('Apogee time:\t', np.round(t[n_a]), 's')
print('Apogee altitude:\t', np.round(y[n_a])/1000, 'km')
print('Apogee speed:   \t',np.round(v_a),'m/s')
print('\n---------------------------------\n')


print('\n---------------------------------\n')
print('Splashdown time:\t', np.round(t[n_sd]), 's')
print('Splashdown altitude:\t', np.round(y[n_sd])/1000, 'km')
print('Splashdown speed:   \t',np.round(v_sd),'m/s')
print('\n---------------------------------\n')



# =========================== Plotting of results =========================== #

# Close all currently open figures, so we avoid mixing up old and new figures.
plt.close('all')

# Trajectory
schema1 = plt.figure('Trajectory')
schema1 = plt.plot(t, y)
schema1 = plt.xlabel("Time [s]")
schema1 = plt.ylabel("Altitude [m]")
schema1 = plt.grid(linestyle='--')

# Position
schema2 = plt.figure('Position')
schema2 = plt.plot(x, y)
schema2 = plt.xlabel("Distance [m]")
schema2 = plt.ylabel("Altitude [m]")
schema2 = plt.grid(linestyle='--')

# Speed
schema3 = plt.figure('Speed')
schema3 = plt.plot(t, Speed)
schema3 = plt.xlabel("Time [s]")
schema3 = plt.ylabel("Speed [m/s]")
schema3 = plt.grid(linestyle='--')

# Acceleration
plt.figure('Acceleration')
plt.plot(t, a_x)
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.grid(linestyle='--')

# Acceleration
plt.figure('Acceleration')
plt.plot(t, a_y)
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s²]")
plt.grid(linestyle='--')
plt.show()