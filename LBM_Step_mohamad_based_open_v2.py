#!/usr/bin/python
# Copyright (C) 2013 FlowKit Ltd, Lausanne, Switzerland
# E-mail contact: contact@flowkit.com
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.

#
# Program based on 2D flow around a cylinder python code modified to solve the general case of the Step
# in D2Q9 array for LBM
#
import os ; from os import * ; import shutil
from numpy import *; from numpy.linalg import *
import matplotlib.pyplot as plt; from matplotlib import cm
import cv2
import numpy as np
import glob
import datetime
###### Flow definition #########################################################
maxIter = 7000 # Total number of time iterations.
frame = 200 ; 
rho = 1.225
lenx, leny = 3, 1
radius,U,vis = 0.4,5,1.48e-5 ####### Macro Variables setup
lref=leny
Re      = lref*rho*U/vis; print("Macro Re: ",Re)
uLB     = 0.1;    vis_lbm = 1e-4 ; 
nx = int(Re*vis_lbm/(uLB*rho));   ny = int(leny*nx/lenx);    ly=ny-1.0; q = 9;
dx = lenx/nx ; dt = dx# Lattice dimensions and populations.
cx = nx/2; cy=ny;    rad_nodes=int(radius*nx/lenx); Ma = 0.22 ; Ma_err = Ma**2    # Coordinates of step and radius size.                   # Velocity in lattice units.
nulb    = uLB*nx/Re; omega = (Ma/((dx/(nx*((1/3)**2)))*Re))+0.5  #omega = ((vis/rho)*(3*dt)/(dx**2))+0.5 
print("\nOmega: "+str(omega)+"\n"+"Ma: "+str(Ma)+"\n"+"Ma_error= "+str(Ma_err)+"  "+str(Ma_err*100)+"%"+"\n"+"nx: "+"\n"+str(nx)+"\n"+"ny: "+str(ny))
print("dt= ",dt)
# Relaxation parameter.
os.system("pause")

################################ Post-process data definition before running Simulation ###########################

crpth=os.path.abspath(os.getcwd())
tv = "total_velocity" ; vxf = "Velocity_X" ; vyf = "Velocity_Y"; p = "results" ###Folders names

#shutil.rmtree(str(p)) #Comment if first time to run script, Uncomment if script was ran to overwrite results folder
os.mkdir(str(p)) ; os.chdir(str(p))  #Results folder definition
os.mkdir(str(tv)) ; os.mkdir(str(vxf)) ; os.mkdir(str(vyf)) #Creation of folder inside results directory
os.chdir(str(crpth))

###### Lattice Constants #######################################################
c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.

t = 1./36. * ones(q)                                   # Lattice weights.
t[asarray([norm(ci)<1.1 for ci in c])] = 1./9.; t[0] = 4./9.

noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)] 
i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right wall. (f1,f5,f6)
i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle. (f2,f0,,f4)
i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left wall. (f3,f6,f7)

i4 = arange(q)[asarray([ci[1]<0  for ci in c])] # f7,f4,f8
i5 = arange(q)[asarray([ci[1]>0  for ci in c])] # f6,f2,f5
###### Function Definitions ####################################################
sumpop = lambda fin: sum(fin,axis=0) # Helper function for density computation.
def equilibrium(rho,u):              # Equilibrium distribution function.
    cu   = dot(c,u.transpose(1,0,2))
    usqr = 3./2.*(u[0]**2+u[1]**2)
    feq = zeros((q,nx,ny))
    for i in range(q): feq[i,:,:] = rho*t[i]*(1.+3*cu[i]+(3**2*0.5*cu[i]**2)-usqr)
    return feq


###### Setup:  obstacle and velocity inlet with perturbation ########
#UpperWall = fromfunction(lambda x,y: y==0, (nx,ny))
LowerWall = fromfunction(lambda x,y: y==ny, (nx,ny))
Step = fromfunction(lambda x,y: (x-cx)**120+(y-cy)**120<rad_nodes**110, (nx,ny))
vel = fromfunction(lambda d,x,y: (1-d)*uLB,(2,nx,ny))   ####### Change 1 to any lower value to change vx vy components.
print("Initialized velocity values \n"); print(vel) ; os.system("pause")

a = datetime.datetime.now()

feq = equilibrium(rho,vel); fin = feq.copy()

###### Main time loop ##########################################################
for time in range(maxIter):
    fin[i5,:,0] = fin[i4,:,1] # Upper wall: outflow condition.
    rho = sumpop(fin)   ;  u = dot(c.transpose(), fin.transpose((1,0,2)))/rho # Calculate macroscopic density and velocity.

    u[:,0,:] =vel[:,0,:] # Left wall: compute density from known populations.
    rho[0,:] = 1./(1.-u[0,0,:]) * (sumpop(fin[i2,0,:])+2.*sumpop(fin[i1,0,:]))

    feq = equilibrium(rho,u) # Left wall: Zou/He boundary condition.
    
    fin[i3,0,:] = fin[i1,0,:] + feq[i3,0,:] - fin[i1,0,:]
    fout = fin - omega * (fin - feq)  # Collision step.
    for i in range(q): 
       # fout[i,UpperWall] = fin[noslip[i],UpperWall]
        fout[i,LowerWall] = fin[noslip[i],LowerWall]
        fout[i,Step] = fin[noslip[i],Step]
    for i in range(q): # Streaming step.
        fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)
        
    
    if (time%frame==0): # Visualization Frames
        macro_factor = (nx*vis)/(vis_lbm*lref)
        cp=plt.clf(); plt.imshow(sqrt(((u[0]*macro_factor)**2)+((u[1]*macro_factor)**2)).transpose(),cmap=cm.jet)
        plt.title("Macro-scale velocity [m/s]");   plt.colorbar(cp)
        plt.savefig(str(p)+"/"+str(tv)+"/vel."+str(time/frame).zfill(4)+str(time)+".png")
        
        vx=plt.clf(); plt.imshow((u[0]*macro_factor).transpose(),cmap=cm.jet)
        plt.title("Macro-scale velocity X [m/s]");   plt.colorbar(vx)
        plt.savefig(str(p)+"/"+str(vxf)+"/vel_X_."+str(time/frame).zfill(4)+str(time)+".png")
        
        vy=plt.clf(); plt.imshow((u[1]*macro_factor).transpose(),cmap=cm.jet)
        plt.title("Macro-scale velocity Y [m/s]");   plt.colorbar(vx)
        plt.savefig(str(p)+"/"+str(vyf)+"/vel_X_."+str(time/frame).zfill(4)+str(time)+".png")

################# VIDEOS ##################################################

img_array = []
for filename in glob.glob(str(p)+"/"+str(tv)+'/*.png'):
    img = cv2.imread(filename) ; height, width, layers = img.shape ; size = (width,height) ; img_array.append(img)
out = cv2.VideoWriter(str(p)+"/"+str(tv)+'/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

for filename in glob.glob(str(p)+"/"+str(vxf)+'/*.png'):
    img = cv2.imread(filename) ; height, width, layers = img.shape ; size = (width,height) ; img_array.append(img)
out = cv2.VideoWriter(str(p)+"/"+str(vxf)+'/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

for filename in glob.glob(str(p)+"/"+str(vyf)+'/*.png'):
    img = cv2.imread(filename) ; height, width, layers = img.shape ; size = (width,height) ; img_array.append(img)
out = cv2.VideoWriter(str(p)+"/"+str(vyf)+'/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
