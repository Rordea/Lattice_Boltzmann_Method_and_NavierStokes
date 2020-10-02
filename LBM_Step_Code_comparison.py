#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.

#
# Program based on 2D flow around a cylinder python code modified to solve the general case of the Step
# in D2Q9 array for LBM
#
import os ; from os import * ; 
from numpy import *; from numpy.linalg import *
import matplotlib.pyplot as plt; from matplotlib import cm
#import cv2
import numpy as np
import glob
import datetime
from PIL import Image 
###### Flow definition #########################################################
 # Total number of time iterations.

seed = 10
rho = 1
lenx, leny = 3, 1
vis = 0.01 ####### Macro Variables setup
lref=leny
Re  = 200;  print("Macro Re: ",Re)
U = vis*Re/lref
uLB     = 0.01;    
nu = 0.01; 
nx = 300;   ny = int(0.85*nx);    ly=ny-1.0; q = 9;
uLB=(nu*Re)/ny
dx = lenx/nx ; dt = 0.0001# Lattice dimensions and populations.

frame = int(0.05/dt) ; 
SimTime = 3.5
maxIter = int(SimTime/(dt))

print("max timestep = ",maxIter)


nulb    = uLB*nx/Re; omega = ((nu)*(3*dt)/(dx**2))+0.5 

Ma = (dx/(lenx*(3**(1/2))))*(omega-0.5)*Re ; Ma_err = Ma**2    # Coordinates of step and radius size.


x=np.linspace(0,lenx,nx);     y=np.linspace(0,leny,ny);


################ Step Definition ###############################


SPx = 1.2   ; SPnx = int(SPx*nx/lenx)
SPy = 0    ; SPny = int(SPy*ny/leny)
SLx = 0.6   ; SLnx = int(SLx*nx/lenx)
SLy = 0.35  ; SLny = int(SLy*ny/leny)

cx = SPnx+(SLnx/2) ;   cy = SPny+(SLny/2)

expX = 110 ; expY = 110;  expR = 110

#radius=SLnx ; 
radius=SLy ; rad_nodes=int(radius*nx/lenx)

print("\nOmega: "+str(omega)+"\n"+"Ma: "+str(Ma)+"\n"+"Ma_error= "+str(Ma_err)+"  "+str(Ma_err*100)+"%"+"\n"+"nx: "+"\n"+str(nx)+"\n"+"ny: "+str(ny))
print("dt= "+str(dt)+"\n"+"mu= "+str(nu)+"\n"+"Total Elm: "+str(int(nx*ny))+"\n"+"Ulb = "+str(uLB)+"\n"+"Uentrada ="+
      str(U))
#Relaxation parameter.
os.system("pause")

################################ Post-process data definition before running Simulation ###########################

crpth=os.path.abspath(os.getcwd())
tv = "total_velocity" ; vxf = "Velocity_X" ; vyf = "Velocity_Y";  pr = "pressure" ###Folders names
cv = 'Convergence'

os.mkdir(str(tv)) ; os.mkdir(str(vxf)) ; os.mkdir(str(vyf)) ; os.mkdir(str(pr)) ; os.mkdir(str(cv)) #Creation of folder inside results directory
os.chdir(str(crpth))

###### Lattice Constants #######################################################
c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) # Lattice velocities.

t = 1./36. * ones(q)                                   # Lattice weights.
t[asarray([norm(ci)<1.1 for ci in c])] = 1./9.; t[0] = 4./9.

noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)] 
i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right wall. (f3,f6,f7)
i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle. (f2,f0,,f4)
i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left wall. (f1,f5,f8)

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

def equilibrium1(rho,u):              # Equilibrium distribution function.
    cu   = dot(c,u.transpose(1,0,2))
    rho_outlet = 1
    u_outlet = -1 +((sumpop(fin[i2,-1,:])+(2.*sumpop(fin[i3,-1,:])))/rho_outlet)
    usqr = 3./2.*(u_outlet**2+u[1]**2)
    feq1 = zeros((q,nx,ny))
    for i in range(q): feq1[i,:,:] = rho*t[i]*(1.+3*cu[i]+(3**2*0.5*cu[i]**2)-usqr)
    return feq1


###### Setup:  obstacle and velocity inlet with perturbation ########
UpperWall = fromfunction(lambda x,y: y==ny, (nx,ny))
LowerWall = fromfunction(lambda x,y: y==0, (nx,ny))
Step = fromfunction(lambda x,y: (((x-cx)**expX)/((SLnx/2)**expX))+(((y-cy)**expY)/(((SLny-1)/2)**expY))<1, (nx,ny))
vel = fromfunction(lambda d,x,y: (1-d)*uLB,(2,nx,ny))   ####### Change 1 to any lower value to change vx vy components.
#print("Initialized velocity values \n"); print(vel) ; os.system("pause")

a = datetime.datetime.now()

feq = equilibrium(1,vel); fin = feq.copy()
feq1 = equilibrium1(1, vel);


u_e_list = []
u_m_list = []
time_conv = []
u_conv = []
u_mag = []

macro_factor = (nx*vis)/(nu*rho*lref)
###### Main time loop ##########################################################
for time in range(maxIter):
    fin[i1,-1,:] = 1*fin[i1,-2,:]#-fin[i1,-3,:]
    
    
    
    rho = sumpop(fin)   ;  u = dot(c.transpose(), fin.transpose((1,0,2)))/rho # Calculate macroscopic density and velocity.
    
    # feq1 = equilibrium1(rho, u)
    # fin[i1,-1,:] = fin[i3,-1,:] - (feq1[i3,-1,:]-feq1[i1,-1,:])
    
    u[:,0,:] =vel[:,0,:] # Left wall: compute density from known populations.
    rho[0,:] = 1./(1.-u[0,0,:]) * (sumpop(fin[i2,0,:])+2.*sumpop(fin[i1,0,:]))
    feq = equilibrium(rho,u) # Left wall: Zou/He boundary condition.  
    fin[i3,0,:] = fin[i1,0,:] + (feq[i3,0,:]-feq[i1,0,:])
    

    # 
    
       #x #y
    #-fin[i1,:,-3] # Upper wall: outflow condition.
    # fin[i3,0,:] = 2*fin[i3,1,:]-fin[i3,2,:]
        
    #rho_LBM = 1.29 ;     u_outlet = -1 + (sumpop(fin[i2,-1,:]) + 2.*sumpop(fin[i1,-1,:]))/rho_LBM #Outlet velocity and density
        
    fout = fin*(1-omega) + (omega * feq)  # Collision step.
    #fout[:,-1,:] = fin[:,-1,:]*(1-omega) + (omega * feq1[:,-1,:])
    
    rho_o = sumpop(fout)   ;  u_o = dot(c.transpose(), fout.transpose((1,0,2)))/rho_o

    for i in range(q): 
        fout[i,UpperWall] = fin[noslip[i],UpperWall]
        fout[i,LowerWall] = fin[noslip[i],LowerWall]
        fout[i,Step] = fin[noslip[i],Step]
        
        
        
    for i in range(q): # Streaming step.
        fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)
        # fin[i,-1,:] = roll(roll(fout[i,-1,:],c[i,0],axis=0),c[i,1],axis=1)
        
    print("Timestep= ",time)
    
    
    u_e=macro_factor*np.max(np.abs(u_o[0]-u[0]))/np.max(np.abs(u[0]))
    u_m=macro_factor*np.max((u_o[0]))
    u_mv=macro_factor*np.max((u_o[1]))
    u_e_list.append(u_e)
    u_m_list.append(u_m)
    
    
    if (time%seed==0) and time>0:
        
        
        u_conv.append(np.average(u_e_list[-10:]))
        time_conv.append(time)
        fig = plt.figure()
        plt.plot(time_conv[seed:],u_conv[seed:])
        plt.savefig(str(tv)+"Convergence_u.png")
        plt.close()
        
        plt.clf()
        
        u_mag.append(np.average(u_m_list[-10:]))
        fig = plt.figure()
        plt.plot(time_conv[seed:],u_mag[seed:])
        plt.savefig(str(tv)+"Convergence_u_mag.png")
        plt.close()
        
        
        
        
        plt.clf()
        plt.plot(time_conv[-20:],u_conv[-20:])
        plt.savefig(str(cv)+"/Convergence"+str(time/seed).zfill(4)+str(time)+".png")
        plt.close()
        
        
    # plt.show()
    if (time%frame==0): # Visualization Frames
        fxs = 6
        fxy = (leny/lenx)*fxs
        fig = plt.figure(figsize=(fxs,fxy), dpi=100)
        
        X1, Y1 = np.meshgrid(x,y)
        cp=((((u[0]*macro_factor)**2)+((u[1]*macro_factor)**2))**(1/2)).transpose()
        plt.contourf(X1, Y1, cp, levels=60 ,cmap=cm.jet)
        plt.colorbar()
        plt.streamplot(X1, Y1, u[0].transpose(), u[1].transpose(),linewidth=0.5, color='black');
        plt.title("Velocidad [m/s]")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(str(tv)+"/vel_X_."+str(time/frame).zfill(4)+str(time)+".png")
        
        plt.clf()
        
        cp1=(rho/3).transpose()
        plt.contourf(X1, Y1, cp1, levels=60 ,cmap=cm.jet)
        plt.colorbar()
        plt.title("Presión [Pa]")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(str(pr)+"/pressure_."+str(time/frame).zfill(4)+str(time)+".png")
        
        
#        plt.imshow((((u[0]*macro_factor)**2)+((u[1]*macro_factor)**2))**(1/2).transpose(),cmap=cm.jet,extent=[x.min(), x.max(), y.min(), y.max()],origin='lower')
#        plt.title("Macro-scale velocity [m/s]");
#        
#        vx=plt.streamplot();
#        plt.imshow((u[0]*macro_factor).transpose(),cmap=cm.jet)
#        plt.title("Macro-scale velocity X [m/s]");   
#        plt.colorbar(vx)
#        plt.savefig(str(p)+"/"+str(vxf)+"/vel_X_."+str(time/frame).zfill(4)+str(time)+".png")
#        
#        vy=plt.streamplot(); 
#        plt.imshow((u[1]*macro_factor).transpose(),cmap=cm.jet)
#        plt.title("Macro-scale velocity Y [m/s]");   
#        plt.colorbar(vx)
#        plt.savefig(str(p)+"/"+str(vyf)+"/vel_X_."+str(time/frame).zfill(4)+str(time)+".png")
        
        

b = datetime.datetime.now()
c = b-a
print(c)
os.system('pause')
################# VIDEOS ##################################################

#img_array = []
#for filename in glob.glob(str(p)+"/"+str(tv)+'/*.png'):
#    img = cv2.imread(filename) ; height, width, layers = img.shape ; size = (width,height) ; img_array.append(img)
#out = cv2.VideoWriter(str(p)+"/"+str(tv)+'/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#for i in range(len(img_array)):
#    out.write(img_array[i])
#out.release()
#
#for filename in glob.glob(str(p)+"/"+str(vxf)+'/*.png'):
#    img = cv2.imread(filename) ; height, width, layers = img.shape ; size = (width,height) ; img_array.append(img)
#out = cv2.VideoWriter(str(p)+"/"+str(vxf)+'/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#for i in range(len(img_array)):
#    out.write(img_array[i])
#out.release()
#
#for filename in glob.glob(str(p)+"/"+str(vyf)+'/*.png'):
#    img = cv2.imread(filename) ; height, width, layers = img.shape ; size = (width,height) ; img_array.append(img)
#out = cv2.VideoWriter(str(p)+"/"+str(vyf)+'/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
#for i in range(len(img_array)):
#    out.write(img_array[i])
#out.release()
