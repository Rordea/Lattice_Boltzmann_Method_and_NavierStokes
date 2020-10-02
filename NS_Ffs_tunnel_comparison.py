import numpy as np ; import matplotlib.pyplot as plt ; from matplotlib import cm
import os ; from os import *
#import math
rho=1
nx=120
ny=int(0.85*nx)
Lx=3 
Ly=1
nu=0.01; 
Re=200
uinf=Re*nu/Ly
print("rho=",rho)
print("nx=",nx)
print("ny=",ny)
print("nu=",nu)
print("Re=",Re)
print("uinf=",uinf)

dx=Lx/(nx-1) 
dy=Ly/(ny-1)
dt=0.001 ;
nt=3500 ; #timesteps

SPx = 1.2 # Step position x [ m ]
SPy = 0 # [ m ]
SLx = 0.6 # [ m ]
SLy = 0.35 # [ m ]
#P = 6
sframe=0.05
frame = sframe/dt
print("frame=",frame)

seed=10

crpth=os.path.abspath(os.getcwd())
tv = "total_velocity" ; vxf = "Velocity_X" ; vyf = "Velocity_Y"; pr = "pressure" ###Folders names
cv = 'Convergence'

os.mkdir(str(tv)) ; os.mkdir(str(vxf)) ; os.mkdir(str(vyf)) ; os.mkdir(str(pr)) ; os.mkdir(str(cv)) #Creation of folder inside results directory
os.chdir(str(crpth))

x=np.linspace(0,Lx,nx);     y=np.linspace(0,Ly,ny);    X, Y = np.meshgrid(x,y)

u=np.ones((ny,nx))*uinf
v=np.zeros((ny,nx))
p=np.ones((ny,nx))
p[:,-1]=-1

u_e_list=[]
u_m_list=[]
u_conv=[]
time_conv=[]
u_mag=[]

for time in range (0,nt):
    un=u.copy();  vn=v.copy()
    errorf=0.01;    e=1
    while e>errorf:
        pn=p.copy()
        for j in range (1,ny-1):
           for i in range (1,nx-1):
               p[j,i] = (((pn[j,i+1]+pn[j,i-1])*(dy**2) + (pn[j+1,i]+pn[j-1,i])*(dx**2)) / (2*(dx**2+dy**2))
                        -(rho*(dx**2)*(dy**2))/(2*(dx**2+dy**2)) *      (((un[j,i+1]-un[j,i-1])/(2*dx) + (vn[j+1,i]-vn[j-1,i])/(2*dy))/(dt)
                        -((un[j,i+1]-un[j,i-1])/(2*dx))**2
                        -2*((un[j+1,i]-un[j-1,i])/(2*dy))*((vn[j,i+1]-vn[j,i-1])/(2*dx))
                        -((vn[j+1,i]-vn[j-1,i])/(2*dy))**2))
        e=np.max(np.abs(p-pn))/np.max(np.abs(p))
        p[0, :]  = p[1, :]
        p[-1, :] = p[-2, :]
        p[:,-1]=-1
    for j in range (1,ny-1):
        for i in range (1,nx-1):
            u[j,i] = (un[j,i] - (dt*un[j,i]/dx)*(un[j,i]-un[j,i-1]) - (dt*vn[j,i]/dy)*(un[j,i]-un[j-1,i]) 
                     + nu*dt*(((un[j,i+1]-2*un[j,i]+un[j,i-1])/(dx**2)) + ((un[j+1,i]-2*un[j,i]+un[j-1,i])/(dy**2))) 
                     - (dt/(dx*2*rho))*(p[j,i+1]-p[j,i-1]))#+(dt*P)
               
            v[j,i] = (vn[j,i] - (dt*un[j,i]/dx)*(vn[j,i]-vn[j,i-1]) - (dt*vn[j,i]/dy)*(vn[j,i]-vn[j-1,i]) 
                     + nu*dt*(((vn[j,i+1]-2*vn[j,i]+vn[j,i-1])/(dx**2)) + ((vn[j+1,i]-2*vn[j,i]+vn[j-1,i])/(dy**2)))
                     - (dt/(dy*2*rho))*(p[j+1,i]-p[j-1,i]))
    u[:,0]=uinf          
    u[0, :]  = 0
    u[-1, :] = 0
    u[:,-1]=u[:,-2]
    u[int(SPy*ny/Ly):int(SPy*ny/Ly)+(int(SLy*ny/Ly)), (int(SPx*nx/Lx)):((int(SPx*nx/Lx))+(int(SLx*nx/Lx)))] = 0 #Step Definition
    v[0:35, (int(SPx*nx/Lx)):((int(SPx*nx/Lx))+(int(SLx*nx/Lx)))] = 0 #Step Definition
    v[0, :]  = 0
    v[-1, :] = 0
    v[:,-1]=v[:,-2]
    courantx=u[j,i]*dt/dx
    couranty=v[j,i]*dt/dy
    print('courantx',courantx)
    
    u_e=np.max(np.abs(u-un))/np.max(np.abs(u))
    u_m=np.max((u))
    u_e_list.append(u_e)
#    u_mv=np.max((u_o[1]))
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
            
        # plt.clf()
        # plt.plot(time_conv[-20:],u_conv[-20:])
        # plt.savefig(str(cv)+"/Convergence"+str(time/seed).zfill(4)+str(time)+".png")
        # plt.close()
    
    if (time%frame==0):
        fxs=6
        fxy=(Ly/Lx)*fxs
        fig = plt.figure(figsize=(fxs,fxy),dpi=100) 
                
        plt.contourf(X, Y, u, levels=60, cmap=cm.jet,origin='lower')
        plt.title("Velocidad [m/s]");plt.colorbar()
        plt.streamplot(X, Y, u, v,color='k',linewidth=1)
        plt.xlabel('X')
        plt.ylabel('Y');
        plt.savefig(str(tv)+"/vel_X_."+str(time/frame).zfill(4)+str(time)+".png")
        
        plt.clf()
        
        plt.contourf(X, Y, p, levels=60, cmap=cm.jet,origin='lower')
        plt.title("Presión [Pa]");plt.colorbar()
        plt.xlabel('X')
        plt.ylabel('Y');
        plt.savefig(str(pr)+"/pressure_."+str(time/frame).zfill(4)+str(time)+".png")

