##A Poisson Equation for pressure, which ensures that continuity is satisfied.
##for incompressible flow, there is no obvious way to couple pressure and velocity.
##So, take the divergence of the momentum equation and use the continuity equation to get a Poisson equation for pressure.
import numpy as np ; import matplotlib.pyplot as plt ; from matplotlib import cm
#import math
nx=181
ny=int(0.85*nx)
Lx=3 
Ly=1
dx=Lx/(nx-1) 
dy=Ly/(ny-1)
dt=0.001 ;
nt=325000 ; #timesteps
nu=0.01; 
rho=1
P = 6
frame = 100
uinf=2
x=np.linspace(0,Lx,nx);     y=np.linspace(0,Ly,ny);    X, Y = np.meshgrid(x,y)


u=np.ones((ny,nx))*uinf
v=np.zeros((ny,nx))
p=np.ones((ny,nx))
p[:,-1]=-1   #outlet pressure

for t in range (0,nt):
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
        p[:,-1]=-1 #outlet pressure
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
    v[0, :]  = 0
    v[-1, :] = 0
    v[:,-1]=v[:,-2]
    courantx=u[j,i]*dt/dx
    couranty=v[j,i]*dt/dy
    print('courantx',courantx)
    print('couranty',couranty)
  
#    if (t%frame==0):
#        cp=plt.clf(); 
#        fig = plt.figure(figsize=(6,6))
#        plt.imshow(u,cmap=cm.jet,origin='lower')
#        plt.title("Velocity [m/s]");   plt.colorbar(cp)
#        plt.show()

    #========================plotting the pressure field as a contour
#    plt.contourf(X, Y, u, cmap=cm.jet)
#    plt.colorbar()
    #====================== plotting the pressure field outlines
#    plt.contourf(X, Y, p, cmap=cm.jet)
#    plt.colorbar()
    #=======================plotting velocity field
    #plt.quiver(X[::1, ::1], Y[::1, ::1], u[::1, ::1], v[::1, ::1]) 
#    plt.streamplot(X, Y, u, v,color='k');
    plt.xlabel('X')
    plt.ylabel('Y');
    plt.imshow(u,cmap=cm.jet,origin='lower')
    plt.colorbar()
    plt.show()
    

