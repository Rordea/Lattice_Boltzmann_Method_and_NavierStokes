##A Poisson Equation for pressure, which ensures t*
#hat continuity is satisfied.
##for incompressible flow, there is no obvious way to couple pressure and velocity.
##So, take the divergence of the momentum equation and use the continuity equation to get a Poisson equation for pressure.
import numpy as np ; import matplotlib.pyplot as plt ; from matplotlib import cm
#import math
nx=120
ny=150
Lx=3
Ly=3
dx=Lx/(nx-1)
dy=Ly/(ny-1)
dt=0.0001 ;
nt=500 #timesteps
nu=0.01
rho=1
P = 1
#frame = 50
x=np.linspace(0,Lx,nx);     y=np.linspace(0,Ly,ny);    X, Y = np.meshgrid(x,y)
u=np.zeros((ny,nx)) ; 
v=np.zeros((ny,nx));  
p=np.zeros((ny,nx))

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
        
        p[0, :]  = p[1, :]   # dp/dy = 0 at y = 0
        #p[-1, :] = p[-2, :]
        e=np.max(np.abs(p-pn))/np.max(np.abs(p))
    for j in range (1,ny-1):
        for i in range (1,nx-1):
            u[j,i] = (un[j,i] - (dt*un[j,i]/dx)*(un[j,i]-un[j,i-1]) - (dt*vn[j,i]/dy)*(un[j,i]-un[j-1,i]) 
                     + nu*dt*(((un[j,i+1]-2*un[j,i]+un[j,i-1])/(dx**2)) + ((un[j+1,i]-2*un[j,i]+un[j-1,i])/(dy**2))) 
                     - (dt/(dx*2*rho))*(p[j,i+1]-p[j,i-1]))+(dt*P)
               
            v[j,i] = (vn[j,i] - (dt*un[j,i]/dx)*(vn[j,i]-vn[j,i-1]) - (dt*vn[j,i]/dy)*(vn[j,i]-vn[j-1,i]) 
                     + nu*dt*(((vn[j,i+1]-2*vn[j,i]+vn[j,i-1])/(dx**2)) + ((vn[j+1,i]-2*vn[j,i]+vn[j-1,i])/(dy**2)))
                     - (dt/(dy*2*rho))*(p[j+1,i]-p[j-1,i]))
               
    u[0, :]  = 0
    u[-1, :] = 0    
    u[0:20, 40:70] = 0
    v[0:20, 40:70] = 0
    v[0, :]  = 0
    v[-1, :] = 0
    v[:,0]=0
    v[:,-1]=0
    
#    if (t%frame==0):
#        cp=plt.clf(); plt.imshow(u,cmap=cm.jet)
#        plt.title("Velocity [m/s]");   plt.colorbar(cp)
#        plt.savefig("vel."+str(t/frame).zfill(4)+str(t)+".png")

fig = plt.figure(figsize=(11,7), dpi=100)
#========================plotting the pressure field as a contour
plt.contourf(X, Y, u, cmap=cm.jet)
plt.colorbar()
#====================== plotting the pressure field outlines
#plt.contour(X, Y, p, cmap=cm.jet)
#=======================plotting velocity field
#plt.quiver(X[::1, ::1], Y[::1, ::1], u[::1, ::1], v[::1, ::1]) 
plt.streamplot(X, Y, u, v);
plt.xlabel('X')
plt.ylabel('Y');
plt.show()