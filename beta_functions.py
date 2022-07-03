from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from collections import OrderedDict


def DP_RK(f,x,y,xStop,h,tol): #Dormand-Prince method
 c2 = 1.0/5.0; c3 = 3.0/10.0; c4 = 4.0/5.0 #ci coefficients
 c5 = 8.0/9.0; c6 = 1.0; c7 = 1.0
 d51 = 35.0/384.0; d53 = 500.0/1113.0; d54 = 125.0/192.0 #d5l coefficients
 d55 = -2187.0/6784.0; d56 = 11.0/84.0
 d41 = 5179.0/57600.0; d43 = 7571.0/16695.0; d44 = 393.0/640.0 #d4l coefficients
 d45 = -92097.0/339200.0; d46 = 187.0/2100.0; d47 = 1.0/40.0
 a21 = 0.2
 a31 = 3.0/40.0; a32 = 9.0/40.0                                 #aij coefficients
 a41 = 44.0/45.0; a42 = -56.0/15.0; a43 = 32.0/9.0
 a51 = 19372.0/6561.0; a52 = -25360.0/2187.0; a53 = 64448.0/6561.0
 a54 = -212.0/729.0
 a61 = 9017.0/3168.0; a62 =-355.0/33.0; a63 = 46732.0/5247.0
 a64 = 49.0/176.0; a65 = -5103.0/18656.0
 a71 = 35.0/384.0; a73 = 500.0/1113.0; a74 = 125.0/192.0;
 a75 = -2187.0/6784.0; a76 = 11.0/84.0
 X = []
 Y = []
 X.append(x)
 Y.append(y)
 stopper = 0 #0 continue the calculation, 1 stop it
 k1 = h*f(x,y)
 for i in range(500):
  k2 = h*f(x + c2*h, y + a21*k1)
  k3 = h*f(x + c3*h, y + a31*k1 + a32*k2)
  k4 = h*f(x + c4*h, y + a41*k1 + a42*k2 + a43*k3)
  k5 = h*f(x + c5*h, y + a51*k1 + a52*k2 + a53*k3 + a54*k4)
  k6 = h*f(x + c6*h, y + a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5)
  k7 = h*f(x + c7*h, y + a71*k1 + a73*k3 + a74*k4 + a75*k5 + a76*k6)
  dy = d51*k1 + d53*k3 + d54*k4 + d55*k5 + d56*k6
  E = (d51 - d41)*k1 + (d53 - d43)*k3 + (d54 - d44)*k4 + (d55 - d45)*k5 + (d56 - d46)*k6 - d47*k6  #Local truncation error
  e = sqrt(np.sum(E**2)/len(y))  #Mean squared error
  hNext = 0.9*h*(tol/e)**0.2

  if e <= tol: #Accept calculation if it is within tolerance
   y = y + dy
   x=x+h
   X.append(x)
   Y.append(y)
   if stopper == 1: break #The calculation reached the end of the interval
   if abs(hNext) > 10.0*abs(h): hNext = 10.0*h #Restriction to avoid big values of h
   if (h > 0.0) == ((x + hNext) >= xStop): #Verify if the next step is the last one, to adjust h
    hNext = xStop - x
    stopper = 1
   k1 = k7*hNext/h
  else:  #Reduce the step when is not within tolerance
   if abs(hNext) < 0.1*abs(h): hNext = 0.1*h
   k1 = k1*hNext/h
  h = hNext
 return np.array(X),np.array(Y)

#Number of generations n_g
n_g=3.0

#b_l values for the SM
b_lM=np.zeros(3)
b_lM[0]=-(4.0/3.0)*n_g-(1.0/10.0)
b_lM[1]=(22.0/3.0)-(4.0/3.0)*n_g-(1.0/6.0)
b_lM[2]=11.0-(4.0/3.0)*n_g

#b_l values for the MSSM
b_lS=np.zeros(3)
b_lS[0]=-33.0/5.0
b_lS[1]=-1.0
b_lS[2]=3.0

#b_kl values for the SM
b_klM=np.array([[-(19.0/15.0)*n_g-(9.0/50.0) , -(1.0/5.0)*n_g-(3.0/10.0) , -(11.0/30.0)*n_g],\
               [-(3.0/5.0)*n_g-(9.0/10.0) , (136.0/3.0)-(49.0/3.0)*n_g-(13.0/6.0) , -(3.0/2.0)*n_g],\
               [-(44.0/15.0)*n_g , -4.0*n_g , 102.0-(76.0/3.0)*n_g]])

#b_kl values for the MSSM
b_klS=np.array([[-199.0/25.0 , -27.0/5.0 , -88.0/5.0],\
               [-9.0/5.0 , -25.0 , -24.0],\
               [-11.0/5.0 , -9.0 , -14.0]])

#GM(Q,g)=[g'1(Q),g'2(Q),g'3(Q)] function for the SM at 2 loops
def GM(Q,g):
  GM=np.zeros(3)
  for i in range(3):
    GM[i]=GM[i]-b_lM[i]*(((g[i])**3)/((16.0*pi**2)*(Q)))
  for i in range(3):
    for k in range(3):
      GM[i]=GM[i]-(b_klM[k,i]*(g[k]**2)*(g[i]**3))/(((16.0*pi**2)**2)*(Q))
  return GM

#GS(Q,g)=[g'1(Q),g'2(Q),g'3(Q)] function for the MSSM at 2 loops
def GS(Q,g):
  GS=np.zeros(3)
  for i in range(3):
    GS[i]=GS[i]-b_lS[i]*(((g[i])**3)/((16.0*pi**2)*(Q)))
  for i in range(3):
    for k in range(3):
      GS[i]=GS[i]-(b_klS[k,i]*(g[k]**2)*(g[i]**3))/(((16.0*pi**2)**2)*(Q))
  return GS


tol=1e-10 #Tolerance
#Initial conditions of the gauge couplings
ec=1.602176634e-19
mt=173.34 #Energy of the quark top mass (GeV)
g=np.array([sqrt(5.0/3.0)*(0.35940), 0.64754, 1.1666]) #Initial conditions of [g1(mt),g2(mt),g3(mt)]
Delta_g3=2.0*0.001*sqrt(pi/0.1179) #g3 error
Delta_g1=(Delta_g3/g[2])*(1/10)*g[0] #g1 error
Delta_g2=(Delta_g3/g[2])*(1/10)*g[1] #g2 error


#Initial conditions of [g1(mt)+-Deltag1,g2(mt),g3(mt)]
g1plus=np.array([sqrt(5.0/3.0)*(0.35940)+Delta_g1, 0.64754, 1.1666])
g1minus=np.array([sqrt(5.0/3.0)*(0.35940)-Delta_g1, 0.64754, 1.1666])

#Initial conditions of [g1(mt),g2(mt)+-Deltag2,g3(mt)]
g2plus=np.array([sqrt(5.0/3.0)*(0.35940), 0.64754+Delta_g2, 1.1666])
g2minus=np.array([sqrt(5.0/3.0)*(0.35940), 0.64754-Delta_g2, 1.1666])

#Initial conditions of [g1(mt),g2(mt),g3(mt)+-Deltag3]
g3plus=np.array([sqrt(5.0/3.0)*(0.35940), 0.64754, 1.1666+Delta_g3])
g3minus=np.array([sqrt(5.0/3.0)*(0.35940), 0.64754, 1.1666-Delta_g3])

Q1,G1=DP_RK(GM,mt,g,1e18,0.1,tol) #beta functions solutions for the SM at 2 loops
Q2,G2=DP_RK(GS,mt,g,1e18,0.1,tol) #beta functions solutions for the MSSM at 2 loops

#Errors of the beta functions solutions
Q3,G3=DP_RK(GS,mt,g1plus,1e18,0.1,tol)
Q4,G4=DP_RK(GS,mt,g1minus,1e18,0.1,tol)
Q5,G5=DP_RK(GS,mt,g2plus,1e18,0.1,tol)
Q6,G6=DP_RK(GS,mt,g2minus,1e18,0.1,tol)
Q7,G7=DP_RK(GS,mt,g3plus,1e18,0.1,tol)
Q8,G8=DP_RK(GS,mt,g3minus,1e18,0.1,tol)

def alpha_1M(Q): #Coupling constant alpha_1 for the SM at 1 loop
    alpha_1M=1.0/(4.0*pi*(g[0]**(-2)+(b_lM[0]/(8.0*pi**2))*log(Q/mt)))
    return alpha_1M

def alpha_2M(Q): #Coupling constant alpha_2 for the SM at 1 loop
    alpha_2M=1.0/(4.0*pi*(g[1]**(-2)+(b_lM[1]/(8.0*pi**2))*log(Q/mt)))
    return alpha_2M

def alpha_3M(Q): #Coupling constant alpha_3 for the SM at 1 loop
    alpha_3M=1.0/(4.0*pi*(g[2]**(-2)+(b_lM[2]/(8.0*pi**2))*log(Q/mt)))
    return alpha_3M

def alpha_1S(Q): #Coupling constant alpha_1 for the MSSM at 1 loop
    alpha_1S=1.0/(4.0*pi*(g[0]**(-2)+(b_lS[0]/(8.0*pi**2))*log(Q/mt)))
    return alpha_1S

def alpha_2S(Q): #Coupling constant alpha_2 for the MSSM at 1 loop
    alpha_2S=1.0/(4.0*pi*(g[1]**(-2)+(b_lS[1]/(8.0*pi**2))*log(Q/mt)))
    return alpha_2S

def alpha_3S(Q): #Coupling constant alpha_3 for the MSSM at 1 loop
    alpha_3S=1.0/(4.0*pi*(g[2]**(-2)+(b_lS[2]/(8.0*pi**2))*log(Q/mt)))
    return alpha_3S

alphaM=np.zeros((len(Q1),3)) #Coupling constants alpha=[alpha1,alpha2,alpha3] for the SM at 2 loops
for i in range(3):
  for j in range(len(Q1)):
    alphaM[j,i]=(G1[j,i]**2)/(4.0*pi)

alphaS=np.zeros((len(Q2),3)) #Coupling constants alpha=[alpha1,alpha2,alpha3] for the MSSM at 2 loops
for i in range(3):
  for j in range(len(Q2)):
    alphaS[j,i]=(G2[j,i]**2)/(4.0*pi)

#Errors of the coupling constants for the MSSM at 2 loops
DalphaS11=np.zeros((len(Q3),3))
for i in range(3):
  for j in range(len(Q3)):
    DalphaS11[j,i]=(G3[j,i]**2)/(4.0*pi)

DalphaS12=np.zeros((len(Q4),3))
for i in range(3):
  for j in range(len(Q4)):
    DalphaS12[j,i]=(G4[j,i]**2)/(4.0*pi)

DalphaS21=np.zeros((len(Q5),3))
for i in range(3):
  for j in range(len(Q5)):
    DalphaS21[j,i]=(G5[j,i]**2)/(4.0*pi)

DalphaS22=np.zeros((len(Q6),3))
for i in range(3):
  for j in range(len(Q6)):
    DalphaS22[j,i]=(G6[j,i]**2)/(4.0*pi)

DalphaS31=np.zeros((len(Q7),3))
for i in range(3):
  for j in range(len(Q7)):
    DalphaS31[j,i]=(G7[j,i]**2)/(4.0*pi)

DalphaS32=np.zeros((len(Q8),3))
for i in range(3):
  for j in range(len(Q8)):
    DalphaS32[j,i]=(G8[j,i]**2)/(4.0*pi)


fM1 = interpolate.interp1d(Q1, alphaM[:,0]) #alpha_1 interpolation of SM at 2 loops
fM2 = interpolate.interp1d(Q1, alphaM[:,1]) #alpha_2 interpolation of SM at 2 loops
fM3 = interpolate.interp1d(Q1, alphaM[:,2]) #alpha_3 interpolation of SM at 2 loops

fS1 = interpolate.interp1d(Q2, alphaS[:,0]) #alpha_1 interpolation of MSSM at 2 loops
fS2 = interpolate.interp1d(Q2, alphaS[:,1]) #alpha_2 interpolation of MSSM at 2 loops
fS3 = interpolate.interp1d(Q2, alphaS[:,2]) #alpha_3 interpolation of MSSM at 2 loops

#Interpolation of the errors of MSSM at 2 loops
fS11 = interpolate.interp1d(Q3, DalphaS11[:,0])
fS12 = interpolate.interp1d(Q4, DalphaS12[:,0])
fS21 = interpolate.interp1d(Q5, DalphaS21[:,1])
fS22 = interpolate.interp1d(Q6, DalphaS22[:,1])
fS31 = interpolate.interp1d(Q7, DalphaS31[:,2])
fS32 = interpolate.interp1d(Q8, DalphaS32[:,2])

x= np.logspace(log10(mt+1),18,1000) #Range of the solutions

CompM=np.zeros((len(Q1),3)) #Comparison between coupling constants of SM at 1 and 2 loops
CompS=np.zeros((len(Q2),3)) #Comparison between coupling constants of MSSM at 1 and 2 loops
for i in range(len(Q1)):
    CompM[i,0]=abs((alphaM[i,0]-alpha_1M(Q1[i]))/alphaM[i,0])
    CompM[i,1]=abs((alphaM[i,1]-alpha_2M(Q1[i]))/alphaM[i,1])
    CompM[i,2]=abs((alphaM[i,2]-alpha_3M(Q1[i]))/alphaM[i,2])

for i in range(len(Q2)):
    CompS[i,0]=abs((alphaS[i,0]-alpha_1S(Q2[i]))/alphaS[i,0])
    CompS[i,1]=abs((alphaS[i,1]-alpha_2S(Q2[i]))/alphaS[i,1])
    CompS[i,2]=abs((alphaS[i,2]-alpha_3S(Q2[i]))/alphaS[i,2])


#Coupling constants evolution in the SM
plt.plot(Q1,4.0*pi*(g[0]**(-2)+(b_lM[0]/(8.0*pi**2))*np.log(Q1/mt)),label=r'$\alpha^{-1}_{1-1L}$', color="dodgerblue",linestyle="--")
plt.plot(Q1,4.0*pi*(g[1]**(-2)+(b_lM[1]/(8.0*pi**2))*np.log(Q1/mt)),label=r'$\alpha^{-1}_{2-1L}$', color="darkorange",linestyle="--")
plt.plot(Q1,4.0*pi*(g[2]**(-2)+(b_lM[2]/(8.0*pi**2))*np.log(Q1/mt)),label=r'$\alpha^{-1}_{3-1L}$', color="forestgreen", linestyle="--")
plt.plot(Q1,1.0/alphaM[:,0],label=r'$\alpha^{-1}_{1-2L}$', color="dodgerblue")
plt.plot(Q1,1.0/alphaM[:,1],label=r'$\alpha^{-1}_{2-2L}$', color="darkorange")
plt.plot(Q1,1.0/alphaM[:,2],label=r'$\alpha^{-1}_{3-2L}$', color="forestgreen")
plt.ylabel(r'$\alpha^{-1}$')
plt.xlabel(r'$Q$ (GeV)')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()


#Coupling constants evolution in the MSSM
plt.plot(Q2,4.0*pi*(g[0]**(-2)+(b_lS[0]/(8.0*pi**2))*np.log(Q2/mt)),label=r'$\alpha^{-1}_{1-1L}$', color="dodgerblue",linestyle="--")
plt.plot(Q2,4.0*pi*(g[1]**(-2)+(b_lS[1]/(8.0*pi**2))*np.log(Q2/mt)),label=r'$\alpha^{-1}_{2-1L}$', color="darkorange",linestyle="--")
plt.plot(Q2,4.0*pi*(g[2]**(-2)+(b_lS[2]/(8.0*pi**2))*np.log(Q2/mt)),label=r'$\alpha^{-1}_{3-1L}$', color="forestgreen", linestyle="--")
plt.plot(Q2,1.0/alphaS[:,0],label=r'$\alpha^{-1}_{1-2L}$', color="dodgerblue")
plt.plot(Q2,1.0/alphaS[:,1],label=r'$\alpha^{-1}_{2-2L}$', color="darkorange")
plt.plot(Q2,1.0/alphaS[:,2],label=r'$\alpha^{-1}_{3-2L}$', color="forestgreen")
plt.ylabel(r'$\alpha^{-1}$')
plt.xlabel(r'$Q$ (GeV)')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()


#Comparison between coupling constants of SM at 1 and 2 loops
plt.plot(Q1,CompM[:,0],label=r'$\Delta\alpha_{1}$',color="dodgerblue")
plt.plot(Q1,CompM[:,1],label=r'$\Delta\alpha_{2}$', color="darkorange")
plt.plot(Q1,CompM[:,2],label=r'$\Delta\alpha_{3}$', color="forestgreen")
plt.xlabel(r'$Q$ (GeV)')
plt.ylabel(r'$\Delta\alpha_{i}$')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()


#Comparison between coupling constants of MSSM at 1 and 2 loops
plt.plot(Q2,CompS[:,0],label=r'$\Delta\alpha_{1}$',color="dodgerblue")
plt.plot(Q2,CompS[:,1],label=r'$\Delta\alpha_{2}$', color="darkorange")
plt.plot(Q2,CompS[:,2],label=r'$\Delta\alpha_{3}$', color="forestgreen")
plt.xlabel(r'$Q$ (GeV)')
plt.ylabel(r'$\Delta\alpha_{i}$')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()


#Comparison between coupling constants of SM and MSSM at 2 loops
plt.plot(x,np.abs((fM1(x)-fS1(x))/fM1(x)),label=r'$\delta_{1}$')
plt.plot(x,np.abs((fM2(x)-fS2(x))/fM2(x)),label=r'$\delta_{2}$')
plt.plot(x,np.abs((fM3(x)-fS3(x))/fM3(x)),label=r'$\delta_{3}$')
plt.xlabel(r'$Q$ (GeV)')
plt.ylabel(r'$\delta_{i}$')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()

#Zoom at the unification area of MSSM
plt.plot(Q2,1.0/alphaS[:,0],label=r'$\alpha^{-1}_{1}$', color="dodgerblue")
plt.plot(Q2,1.0/alphaS[:,1],label=r'$\alpha^{-1}_{2}$', color="darkorange")
plt.plot(Q2,1.0/alphaS[:,2],label=r'$\alpha^{-1}_{3}$', color="forestgreen")
plt.fill_between(x, 1/fS11(x), 1/fS12(x), color="dodgerblue", alpha=0.3)
plt.fill_between(x, 1/fS21(x), 1/fS22(x), color="darkorange", alpha=0.3)
plt.fill_between(x, 1/fS31(x), 1/fS32(x), color="forestgreen", alpha=0.3)
plt.xlim(1e16,1e17)
plt.ylim(22.5,25)
plt.ylabel(r'$\alpha^{-1}$')
plt.xlabel(r'$Q$ (GeV)')
plt.xscale("log")
plt.legend()
plt.grid()
plt.show()
