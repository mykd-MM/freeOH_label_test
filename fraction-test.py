import numpy as np
import math
import numba
import time
from collections import Counter
from numba import njit
from numba import types

@njit(["Tuple((float64[:], float64))(float64[:], float64[:], float64[:])"],cache=True)
def pbc_r2(i,j,cellsize):
    k=np.zeros((3),dtype=numba.float64)
    inv_cellsize = 1.0 / cellsize
    xdist = j[0]-i[0]
    ydist = j[1]-i[1]
    zdist = j[2]-i[2]

    k[0] =  xdist-cellsize[0]*np.rint(xdist*inv_cellsize[0])
    k[1] =  ydist-cellsize[1]*np.rint(ydist*inv_cellsize[1])
    k[2] =  zdist-cellsize[2]*np.rint(zdist*inv_cellsize[2])

    return k, k[0]**2+k[1]**2+k[2]**2


@njit(["Tuple((float64[:,:], float64[:]))(float64[:,:], float64[:,:], float64[:])"],cache=True)
def pbc_r2_vec(i,j,cellsize):
    #assert i.size == j.size
    r2=np.zeros((i.shape[0]),dtype=numba.float64)
    k=np.zeros((i.shape[0],3),dtype=numba.float64)
    inv_cellsize = 1.0 / cellsize
    for idx in numba.prange(i.shape[0]):

        xdist = j[idx,0]-i[idx,0]
        ydist = j[idx,1]-i[idx,1]
        zdist = j[idx,2]-i[idx,2]
    
        k[idx,0] =  xdist-cellsize[0]*np.rint(xdist*inv_cellsize[0])
        k[idx,1] =  ydist-cellsize[1]*np.rint(ydist*inv_cellsize[1])
        k[idx,2] =  zdist-cellsize[2]*np.rint(zdist*inv_cellsize[2])
     
        r2[idx]=k[idx,0]**2+k[idx,1]**2+k[idx,2]**2

    return k, r2


@njit(inline='always')
def get_min(v0, v1, v2, v3):
    vindexmin = 0
    vmin=v0
    if vmin > v1: vindexmin = 1; vmin = v1
    if vmin > v2: vindexmin = 2; vmin = v2
    if vmin > v3: vindexmin = 3; vmin = v3 
    return vindexmin

@njit(cache=True)
#@njit(["UniTuple(uint8[:], uint8[:], float)(float64[:,:], float64[:,:], float64, float64[:], bool_)"],cache=True)
def interface_hbonding_new(mol1Coord,mol2Coord,cos_HAngle,cellsize,is_angle=True):
    # Important lesson for the free-OH interfacial count:
    # Acceptor needs to be a len-2 1-d array with integer counter (0: no bonding)
    # Donor needs to a len-2 1-d array 
    # Donor array 0/1 counters needs to be opposite of Acceptor (for free-OH lifetime calculations)

    donor=np.ones((2),dtype=numba.uint8)
    acceptor=np.zeros((2),dtype=numba.uint8)
    r2O2H21=r2O2H22=r2O1H11=r2O1H12=0.0
    rIVec=np.zeros((5,3), dtype=numba.float64)
    rJVec=np.zeros((5,3), dtype=numba.float64)
    r2Vec=np.empty(5, dtype=numba.float64)
    cosAngle=1.0

    rIVec[0] = rIVec[1] = rIVec[2] = mol1Coord[0]
    rJVec[3] = mol1Coord[1]
    rJVec[4] = mol1Coord[2]

    for j in range(int(mol2Coord.shape[0]/3)):
       
        mol2Index=3*j

        rIVec[3] = rIVec[4] = rJVec[0] = mol2Coord[mol2Index]
        rJVec[1] = mol2Coord[mol2Index+1]
        rJVec[2] = mol2Coord[mol2Index+2]

        _,r2Vec=pbc_r2_vec(rIVec,rJVec,cellsize)               

        minkey=get_min(r2Vec[1],r2Vec[2],r2Vec[3],r2Vec[4])

        if minkey == 0:
           _,r2O2H21=pbc_r2(mol2Coord[mol2Index],mol2Coord[mol2Index+1],cellsize)
           cosAngleA1=(-r2Vec[1]+r2O2H21+r2Vec[0])/(2*math.sqrt(r2O2H21*r2Vec[0]))
           if cosAngleA1 > cos_HAngle:
              acceptor[0]+=1 
        
        if minkey == 1:
           _,r2O2H22=pbc_r2(mol2Coord[mol2Index],mol2Coord[mol2Index+2],cellsize)
           cosAngleA2=(-r2Vec[2]+r2O2H22+r2Vec[0])/(2*math.sqrt(r2O2H22*r2Vec[0]))
           if cosAngleA2 > cos_HAngle:
              acceptor[1]+=1

        if minkey == 2:
           _,r2O1H11=pbc_r2(mol1Coord[0],mol1Coord[1],cellsize)
           cosAngleD1=(-r2Vec[3]+r2O1H11+r2Vec[0])/(2*math.sqrt(r2O1H11*r2Vec[0]))
           if cosAngleD1 > cos_HAngle:
              donor[0]=0   
        
        if minkey == 3:
           _,r2O1H12=pbc_r2(mol1Coord[0],mol1Coord[2],cellsize)
           cosAngleD2=(-r2Vec[4]+r2O1H12+r2Vec[0])/(2*math.sqrt(r2O1H12*r2Vec[0]))
           if cosAngleD2 > cos_HAngle:
              donor[1]=0

    if is_angle: # angle between the relevant Z-vector and free O-H bond
       if donor[0] ==1 and donor [1] == 0:
          rO1H11,r2O1H11=pbc_r2(mol1Coord[0],mol1Coord[1],cellsize)
          cosAngle=np.sign(mol1Coord[0,2])*(rO1H11[2]/math.sqrt(r2O1H11))
       elif donor[0] ==0 and donor [1] == 1:
          rO1H12,r2O1H12=pbc_r2(mol1Coord[0],mol1Coord[2],cellsize)
          cosAngle=np.sign(mol1Coord[0,2])*(rO1H12[2]/math.sqrt(r2O1H12))
       else:
          cosAngle = 100.0
       
            
    return acceptor, donor, cosAngle

@njit(cache=True)
def freeoh_count_jit(coord=np.array([[]]),\
    molInterfaceIndex=np.array([]),\
    hNeighbourList=np.array([]),\
    topol=np.array([[]]),\
    cos_HAngle=0.0,\
    cellsize=np.array([]),\
    is_orig_def=False,is_new_def=True):

    NAtomsMol=3                                                                     #No. of atoms in a molecule
    _M=molInterfaceIndex.shape[0]
    _N=hNeighbourList.shape[1]

    acceptorArray=np.zeros((_M,2),dtype=numba.int64)
    donorArray=np.zeros((_M,2),dtype=numba.int64)
    cosAngle=np.zeros(_M,dtype=np.float64)
    neighAtoms=np.zeros(_M,dtype=numba.int64)
    mol1Coord=np.zeros((NAtomsMol,3),dtype=np.float64)
    mol2Coord=np.zeros((_N*NAtomsMol,3),dtype=np.float64)

    labelArray=np.empty(_M, dtype="U10")
    for i in range(_M):                                       # loop over selected molecules
 
        mol1Coord[0]=coord[topol[molInterfaceIndex[i],0]]                         # extract center molecule
        mol1Coord[1]=coord[topol[molInterfaceIndex[i],1]]                         # extract center molecule
        mol1Coord[2]=coord[topol[molInterfaceIndex[i],2]]                         # extract center molecule
        for indexJ,j in enumerate(hNeighbourList[i]):
            mol2Coord[0+NAtomsMol*indexJ]=coord[topol[j,0]]                       # extract neighbors   
            mol2Coord[1+NAtomsMol*indexJ]=coord[topol[j,1]]                       # extract neighbors   
            mol2Coord[2+NAtomsMol*indexJ]=coord[topol[j,2]]                       # extract neighbors   

        neighAtoms[i] = len(np.array([index for index in hNeighbourList[i] if index!=-1]))*NAtomsMol # get actual number of neighbor atoms
        if is_orig_def:
           pass
        elif is_new_def:
           acceptorArray[i],donorArray[i],cosAngle[i]=interface_hbonding_new(mol1Coord,mol2Coord[:neighAtoms[i]],cos_HAngle,cellsize)
           labelArray[i]="D"*np.abs(2-np.sum(donorArray[i]))+"A"*np.sum(acceptorArray[i])
    #freeOHMask[np.where(cosAngle > 1.0)] = False
    del_index=np.argwhere(cosAngle > 1.0)
    freeOHCos=np.delete(cosAngle, del_index.flatten())

    return acceptorArray, donorArray, labelArray, freeOHCos

@numba.guvectorize(["void(float64[:],float64[:],float64[:],float64[:])"],
             "(m)->(m),(m),(m)",nopython=True,cache=True)
def freeOH_cosine_vec(cosAngle,cos2Angle,cos3Angle,Angle):
    for i in range(cosAngle.shape[0]):
        Angle[i]=np.arccos(cosAngle[i])  #in radians
        cos2Angle[i]=cosAngle[i]**2
        cos3Angle[i]=cosAngle[i]**3


def main():
    #sample data
    molInterfaceIndex=np.array([  2,   7,   8,  12,  15,  17,  25,  26,  27,  31,  32,  38,  47,  48,  51,  56,  57,  62,
      65,  66,  68,  69,  75,  80,  81,  98, 101, 106, 107, 110, 111, 113, 122, 124, 126, 129,
     140, 141, 142, 143, 144, 152, 153, 154, 165, 166, 173, 175, 177, 184, 187, 188, 190, 195,
     200, 201, 205, 208, 212, 214, 217, 220, 224, 225, 229, 230, 243, 247, 248, 254, 261, 262,
     268, 270, 274, 278, 282, 283, 284, 285, 288, 289, 290, 292, 299, 300, 302, 303, 308, 311,
     312, 324, 325, 332, 333, 337, 338, 345, 350, 361, 364, 370, 373, 377, 381, 383, 384, 389,
     400, 404, 407, 411, 416, 422, 423, 424, 430, 432, 433, 435, 447, 453, 456, 458, 461, 463,
     472, 478, 481, 482, 489, 492, 496, 497, 505, 513, 518, 522, 523, 526, 527, 534, 535, 542,
     543, 544, 546, 547, 562, 563, 565, 568, 570, 572, 576, 577, 579, 587, 588, 593, 594, 595,
     600, 609, 617, 622, 623, 626, 633, 636, 637, 639])
    
    #read data
    coord=np.loadtxt('coord-test.out', dtype=np.float64, delimiter=",")
    hNeighbourList=np.loadtxt('hNeighbourList-test.out', delimiter=",").astype(np.int64)
    
    #const
    cos_HAngle=np.cos(50.0*np.pi/180)
    cellsize=np.array([26.40,26.40,70])
    
    #standard for HOH
    topol=np.zeros((640,3),dtype=int)
    for i in range(640):
        for j in range(3):
            topol[i,j]=i*3+j
    
    interfacialLabels = ['DA', 'DDA', 'DAA']
    _,_,labelArray,freeOHCos=freeoh_count_jit(coord,molInterfaceIndex,hNeighbourList,topol,cos_HAngle,cellsize)
    hbondDict=Counter(labelArray)
    interfacialOH = {k: hbondDict[k] for k in interfacialLabels}   
    cos2Vec,cos3Vec,angleVec=freeOH_cosine_vec(freeOHCos)
    
    print(interfacialOH)
    print("cos: {:.5g}; cos^2: {:.5g}; cos^3: {:.5g}; Angle: {:.4g}".format(np.mean(freeOHCos),\
         np.mean(cos2Vec),\
         np.mean(cos3Vec),\
         np.mean(angleVec)*180/np.pi))
    
    timesave=[]
    for i in range(100):
        start = time.perf_counter()
        freeoh_count_jit(coord,molInterfaceIndex,hNeighbourList,topol,cos_HAngle,cellsize)
        timesave.append(time.perf_counter()-start)
    print("freeoh_count_jit took {:.3g} ms".format(np.mean(timesave[:1])*1000.0))

if __name__ == "__main__":
    main()