# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:34:54 2020

@author: Nigel
"""

import numpy as np
import random as rnd
from numpy import linalg as LA
from scipy.sparse.linalg import eigsh
import math

constantC = 6.678*(10**-42)

def calculateTFieldIsing(omega):
    return omega/2

def calculateLFieldIsingNs(delta,omega,ns):
    sumn=0
    for n in range(1,20):
        sumn += (ns/n)**6
    sumn *= (omega/2)
    return (sumn - delta/2)

def calculateLFieldIsingR1(delta,r1):
    sumn=0
    for n in range(1,20):
        sumn += (1/n)**6
    sumn *= (constantC/(2*(r1**6)))
    return (sumn - delta/2)

def calculateInteractionPauliNs(omega, ns, n):
    return (omega/4)*((ns/n)**6)

def calculateInteractionPauliR1(r1, n):
    return (constantC/(4*((r1*n)**6)))

def calculateTFieldRydberg(omega):
    return omega/2

def calculateLFieldRydberg(delta):
    return -delta

def calculateInteractionRydbergNs(omega, ns, n):
    return omega*((ns/n)**6)

def calculateInteractionRydbergR1(r1, n):
    return (constantC/((r1*n)**6))

def createSingleParticleOperator(operator, nParticles, nParticle):
    if(nParticle == 1):
        return np.kron(operator, np.identity(2**(nParticles-1)))
    elif(nParticle == nParticles):
        return np.kron(np.identity(2**(nParticles-1)),operator)
    else:
        return np.kron(np.kron(np.identity(2**(nParticle-1)),operator), np.identity(2**(nParticles-nParticle)))

def createHamiltonianTransverse(factor, nParticles):
    result = np.zeros([2**nParticles, 2**nParticles])
    sigmax=np.matrix('0,1;1,0')
    for i in range(nParticles):
        result += createSingleParticleOperator(sigmax, nParticles, i+1)
    return factor*result

def createHamiltonianLongitudinal(factor, nParticles):
    result = np.zeros([2**nParticles, 2**nParticles])
    sigmaz=np.matrix('1,0;0,-1')
    for i in range(nParticles):
        result += createSingleParticleOperator(sigmaz, nParticles, i+1)
    return factor*result

def createHamiltonianNRydberg(factor, nParticles):
    result = np.zeros([2**nParticles, 2**nParticles])
    n_op=np.matrix('1,0;0,0')
    for i in range(nParticles):
        result += createSingleParticleOperator(n_op, nParticles, i+1)
    return factor*result

def createHamiltonianInteractionPauliNs(omega, ns, nParticles, limit=0):
    result = np.zeros([2**nParticles, 2**nParticles])
    sigmaz=np.matrix('1,0;0,-1')
    for i in range(nParticles):
        maxj = nParticles
        if limit!=0 and nParticles > i+2+limit:
            maxj = i+1+limit
        for j in range(i+1,maxj):
            v=calculateInteractionPauliNs(omega, ns, j-i)
            result += v*createSingleParticleOperator(sigmaz, nParticles, i+1)*createSingleParticleOperator(sigmaz, nParticles, j+1)
    return result

def createHamiltonianInteractionPauliR1(r1, nParticles, limit=0):
    result = np.zeros([2**nParticles, 2**nParticles])
    sigmaz=np.matrix('1,0;0,-1')
    for i in range(nParticles):
        maxj = nParticles
        if limit!=0 and nParticles > i+2+limit:
            maxj = i+1+limit
        for j in range(i+1,maxj):
            v=calculateInteractionPauliR1(r1, j-i)
            result += v*createSingleParticleOperator(sigmaz, nParticles, i+1)*createSingleParticleOperator(sigmaz, nParticles, j+1)
    return result

def createHamiltonianInteractionRydbergNs(omega, ns, nParticles, limit=0):
    result = np.zeros([2**nParticles, 2**nParticles])
    n_op=np.matrix('1,0;0,0')
    for i in range(nParticles):
        maxj = nParticles
        if limit!=0 and nParticles > i+2+limit:
            maxj = i+1+limit
        for j in range(i+1,maxj):
            v=calculateInteractionRydbergNs(omega, ns, j-i)
            result += v*createSingleParticleOperator(n_op, nParticles, i+1)*createSingleParticleOperator(n_op, nParticles, j+1)
    return result

def createHamiltonianInteractionRydbergR1(r1, nParticles, limit=0):
    result = np.zeros([2**nParticles, 2**nParticles])
    n_op=np.matrix('1,0;0,0')
    for i in range(nParticles):
        maxj = nParticles
        if limit!=0 and nParticles > i+2+limit:
            maxj = i+1+limit
        for j in range(i+1,maxj):
            v=calculateInteractionRydbergR1(r1, j-i)
            result += v*createSingleParticleOperator(n_op, nParticles, i+1)*createSingleParticleOperator(n_op, nParticles, j+1)
    return result

def createFullIsingHamiltonianNs(delta, omega, ns, limit, nParticles):
    result = np.zeros([2**nParticles, 2**nParticles])
    result += createHamiltonianTransverse(calculateTFieldIsing(omega), nParticles)
    result += createHamiltonianLongitudinal(calculateLFieldIsingNs(delta,omega,ns), nParticles)
    result += createHamiltonianInteractionPauliNs(omega, ns, nParticles, limit)
    return result

def createFullIsingHamiltonianR1(delta, omega, r1, limit, nParticles):
    result = np.zeros([2**nParticles, 2**nParticles])
    result += createHamiltonianTransverse(calculateTFieldIsing(omega), nParticles)
    result += createHamiltonianLongitudinal(calculateLFieldIsingR1(delta,r1), nParticles)
    result += createHamiltonianInteractionPauliR1(r1, nParticles, limit)
    return result

def createFullRydbergHamiltonianNs(delta, omega, ns, limit, nParticles):
    result = np.zeros([2**nParticles, 2**nParticles])
    result += createHamiltonianTransverse(calculateTFieldRydberg(omega), nParticles)
    result += createHamiltonianNRydberg(calculateLFieldRydberg(delta), nParticles)
    result += createHamiltonianInteractionRydbergNs(omega, ns, nParticles, limit)
    return result

def createFullRydbergHamiltonianR1(delta, omega, r1, limit, nParticles):
    result = np.zeros([2**nParticles, 2**nParticles])
    result += createHamiltonianTransverse(calculateTFieldRydberg(omega), nParticles)
    result += createHamiltonianNRydberg(calculateLFieldRydberg(delta), nParticles)
    result += createHamiltonianInteractionRydbergR1(r1, nParticles, limit)
    return result

def computeGroundState(H):
    nDim = H.shape[0]
    eigvals, eigvecs = eigsh(H, nDim, which="SM")
    return eigvecs[:,0]

def computeHighState(H):
    nDim = H.shape[0]
    eigvals, eigvecs = eigsh(H, nDim, which="SM")
    return eigvecs[:,eigvecs.shape[1]-1]

def logarithmicSteps(start, end, nSteps):
    factor=(end/start)**(1/(nSteps-1))
    result=np.zeros(nSteps)
    for i in range(nSteps):
        result[i]=start*(factor**i)
    return result

def logarithmicStepsBetween(start, end, nSteps):
    factor=(end/start)**(1/(nSteps-1))
    result=np.zeros(nSteps-1)
    for i in range(nSteps-1):
        result[i]=start*(factor**(i+0.5))
    return result


def computeSnapshotMatrixIsingNs(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, ns, limit, nParticles):
    stepsDelta=logarithmicSteps(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicSteps(omegaStart, omegaEnd, nOmega)
    print("Steps Delta")
    print(stepsDelta)
    print("Steps Omega")
    print(stepsOmega)
    result = np.zeros(shape=(2**nParticles,nDelta*nOmega))
    for i in range(nDelta):
        for j in range(nOmega):
            result[:,i*nOmega + j]=computeGroundState(createFullIsingHamiltonianNs(stepsDelta[i], stepsOmega[j], ns, limit, nParticles))
    return result

def computeSnapshotMatrixIsingR1(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1, limit, nParticles):
    stepsDelta=logarithmicSteps(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicSteps(omegaStart, omegaEnd, nOmega)
    print("Steps Delta")
    print(stepsDelta)
    print("Steps Omega")
    print(stepsOmega)
    result = np.zeros(shape=(2**nParticles,nDelta*nOmega))
    for i in range(nDelta):
        for j in range(nOmega):
            result[:,i*nOmega + j]=computeGroundState(createFullIsingHamiltonianR1(stepsDelta[i], stepsOmega[j], r1, limit, nParticles))
    return result

def computeSnapshotMatrixRydbergNs(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, ns, limit, nParticles):
    stepsDelta=logarithmicSteps(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicSteps(omegaStart, omegaEnd, nOmega)
    print("Steps Delta")
    print(stepsDelta)
    print("Steps Omega")
    print(stepsOmega)
    result = np.zeros(shape=(2**nParticles,nDelta*nOmega))
    for i in range(nDelta):
        for j in range(nOmega):
            result[:,i*nOmega + j]=computeGroundState(createFullRydbergHamiltonianNs(stepsDelta[i], stepsOmega[j], ns, limit, nParticles))
    return result

def computeSnapshotMatrixRydbergR1(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1, limit, nParticles):
    stepsDelta=logarithmicSteps(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicSteps(omegaStart, omegaEnd, nOmega)
    print("Steps Delta")
    print(stepsDelta)
    print("Steps Omega")
    print(stepsOmega)
    result = np.zeros(shape=(2**nParticles,nDelta*nOmega))
    for i in range(nDelta):
        for j in range(nOmega):
            result[:,i*nOmega + j]=computeGroundState(createFullRydbergHamiltonianR1(stepsDelta[i], stepsOmega[j], r1, limit, nParticles))
    return result

def computeRelErrorGridIsingNs(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, var, limit, nParticles, B):
    stepsDelta=logarithmicStepsBetween(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicStepsBetween(omegaStart, omegaEnd, nOmega)
    print("Steps Delta Between")
    print(stepsDelta)
    print("Steps Omega Between")
    print(stepsOmega)
    result = np.zeros((nDelta-1, nOmega-1))
    for i in range(nDelta-1):
        for j in range(nOmega-1):
            H = createFullIsingHamiltonianNs(stepsDelta[i], stepsOmega[j], var, limit, nParticles)
            h = calculateReducedHamiltonian(H, B)
            hGround = computeGroundState(h)
            HGroundAppr = np.dot(B,hGround)
            HGround = computeGroundState(H)
            error=np.zeros(2**nParticles)
            if (HGround[0]- HGroundAppr[0]) < HGround[0]:
                error=HGround[0] - HGroundAppr[0]
            else:
                error=HGround[0] + HGroundAppr[0]
            result[i][j]=LA.norm(error)/LA.norm(HGround)
    return result

def computeRelErrorGridRydbergNs(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, var, limit, nParticles, B):
    stepsDelta=logarithmicStepsBetween(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicStepsBetween(omegaStart, omegaEnd, nOmega)
    print("Steps Delta Between")
    print(stepsDelta)
    print("Steps Omega Between")
    print(stepsOmega)
    result = np.zeros((nDelta-1, nOmega-1))
    for i in range(nDelta-1):
        for j in range(nOmega-1):
            H = createFullRydbergHamiltonianNs(stepsDelta[i], stepsOmega[j], var, limit, nParticles)
            h = calculateReducedHamiltonian(H, B)
            hGround = computeGroundState(h)
            HGroundAppr = np.dot(B,hGround)
            HGround = computeGroundState(H)
            error=np.zeros(2**nParticles)
            if (HGround[0]- HGroundAppr[0]) < HGround[0]:
                error=HGround[0] - HGroundAppr[0]
            else:
                error=HGround[0] + HGroundAppr[0]
            result[i][j]=LA.norm(error)/LA.norm(HGround)
    return result

def computeRelErrorGridIsingR1(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, var, limit, nParticles, B):
    stepsDelta=logarithmicStepsBetween(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicStepsBetween(omegaStart, omegaEnd, nOmega)
    print("Steps Delta Between")
    print(stepsDelta)
    print("Steps Omega Between")
    print(stepsOmega)
    result = np.zeros((nDelta-1, nOmega-1))
    for i in range(nDelta-1):
        for j in range(nOmega-1):
            H = createFullIsingHamiltonianR1(stepsDelta[i], stepsOmega[j], var, limit, nParticles)
            h = calculateReducedHamiltonian(H, B)
            hGround = computeGroundState(h)
            HGroundAppr = np.dot(B,hGround)
            HGround = computeGroundState(H)
            error=np.zeros(2**nParticles)
            if (HGround[0]- HGroundAppr[0]) < HGround[0]:
                error=HGround[0] - HGroundAppr[0]
            else:
                error=HGround[0] + HGroundAppr[0]
            result[i][j]=LA.norm(error)/LA.norm(HGround)
    return result

def computeRelErrorGridRydbergR1(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, var, limit, nParticles, B):
    stepsDelta=logarithmicStepsBetween(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicStepsBetween(omegaStart, omegaEnd, nOmega)
    print("Steps Delta Between")
    print(stepsDelta)
    print("Steps Omega Between")
    print(stepsOmega)
    result = np.zeros((nDelta-1, nOmega-1))
    for i in range(nDelta-1):
        for j in range(nOmega-1):
            H = createFullRydbergHamiltonianR1(stepsDelta[i], stepsOmega[j], var, limit, nParticles)
            h = calculateReducedHamiltonian(H, B)
            hGround = computeGroundState(h)
            HGroundAppr = np.dot(B,hGround)
            HGround = computeGroundState(H)
            error=np.zeros(2**nParticles)
            if (HGround[0]- HGroundAppr[0]) < HGround[0]:
                error=HGround[0] - HGroundAppr[0]
            else:
                error=HGround[0] + HGroundAppr[0]
            result[i][j]=LA.norm(error)/LA.norm(HGround)
    return result

def findNumberSingularValues(s, magnitude):
    for i in range(s.size):
        if s[i] < 10**(magnitude):
            return i
    return (s.size -1)

def extractBFromU(U,n):
    return np.split(U, [n], axis=1)[0]

def calculateReducedHamiltonian(H,B):
    return np.matmul(np.transpose(B),np.matmul(H,B))
                             
def createRandomState(nDim):
    result = np.empty(nDim)
    for i in range(nDim):
        result[i] = rnd.random()
    return result

def normalizeVector(v):
    total = 0
    for i in range(v.size):
        total += v[i]**2
    return v/total**0.5

    
def printEigVecWithBase(v):
    nDim = v.size
    for i in range(nDim):
        print(bin(i))
        print(v[i])

def computeAndCompareSVDApprox(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, var, limit, nParticles, nSing):
    A = computeSnapshotMatrixIsingNs(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, var, limit, nParticles)
    u, s, v = LA.svd(A)
    print("Singular values")
    print(s)
    #nSing = findNumberSingularValues(s,minMagnS)
    print("Number of Singular Values considered")
    print(nSing)
    B = extractBFromU(u,nSing)
    errorGrid = computeRelErrorGridIsingNs(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, var, limit, nParticles, B)
    maxErrorRel=np.amax(errorGrid)
    minErrorRel=np.amin(errorGrid)
    print("error grid")
    print(errorGrid)
    print("max relative error")
    print(maxErrorRel)
    print("min relative error")
    print(minErrorRel)
  
    
    
np.set_printoptions(linewidth=200)
omegaStart = 2*math.pi*1*10**6
omegaEnd = 2*math.pi*1*10**8
omegaTest = 2*math.pi*1*10**7
nOmega=8
deltaStart = 2*math.pi*100*10**6
deltaEnd = 2*math.pi*100*10**8
deltaTest = 2*math.pi*100*10**7
nDelta = 8
limit=8
nParticles = 8
nSup=2.1
r1= 20*10**-9 #Rb = 9 mum for omega = 2 pi * 2 MHz
minMagnS=-2
nSing = 7
computeAndCompareSVDApprox(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, nSup, limit, nParticles, nSing)
# printEigVecWithBase(computeHighStateIsing(deltaStart, omegaStart, ns, limit, nParticles))
# calculateReducedHamiltonian(0,u,2)
# H = createFullRydbergHamiltonianNs(deltaStart, omegaStart, ns,limit,nParticles)
# print("H done")
# nDim = H.shape[0]
# eigvals, eigvecs = eigsh(H, nDim, which="SM")
# print(eigvals)
# printEigVecWithBase(eigvecs[:,eigvecs.shape[1]-1])
# u, s, v = LA.svd(np.transpose(eigvecs))
# print("eigvals")
# print(eigvals)
# print("eigvecs")
# print(eigvecs)
# print("eigvec 0")
# printEigVecWithBase(eigvecs[:,0])
