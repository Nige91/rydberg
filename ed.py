# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:34:54 2020

@author: Nigel
"""

import numpy as np
import random as rnd
from numpy import linalg as LA
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
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
        return sp.kron(operator, sp.identity(2**(nParticles-1)))
    elif(nParticle == nParticles):
        return sp.kron(sp.identity(2**(nParticles-1)),operator)
    else:
        return sp.kron(sp.kron(sp.identity(2**(nParticle-1)),operator), sp.identity(2**(nParticles-nParticle)))

def createHamiltonianTransverse(factor, nParticles):
    result = sp.coo_matrix((2**nParticles, 2**nParticles))
    sigmax=sp.coo_matrix(np.matrix('0,1;1,0'))
    for i in range(nParticles):
        result += createSingleParticleOperator(sigmax, nParticles, i+1)
    return factor*result

def createHamiltonianLongitudinal(factor, nParticles):
    result = sp.coo_matrix((2**nParticles, 2**nParticles))
    sigmaz=sp.coo_matrix(np.matrix('1,0;0,-1'))
    for i in range(nParticles):
        result += createSingleParticleOperator(sigmaz, nParticles, i+1)
    return factor*result

def createHamiltonianNRydberg(factor, nParticles):
    result = sp.coo_matrix((2**nParticles, 2**nParticles))
    n_op=sp.coo_matrix(np.matrix('1,0;0,0'))
    for i in range(nParticles):
        result += createSingleParticleOperator(n_op, nParticles, i+1)
    return factor*result

def createHamiltonianInteraction(omega, r1ns, which, nParticles, limit=0):
    result = sp.coo_matrix((2**nParticles, 2**nParticles))
    op = sp.coo_matrix((2, 2))
    if which == "IsR1" or which == "IsNs":
        op = sp.coo_matrix(np.matrix('1,0;0,-1'))
    elif which == "RyR1" or which == "RyNs":
        op = sp.coo_matrix(np.matrix('1,0;0,0'))
    for i in range(nParticles):
        maxj = nParticles
        if limit!=0 and nParticles > i+2+limit:
            maxj = i+1+limit
        for j in range(i+1,maxj):
            factor=0
            if which == "IsR1":
                factor=calculateInteractionPauliR1(r1ns, j-i)
            elif which == "IsNs":
                factor=calculateInteractionPauliNs(omega, r1ns, j-i)
            elif which == "RyR1":
                factor=calculateInteractionRydbergR1(r1ns, j-i)
            elif which == "RyNs":
                factor=calculateInteractionRydbergNs(omega, r1ns, j-i)
            else:
                raise NameError('invalied Hamiltonian type:'+which)
            result += factor*createSingleParticleOperator(op, nParticles, i+1)*createSingleParticleOperator(op, nParticles, j+1)
    return result

def createFullHamiltonian(delta, omega, r1ns, which, limit, nParticles):
    result = sp.coo_matrix((2**nParticles, 2**nParticles))
    if which == "IsR1":
        result += createHamiltonianTransverse(calculateTFieldIsing(omega), nParticles)
        result += createHamiltonianLongitudinal(calculateLFieldIsingR1(delta,r1), nParticles)
        result += createHamiltonianInteraction(omega, r1ns, which, nParticles, limit)
    elif which == "IsNs":
        result += createHamiltonianTransverse(calculateTFieldIsing(omega), nParticles)
        result += createHamiltonianLongitudinal(calculateLFieldIsingNs(delta, omega, r1ns), nParticles)
        result += createHamiltonianInteraction(omega, r1ns, which, nParticles, limit)
    elif which == "RyR1":
        result += createHamiltonianTransverse(calculateTFieldRydberg(omega), nParticles)
        result += createHamiltonianLongitudinal(calculateLFieldRydberg(delta), nParticles)
        result += createHamiltonianInteraction(omega, r1ns, which, nParticles, limit)
    elif which == "RyNs":
        result += createHamiltonianTransverse(calculateTFieldRydberg(omega), nParticles)
        result += createHamiltonianLongitudinal(calculateLFieldRydberg(delta), nParticles)
        result += createHamiltonianInteraction(omega, r1ns, which, nParticles, limit)
    return result

def computeGroundState(H):
    eigvals, eigvecs = eigsh(H, 1, which="SA")
    return eigvecs[:,0]

def computeHighState(H):
    eigvals, eigvecs = eigsh(H, 1, which="LA")
    return eigvecs[:,0]

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

def computeSnapshotMatrix(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1ns, whichH, limit, nParticles):
    stepsDelta=logarithmicSteps(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicSteps(omegaStart, omegaEnd, nOmega)
    print("Steps Delta")
    print(stepsDelta)
    print("Steps Omega")
    print(stepsOmega)
    result = np.zeros(shape=(2**nParticles,nDelta*nOmega))
    for i in range(nDelta):
        for j in range(nOmega):
            result[:,i*nOmega + j]=computeGroundState(createFullHamiltonian(stepsDelta[i], stepsOmega[j], r1ns, whichH, limit, nParticles))
    return result

def computeRelErrorGrid(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, r1ns, whichH, limit, nParticles, B):
    stepsDelta=logarithmicStepsBetween(deltaStart, deltaEnd, nDelta)
    stepsOmega=logarithmicStepsBetween(omegaStart, omegaEnd, nOmega)
    print("Steps Delta Between")
    print(stepsDelta)
    print("Steps Omega Between")
    print(stepsOmega)
    result = np.zeros((nDelta-1, nOmega-1))
    for i in range(nDelta-1):
        for j in range(nOmega-1):
            H = createFullHamiltonian(stepsDelta[i], stepsOmega[j], r1ns, whichH, limit, nParticles)
            h = calculateReducedHamiltonian(H.toarray(), B)
            hGround = computeGroundState(h)
            HGroundAppr = np.dot(B,hGround)
            HGround = computeGroundState(H)
            error=np.zeros(2**nParticles)
            if abs(HGround[0]- HGroundAppr[0]) < abs(HGround[0]):
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

def computeAndCompareSVDApprox(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, r1ns, whichH, limit, nParticles, nSing):
    A = computeSnapshotMatrix(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1ns, whichH, limit, nParticles)
    u, s, v = LA.svd(A)
    print("Singular values")
    print(s)
    #nSing = findNumberSingularValues(s,minMagnS)
    print("Number of Singular Values considered")
    print(nSing)
    B = extractBFromU(u,nSing)
    errorGrid = computeRelErrorGrid(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, r1ns, whichH, limit, nParticles, B)
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
limit=8 #limits the interaction between particles to a certain range, since it scales with r^-6, it can definitely be neglected at a certain range
nParticles = 10
nSup=2.1 
r1= 20*10**-9 #spacing between atoms. Rb = 9 mum for omega = 2 pi * 2 MHz
nSing = 7 #number of singular values considered
whichH = "RyNs" #type of Hamiltonian
computeAndCompareSVDApprox(deltaStart, deltaEnd, deltaTest, nDelta, omegaStart, omegaEnd, omegaTest, nOmega, nSup, whichH, limit, nParticles, nSing)
# printEigVecWithBase(computeHighStateIsing(deltaStart, omegaStart, ns, limit, nParticles))
# calculateReducedHamiltonian(0,u,2)
# H = createFullRydbergHamiltonianNs(deltaStart, omegaStart, nSup,limit,nParticles)
# print("H done")
# nDim = H.shape[0]
# eigvals, eigvecs = eigsh(H, nEigv, which="SA")
# print(eigvals)
# printEigVecWithBase(eigvecs[:,eigvecs.shape[1]-1])
# u, s, v = LA.svd(np.transpose(eigvecs))
# print("eigvals")
# print(eigvals)
# print("eigvecs")
# print(eigvecs)
# print("eigvec 0")
# printEigVecWithBase(eigvecs[:,0])
