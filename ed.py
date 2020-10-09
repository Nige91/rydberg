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
import matplotlib.pyplot as plt
import time

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
                # print(i)
                # print(j)
                # print(factor)
                # print(calculateLFieldRydberg(delta))
            elif which == "RyNs":
                factor=calculateInteractionRydbergNs(omega, r1ns, j-i)
            else:
                raise NameError('invalied Hamiltonian type:'+which)
            result += factor*createSingleParticleOperator(op, nParticles, i+1).dot(createSingleParticleOperator(op, nParticles, j+1))
    return result

def createFullHamiltonian(delta, omega, r1ns, which, limit, nParticles):
    result = sp.coo_matrix((2**nParticles, 2**nParticles))
    if which == "IsR1":
        result += createHamiltonianTransverse(calculateTFieldIsing(omega), nParticles)
        result += createHamiltonianLongitudinal(calculateLFieldIsingR1(delta,r1ns), nParticles)
        result += createHamiltonianInteraction(omega, r1ns, which, nParticles, limit)
    elif which == "IsNs":
        result += createHamiltonianTransverse(calculateTFieldIsing(omega), nParticles)
        result += createHamiltonianLongitudinal(calculateLFieldIsingNs(delta, omega, r1ns), nParticles)
        result += createHamiltonianInteraction(omega, r1ns, which, nParticles, limit)
    elif which == "RyR1":
        result += createHamiltonianTransverse(calculateTFieldRydberg(omega), nParticles)
        result += createHamiltonianNRydberg(calculateLFieldRydberg(delta), nParticles)
        result += createHamiltonianInteraction(omega, r1ns, which, nParticles, limit)
    elif which == "RyNs":
        result += createHamiltonianTransverse(calculateTFieldRydberg(omega), nParticles)
        result += createHamiltonianNRydberg(calculateLFieldRydberg(delta), nParticles)
        result += createHamiltonianInteraction(omega, r1ns, which, nParticles, limit)
    return result

def computeGroundState(H):
    eigvals, eigvecs = eigsh(H, 1, which="SA")
    return eigvecs[:,0]

def computeHighState(H):
    eigvals, eigvecs = eigsh(H, 1, which="LA")
    return eigvecs[:,0]

def linearSteps(start, end, nSteps):
    result=np.zeros(nSteps)
    stepSize = (end-start)/(nSteps-1)
    for i in range(nSteps):
        result[i]=start + i*stepSize
    return result

def linearStepsBetween(start, end, nSteps):
    result=np.zeros(nSteps-1)
    stepSize = (end-start)/(nSteps-1)
    for i in range(nSteps-1):
        result[i]=start + (i+0.5)*stepSize
    return result

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

def computeSnapshotMatrix(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1nsStart, r1nsEnd, nR1ns, whichH, limit, nParticles):
    stepsDelta=linearSteps(deltaStart, deltaEnd, nDelta)
    stepsOmega=linearSteps(omegaStart, omegaEnd, nOmega)
    stepsR1ns = linearSteps(r1nsStart, r1nsEnd, nR1ns)
    # print("Steps Delta")
    # print(stepsDelta)
    # print("Steps Omega")
    # print(stepsOmega)
    result = np.zeros(shape=(2**nParticles,nDelta*nOmega*nR1ns))
    for i in range(nDelta):
        for j in range(nOmega):
            for k in range(nR1ns):
                result[:,i*nOmega*nR1ns + j*nR1ns + k]=computeGroundState(createFullHamiltonian(stepsDelta[i], stepsOmega[j], stepsR1ns[k], whichH, limit, nParticles))
    return result

def computeRelErrorGrid(HGrid, GSGrid, nDelta, nOmega, nR1ns, B, nParticles):
    result = np.zeros((nDelta-1, nOmega-1, nR1ns - 1))
    for i in range(nDelta-1):
        for j in range(nOmega-1):
            for k in range(nR1ns-1):
                # h = calculateReducedHamiltonian(HGrid[i][j][k], B)
                # hGround = computeGroundState(h)
                # HGroundAppr = np.dot(B,hGround)
                try:
                    HGroundAppr = computeGroundStateSVD(HGrid[i][j][k], B)
                except:
                    print("error")
                    print("ndelta, nomega, nr1ns")
                    print(i)
                    print(j)
                    print(k)
                    print(B.shape)
                HGround = GSGrid[i][j][k]
                error=np.zeros(2**nParticles)
                if abs(HGround[0]- HGroundAppr[0]) < abs(HGround[0]):
                    error=HGround[0] - HGroundAppr[0]
                else:
                    error=HGround[0] + HGroundAppr[0]
                result[i][j][k]=LA.norm(error)/LA.norm(HGround)
    return result

def extractBFromU(U,n):
    return np.split(U, [n], axis=1)[0]

def calculateReducedHamiltonian(H,B):
    return np.matmul(np.transpose(B),np.matmul(H,B))

def calculateHTestGrid(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1nsStart, r1nsEnd, nR1ns, whichH, limit, nParticles):
    stepsDelta=linearStepsBetween(deltaStart, deltaEnd, nDelta)
    stepsOmega=linearStepsBetween(omegaStart, omegaEnd, nOmega)
    stepsR1 = linearStepsBetween(r1nsStart, r1nsEnd, nR1ns)
    # print("Steps Delta Between")
    # print(stepsDelta)
    # print("Steps Omega Between")
    # print(stepsOmega)
    HGrid=np.zeros((nDelta - 1, nOmega - 1, nR1ns - 1, 2**nParticles, 2**nParticles))
    GSGrid=np.zeros((nDelta - 1, nOmega - 1, nR1ns - 1, 2**nParticles))
    for i in range(nDelta-1):
        for j in range(nOmega-1):
            for k in range(nR1ns - 1):
                H = createFullHamiltonian(stepsDelta[i], stepsOmega[j], stepsR1[k], whichH, limit, nParticles)
                HGrid[i][j][k] = H.toarray()
                GSGrid[i][j][k] = computeGroundState(H)
    return HGrid, GSGrid

def plotSingularValueAccuracy(nSingVals, maxErrors, whichH, nParticles):
    if whichH == "IsR1":
        title = "Accuracy in function of $n_{Sing}$ for the Ising Hamiltonian"
    elif whichH == "RyR1":
        title = "Accuracy in function of $n_{Sing}$ for the Rydberg Hamiltonian"
    elif whichH == "RyNs":
        title = "Accuracy in function of $n_{Sing}$ for the Rydberg Hamiltonian"
    elif whichH == "IsNs":
        title = "Accuracy in function of $n_{Sing}$ for the Ising Hamiltonian"
    plt.plot(nSingVals, maxErrors, label="$n_{Part}$="+str(nParticles))
    # print("nSingVals")
    # print(nSingVals)
    # print("maxErrors")
    # print(maxErrors)
    plt.yscale("log")
    plt.xlabel("Number of singular values considered")
    plt.ylabel("biggest relative inaccuracy for $v_0$ of a test Hamiltonian")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig('SVDTest.svg', format='svg')
    # file1 = open(whichH+".txt","w")
    # file1.write(description)
    # file1.close()
    plt.show()

def computeAndCompareSVDApprox(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1nsStart, r1nsEnd, nR1ns, whichH, limit, nParticles, nSingStart, nSingEnd):
    A = computeSnapshotMatrix(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1nsStart, r1nsEnd, nR1ns, whichH, limit, nParticles)
    print(A.shape)
    u, s, v = LA.svd(A)
    print(s)
    maxErrors = np.zeros(nSingEnd - nSingStart + 1)
    nSingVals = np.zeros(nSingEnd - nSingStart + 1)
    index = 0
    HGrid, GSGrid = calculateHTestGrid(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1nsStart, r1nsEnd, nR1ns, whichH, limit, nParticles)
    for nSing in range(nSingStart, nSingEnd +1):
        B = extractBFromU(u,nSing)
        errorGrid = computeRelErrorGrid(HGrid, GSGrid, nDelta, nOmega, nR1ns, B, nParticles)
        maxErrors[index]=np.amax(errorGrid)
        nSingVals[index]=nSing
        index += 1
    plotSingularValueAccuracy(nSingVals, maxErrors, whichH, nParticles)
    HGrid = None
    GSGrid = None
    return u
    
def computeGroundStateSVD(H, B):
    h = calculateReducedHamiltonian(H, B)
    hGround = computeGroundState(h)
    HGroundAppr = np.dot(B,hGround)
    return HGroundAppr
    

def findNumberSingularValues(s, magnitude):
    for i in range(s.size):
        if s[i] < 10**(magnitude):
            return i
    return (s.size -1)
                             
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
    
def composeDescription(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1Start, r1End, nR1, whichH, limit, nParticles):
    description = (
        "H = "+whichH+""
        ", $\Omega_{min}^{SVD}$ = "+numberToString(omegaStart)+""
        ", $\Omega_{max}^{SVD}$ = "+numberToString(omegaEnd)+""
        ", $\delta_{min}^{SVD}$ = "+numberToString(deltaStart)+""
        ", $\delta_{max}^{SVD}$ = "+numberToString(deltaEnd)+""
        ", $R1_{min}^{SVD}$ = "+numberToString(r1Start)+""
        ", $R1_{max}^{SVD}$ = "+numberToString(r1End)+""
        ", $N_{Particles}$ = "+numberToString(nParticles))
    return description

def numberToString(num):
    magnitude = 0
    if abs(num) >= 1:
        while abs(num) >= 10:
            magnitude += 1
            num /= 10.0
    else:
        while abs(num) <= 1:
            magnitude -= 1
            num *= 10.0
    if magnitude != 0:
        return ('%.2f' % num)+"$\cdot 10^"+str(magnitude)+"$"
    else:
        return str(num)
        
def testHamiltonianInteraction():
    nParticles = 8
    n1=2
    n2=3
    result = sp.coo_matrix((2**nParticles, 2**nParticles))
    nop = sp.coo_matrix(np.matrix('1,0;0,0'))
    n1op = createSingleParticleOperator(nop, nParticles, n1)
    n2op = createSingleParticleOperator(nop, nParticles, n2)
    
def indexToBaseStateString(i, nParticles):
    result = format(i, f'{int(nParticles):03}b')
    result = result.replace('0', 'r')
    result = result.replace('1', 'g')
    return result
    
    
def sortEigVecByProbability(eigvec):
    start = time.time()
    result= [(0,0)]*eigvec.size
    nDigits=math.log2(eigvec.size)
    for i in range(eigvec.size):
        result[i] = (indexToBaseStateString(i, nDigits),eigvec[i]**2)
    result.sort(key=lambda tup: tup[1], reverse=True)
    end = time.time()
    print("sorting time")
    print(end-start)
    return result

def classifyStateStringN13(string):
    if string == 'rgrgrgrgrgrgr':
        return 2
    elif string == 'rggrggrggrggr':
        return 3
    elif string == 'rgggrgggrgggr':
        return 4
    else:
        return 1
    
def colorPlot(x,y,c):
    xStep = x[1] - x[0]
    xCorners = np.zeros((y.size + 1,x.size + 1))
    yStep = y[1] - y[0]
    yCorners = np.zeros((y.size + 1,x.size + 1))
    xCorners[0][0] = x[0] - xStep/2
    yCorners[0][0] = y[0] - yStep/2
    for i in range(x.size):
        xCorners[0][i+1] = x[i] + xStep/2
    for i in range(y.size):
        xCorners[i+1] = xCorners[0]
    for i in range(y.size):
        yCorners[i+1][0] = y[i] + yStep/2
    for i in range(y.size):
        yCorners[:,i+1] = yCorners[:,0]   
    plt.close()
    plt.pcolor(xCorners, yCorners, c)
    plt.colorbar()
    plt.xlabel("$\delta/\Omega$")
    plt.ylabel("$R_b/a (=n_s)$")
    plt.title("Phase diagram of crystalline Phases")
    plt.savefig('PhaseColorPlot.svg', format='svg')
    
    plt.show()
       
def computeCrystallinePhases():
    omegaStart = 2*math.pi*2*10**6
    omegaEnd = 2*math.pi*2*10**6
    nOmegaSVD=2
    deltaStart = 2*math.pi*0*10**6
    deltaEnd = 2*math.pi*5*10**6
    nDeltaSVD = 5
    nDeltaComp = 11
    r1nsStart= 0.5
    r1nsEnd= 3.5
    nR1nsSVD= 9
    nR1nsComp= 11
    nSingStart = 1
    nSingEnd = 40
    nSingUsed1 = 3
    nSingUsed2 = 5
    nSingUsed3 = 7
    limit=8
    nParticles = 13
    which = "RyNs"
    omega = 2*math.pi*2*10**6
    delta = 2*math.pi*2*10**6
    ns = 3.5
    
    
    u = computeAndCompareSVDApprox(deltaStart, deltaEnd, nDeltaSVD, omegaStart, omegaEnd, nOmegaSVD, r1nsStart, r1nsEnd, nR1nsSVD, which, limit, nParticles, nSingStart, nSingEnd)
    B1 = extractBFromU(u, nSingUsed1)
    B2 = extractBFromU(u, nSingUsed2)
    B3 = extractBFromU(u, nSingUsed3)
    print("SVD Done")
    stepsDelta = linearSteps(deltaStart, deltaEnd, nDeltaComp)
    stepsR1ns = linearSteps(r1nsStart, r1nsEnd, nR1nsComp)
    resultNormal = np.zeros((nDeltaComp, nR1nsComp))
    resultSVD1 = np.zeros((nDeltaComp, nR1nsComp))
    resultSVD2 = np.zeros((nDeltaComp, nR1nsComp))
    resultSVD3 = np.zeros((nDeltaComp, nR1nsComp))
    timeNormal = np.zeros(stepsR1ns.size*stepsDelta.size)
    timeSVD1 = np.zeros(stepsR1ns.size*stepsDelta.size)
    timeSVD2 = np.zeros(stepsR1ns.size*stepsDelta.size)
    timeSVD3 = np.zeros(stepsR1ns.size*stepsDelta.size)
    timeSort = np.zeros(4*stepsR1ns.size*stepsDelta.size)
    errors1 = 0
    errors2 = 0
    errors3 = 0
    for i in range(stepsR1ns.size):
        for j in range(stepsDelta.size):
            H = createFullHamiltonian(stepsDelta[j], omegaStart, stepsR1ns[i], which, limit, nParticles)
            try:
                start = time.time()
                groundState = computeGroundState(H)
                end = time.time()
                timeNormal[i*stepsDelta.size + j] = end-start
                start = time.time()
                sortedGroundState = sortEigVecByProbability(groundState)
                resultNormal[i][j] = classifyStateStringN13(sortedGroundState[0][0])
                end = time.time()
                timeSort[4*(i*stepsDelta.size + j)] = end - start
                print("normal done")
                
                start = time.time()
                groundState = computeGroundStateSVD(H.toarray(), B1)
                end = time.time()
                timeSVD1[i*stepsDelta.size + j] = end-start
                start = time.time()
                sortedGroundState = sortEigVecByProbability(groundState)
                resultSVD1[i][j] = classifyStateStringN13(sortedGroundState[0][0])
                end = time.time()
                timeSort[4*(i*stepsDelta.size + j) + 1] = end - start
                if resultSVD1[i][j] != resultNormal[i][j]:
                    errors1 += 1
                print("1 done")
                
                start = time.time()
                groundState = computeGroundStateSVD(H.toarray(), B2)
                end = time.time()
                timeSVD2[i*stepsDelta.size + j] = end-start
                start = time.time()
                sortedGroundState = sortEigVecByProbability(groundState)
                resultSVD2[i][j] = classifyStateStringN13(sortedGroundState[0][0])
                end = time.time()
                timeSort[4*(i*stepsDelta.size + j) + 2] = end - start
                if resultSVD2[i][j] != resultNormal[i][j]:
                    errors2 += 1
                print("2 done")
                
                start = time.time()
                groundState = computeGroundStateSVD(H.toarray(), B3)
                end = time.time()
                timeSVD3[i*stepsDelta.size + j] = end-start
                start = time.time()
                sortedGroundState = sortEigVecByProbability(groundState)
                resultSVD3[i][j] = classifyStateStringN13(sortedGroundState[0][0])
                end = time.time()
                timeSort[4*(i*stepsDelta.size + j) + 3] = end - start
                if resultSVD3[i][j] != resultNormal[i][j]:
                    errors3 += 1
                print("3 done")
                
            except:
                print("error")
                print(i)
                print(j)
            print(str(i)+", "+str(j)+" done")
    print("pixel mean SVD n="+str(nSingUsed1))
    print(timeSVD1.mean())
    print("errors per pixel")
    print(errors1/(stepsR1ns.size*stepsDelta.size))
    print("pixel mean SVD n="+str(nSingUsed2))
    print(timeSVD2.mean())
    print("errors per pixel")
    print(errors2/(stepsR1ns.size*stepsDelta.size))
    print("pixel mean SVD n="+str(nSingUsed3))
    print(timeSVD3.mean())
    print("errors per pixel")
    print(errors3/(stepsR1ns.size*stepsDelta.size))
    print("pixel mean normal")
    print(timeNormal.mean())
    print("sorting mean time")
    print(timeSort.mean())
    print("results normal")
    print(resultNormal)
    print("results 1")
    print(resultSVD1)
    print("results 2")
    print(resultSVD2)
    print("results 3")
    print(resultSVD3)
    colorPlot(stepsDelta/omega, stepsR1ns, resultNormal)
    
def testSVD():
    omegaStart = 2*math.pi*2*10**6
    omegaEnd = 2*math.pi*2*10**6
    nOmega=2
    deltaStart = 2*math.pi*0*10**6
    deltaEnd = 2*math.pi*5*10**6
    nDelta = 5
    limit=8 
    r1nsStart= 0.5 
    r1nsEnd= 3.5
    nR1ns= 9
    nSingStart = 10 
    nSingEnd = 40
    nSingUsed = 40
    which = "RyNs" 
    omega = 2*math.pi*2*10**6
    delta = 2*math.pi*2*10**6
    r1=2.87*10**-9
    ns=1
    nParticles = 13
    
    computeAndCompareSVDApprox(deltaStart, deltaEnd, nDelta, omegaStart, omegaEnd, nOmega, r1nsStart, r1nsEnd, nR1ns, which, limit, nParticles, nSingStart, nSingEnd, nSingUsed)

np.set_printoptions(linewidth=200)

computeCrystallinePhases()

