import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from math import sin, cos, tan, inf
from copy import *
# from math import inf
from scipy.spatial import distance

def getNearestDescriptor(estimatedPts, descriptors):
    kp = list(deepcopy(descriptors))
    finalkp = []
    
    for red in estimatedPts:
        minimalcd = ()
        mimimalidx = 0
        minimal = inf
        for n, white in enumerate(kp):
            d = distance.euclidean(red, white)
            if d < minimal: 
                minimal = d
                mimimalidx = n
            # np.delete(kp, white)
                minimalcd = white
        kp.pop(mimimalidx)
        finalkp.append(minimalcd)
    return finalkp

class Camera:
    #pose params
    x,y,z=0,0,0
    yaw,pitch,roll=0,0,0
    #inner params
    hFOV=123 #
    hRes, vRes=1280, 960 #
    # hRes, vRes=960, 1280
    nearZ, farZ=0.1, 50  #
    kPI=np.pi/180
    swapCS = [[0, 1, 0, 0], [0, 0, 1, 0],
              [1, 0, 0, 0], [0, 0, 0, 1]]

    def calcProjMatrix(self):
        ar=self.hRes/self.vRes
        tn=tan(self.hFOV*self.kPI/2)
        d1=self.nearZ-self.farZ
        d2=self.nearZ+self.farZ
        A, B, C, D = 1/ar/tn, 1/tn, d2/d1, 2*self.farZ*self.nearZ/d1
        mproj=np.array([[A, 0, 0, 0], [0, B, 0, 0], [0, 0, C, D], [0, 0, -1, 0]])
        return mproj

    def calcViewMatrix(self):
        y,p,r=self.yaw*self.kPI, self.pitch*self.kPI, self.roll*self.kPI
        # swap between world CS and screen CS
        Z, X, Y = -self.x, -self.y, -self.z
        # calc rotation matricies
        cr, cp, cy = cos(r), cos(p), cos(y)
        sr, sp, sy = sin(r), sin(p), sin(y)
        myaw=[[cy, -sy, 0, 0], 
              [sy, cy, 0, 0], 
              [0, 0, 1, 0], 
              [0, 0, 0, 1]] #z
        
        mpit=[[cp, 0, sp, 0], 
              [0, 1, 0, 0], 
              [-sp, 0, cp, 0], 
              [0, 0, 0, 1]] #y
        
        mrol=[[1, 0, 0, 0], 
              [0, cr, -sr, 0], 
              [0, sr, cr, 0], 
              [0, 0, 0, 1]] #x
        # calc full view transformation
        mshift = [[1, 0, 0, X], 
                  [0, 1, 0, Y], 
                  [0, 0, 1, Z], 
                  [0, 0, 0, 1]]
        cam_matrix = np.array(mshift) @ self.swapCS @ myaw @ mpit @ mrol
        return cam_matrix

    def transfNDCToScreen(self, pNDC):
        pScreen = pNDC / pNDC[-1]
        pScreen += [1, 1, 0, 0]  # shifting & scaling NDC coords
        pScreen *= [0.5 * self.hRes, 0.5 * self.vRes, 1, 1]
        # print(f'pScreen: {pScreen[:2]}')
        return pScreen[:2]

    def transfPtsFromWorldToScreen(self, pts, M=None):
        res=[]
        if M is None: M=self.calcProjMatrix()@self.calcViewMatrix()
        for p in pts:
            p_ = [p[0], -p[1], p[2], 1]
            pNDC = M @ p_
            pScreen = self.transfNDCToScreen(pNDC)
            res.append(tuple(pScreen))
        # print(f'!!!!transfPtsFromWorldToScreen: {res}')
        return res
    
    def _transfPtsFromWorldToScreen(self, pts, M=None):
        res=[]
        if M is None: M=self.calcProjMatrix()@self.calcViewMatrix()
        for p in pts:
            p_ = [p[0], -p[1], p[2], 1]
            pNDC = M @ p_
            pScreen = self.transfNDCToScreen(pNDC)
            res.append(tuple(pScreen))
        # print(f'!!!!transfPtsFromWorldToScreen: {res}')
        return res
    
    def calcJacobianMatsFromWorldToScreen(self, pts, delta=0.001, M=None):
        res = []
        if M is None: M = self.calcProjMatrix() @ self.calcViewMatrix()
        dXYZ=[delta, -delta, delta]
        for p in pts:
            G = []
            # numerical gradient calculation for each point
            for iaxis in range(3):
                # original pt
                p1_ = [p[0], p[1], p[2], 1]
                pNDC1 = M @ p1_ # УМНОЖЕНИЕ ВЕКТОРА ПОЗИЦИИ НА МАТРИЦУ ПРОЕКЦИИ?
		 # ПРОСТРАНСТВО КЛИПЕРА?
		 # ЧТОБЫ ПЕРЕВЕСТИ В НОРМИРОВАННОЕ ПРОСТРАНСТВО
                pScreen1 = self.transfNDCToScreen(pNDC1)
                # shifted pt
                p2_ = [p[0], p[1], p[2], 1]
                p2_[iaxis] += dXYZ[iaxis]
                pNDC2 = M @ p2_
                pScreen2 = self.transfNDCToScreen(pNDC2)
                # difference
                dFdx = tuple((pScreen2 - pScreen1)/delta)
                G.append(dFdx)
            res.append(list(G))
        return res

    def doGradDescent(self, screenPts, estWorldPts, eta=0.000001, M=None):
        if M is None: M = self.calcProjMatrix() @ self.calcViewMatrix()
        estScreenPts = self.transfPtsFromWorldToScreen(estWorldPts, M=M)
        estScreenPts = np.array(estScreenPts)

        jmats = self.calcJacobianMatsFromWorldToScreen(estWorldPts, M=M)
        screenPts = np.array(screenPts)
        new_sc_pts = getNearestDescriptor(estScreenPts, screenPts)
        print('len', len(new_sc_pts), 'NEW SC PTS: ', new_sc_pts)
        errVecs = estScreenPts - new_sc_pts
        res = []
        for i in range(len(estWorldPts)):
            J = np.array(jmats[i])
            E = errVecs[i]
            changes = np.zeros(3)
            for j in range(len(E)):
                grad = J[:, j] # срез j-го столбца
                e = E[j]
                changes += -eta * grad * e # как назвать? что за эта?
            res.append(estWorldPts[i] + changes)
        return res


