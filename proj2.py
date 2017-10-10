import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import cv2
import pycuber as pc
from pycuber.solver import CFOPSolver

colorCode=[['red',(0,0,255)],['blue',(255,0,0)],['orange',(0,165,255)],['yellow',(0,255,255)],['white',(255,266,255)],['green',(0,255,0)]]
colIndex = 0
colors=[]
faces=[]
faceImage=[]
faceContours=[]
faceCentroids=[]
refPt = []

def calculateContours(inImg):
    img,contours, hierarchy = cv2.findContours(inImg,1, cv2.CHAIN_APPROX_SIMPLE)
    height,width=inImg.shape
    imgArea = height*width
    colImg=np.zeros((height,width,3),np.uint8)
    selectedContours=[]
    areas=[]
    for contour in contours:
        contArea = cv2.contourArea(contour)
        if  contArea<imgArea/25 and contArea > imgArea/70:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(contArea)/hull_area
            if solidity > 0.85:
                cv2.drawContours(colImg, [contour], 0, (255,0,0), 1)
                selectedContours.append(contour)
                areas.append(contArea)


    return selectedContours,colImg;

def orderContours(cnts, centroids):
    for x in (0,3,6):
        (cnts[x:x+3], centroids[x:x+3]) = zip(*sorted(zip(cnts[x:x+3],centroids[x:x+3]), key=lambda b:b[1][0], reverse=True))

    return cnts, centroids;

def areRubiksFaces(centroids):

    for x in [0,3,6]:
        a = centroids[x]
        b = centroids[1+x]
        c = centroids[2+x]

        if(abs((a[0]-c[0])*(b[1]-a[1])-(a[0]-b[0])*(c[1]-a[1])) > 3000):
            return False

    for x in [0,1,2]:
        a = centroids[x]
        b = centroids[3+x]
        c = centroids[6+x]

        if(abs((a[0]-c[0])*(b[1]-a[1])-(a[0]-b[0])*(c[1]-a[1])) > 3000):
            return False

    return True;

def calculateCentroids(contours):
    centroids=[]
    for c in cont:
        M = cv2.moments(c)
        if not  M['m00'] == 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx = cy = 0
        centroids.append((cx,cy))
    return centroids;

def threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(5,5))
    img = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,27,2)

    return img;

def checkContours(cont):
    if(len(cont) == 9):
        centroids = calculateCentroids(cont)
        cont, centroids = orderContours(cont,centroids)
        if(areRubiksFaces(centroids)):
            return True, centroids

    return False,[];

def drawCentroids(centroids,colImg):
    for x in (0,1,2):
        cv2.circle(colImg,centroids[3*x],x+1,(0,0,255))
        cv2.circle(colImg,centroids[1+3*x],x+1,(0,255,0))
        cv2.circle(colImg,centroids[2+3*x],x+1,(255,0,0))
    return colImg;


def getFaces(contours,frame):
    faceColors = []
    for x in  range(0,9):
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contours[x]], 0, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(frame, mask=mask)[:3]
        faceColors.append(mean)

    return faceColors;

def clickCallback(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]


def labelFaces(faceContours, faceCentroids, faceImage, faces, colors):
    global refPt
    fI = []
    for im in faceImage:
        fI.append(im.copy())

    colorsDict = OrderedDict({
            "red": colors[0],
            "blue": colors[1],
            "orange": colors[2],
            "yellow": colors[3],
            "white": colors[4],
            "green": colors[5]})

    lab = np.zeros((len(colorsDict), 1, 3), dtype="uint8")
    colorNames = []
    for (j, (name, bgr)) in enumerate(colorsDict.items()):
        lab[j] = bgr
        colorNames.append(name)

    result = []
    for i in range(0,6):
        result.append([])
        result[i] = []
        for x in  range(0,9):
            mask = np.zeros(faceImage[i].shape[:2], dtype="uint8")

            cv2.drawContours(mask, [faceContours[i][x]], -1, 255, -1)
            mask = cv2.erode(mask, None, iterations=1)
            mean = cv2.mean(faceImage[i], mask=mask)[:3]

            minDist = (np.inf, None)

            for (k, row) in enumerate(lab):
                d = dist.euclidean(row[0], mean)
                if d < minDist[0]:
                    minDist = (d, k)

            text = colorNames[minDist[1]]
            result[i].append(text)
            #text = text + str(x)
            cv2.putText(faceImage[i], text, (faceCentroids[i][x]),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255,255,0), 1)
        cv2.namedWindow(str(i),cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(str(i), clickCallback)
        while True:
            cv2.imshow(str(i),faceImage[i])

            if not refPt == []:
                x = 0
                for el in range(0,9):
                    if cv2.pointPolygonTest(faceContours[i][el],refPt[0],False) == 1:
                        newImage = fI[i].copy()
                        cv2.putText(newImage, "New Color!", (faceCentroids[i][el]),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255,255,0), 1)
                        cv2.imshow(str(i),newImage)
                        x = el
                        break
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('r'):
                        result[i][x] = 'red'
                        refPt = []

                    if key == ord('b'):
                        result[i][x] = 'blue'
                        refPt = []

                    if key == ord('o'):
                        result[i][x] = 'orange'
                        refPt = []

                    if key == ord('w'):
                        result[i][x] = 'white'
                        refPt = []

                    if key == ord('y'):
                        result[i][x] = 'yellow'
                        refPt = []

                    if key == ord('g'):
                        result[i][x] = 'green'
                        refPt = []


                    faceImage[i] = fI[i].copy()
                    index = 0
                    for col in result[i]:
                        cv2.putText(faceImage[i], col, (faceCentroids[i][index]),cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255,255,0), 1)
                        index = index + 1
                    break

            if cv2.waitKey(1) & 0xFF == ord(' '):
                break


    return result


cap = cv2.VideoCapture(0)
detecting = True
while(detecting):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    img = threshold(frame)
    cv2.namedWindow('threshold',cv2.WINDOW_NORMAL)
    cv2.imshow('threshold',img)


    cont,colImg= calculateContours(img)
    faceDetected, centroids = checkContours(cont)

    ##when a face has been detected##
    if(faceDetected):
        faceColors = getFaces(cont,frame)

        colors.append(faceColors[4])
        faces.append(faceColors)
        faceImage.append(frame)
        faceContours.append(cont)
        faceCentroids.append(centroids)
        colIndex = colIndex + 1
        if(colIndex == 6):
            colIndex = 0
            detecting = False
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.imshow('img', frame)

        cap.release()
        cv2.waitKey(0)
        cap = cv2.VideoCapture(0)

    cv2.namedWindow('contours',cv2.WINDOW_NORMAL)
    cv2.namedWindow('threshold',cv2.WINDOW_NORMAL)
    cv2.putText(colImg, colorCode[colIndex][0], (100,20),
        cv2.FONT_HERSHEY_TRIPLEX, 1.2, colorCode[colIndex][1], 2)
    cv2.imshow('contours',colImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('r'):
        colIndex = 0
        faces = []
        colors = []
        faceContours = []
        faceImage = []


cap.release()
colors = labelFaces(faceContours, faceCentroids, faceImage, faces, colors)

print(colors)

########  MAPPING  ########
f = pc.Centre(F="red")
l = pc.Centre(L="blue")
r = pc.Centre(R="yellow")
u = pc.Centre(U="white")
d = pc.Centre(D="green")
b = pc.Centre(B="orange")

flu = pc.Corner(F=colors[0][6],L=colors[1][8],U=colors[4][6])
fru = pc.Corner(F=colors[0][8],R=colors[3][6],U=colors[4][0])
fu = pc.Edge(F=colors[0][7],U=colors[4][3])
fl = pc.Edge(F=colors[0][3],L=colors[1][5])
fr = pc.Edge(F=colors[0][5],R=colors[3][3])
fld = pc.Corner(F=colors[0][0],L=colors[1][2],D=colors[5][0])
fd = pc.Edge(F=colors[0][1],D=colors[5][3])
fdr = pc.Corner(F=colors[0][2],R=colors[3][0],D=colors[5][6])

lu = pc.Edge(L=colors[1][7],U=colors[4][7])
ld = pc.Edge(L = colors[1][1],D=colors[5][1])
lb = pc.Edge(L= colors[1][3],B=colors[2][5])
lub = pc.Corner(L=colors[1][6],U=colors[4][8],B=colors[2][8])
ldb = pc.Corner(L=colors[1][0],B = colors[2][2],D=colors[5][2])

ru = pc.Edge(R=colors[3][7],U=colors[4][1])
rd = pc.Edge(R=colors[3][1],D=colors[5][7])
rb = pc.Edge(R=colors[3][5],B=colors[2][3])

rub = pc.Corner(R=colors[3][8],U=colors[4][2],B=colors[2][6])
rdb = pc.Corner(R=colors[3][2],D=colors[5][8],B=colors[2][0])

ub = pc.Edge(U=colors[4][5], B=colors[2][7])
db = pc.Edge(D=colors[5][5], B=colors[2][1])

#########  SOLVE  ##########
c = pc.Cube([f,l,r,u,d,b,flu,fru,fu,fl,fr,fld,fd,fdr,lu,ld,lb,lub,ldb,ru,rd,rb,rub,rdb,ub,db])
sol = CFOPSolver(c)

final_result = sol.solve(c)

ind = 0
for instr in final_result:
    if instr == 'F':
        face = 'red'
        orient = 'clockwise'
    if instr == 'F\'':
        face = 'red'
        orient = 'counter-clockwise'
    if instr == 'R':
        face = 'yellow'
        orient = 'clockwise'
    if instr == 'R\'':
        face = 'yellow'
        orient = 'counter-clockwise'
    if instr == 'L':
        face = 'blue'
        orient = 'clockwise'
    if instr == 'L\'':
        face = 'blue'
        orient = 'counter-clockwise'
    if instr == 'U':
        face = 'white'
        orient = 'clockwise'
    if instr == 'U\'':
        face = 'white'
        orient = 'counter-clockwise'
    if instr == 'B':
        face = 'orange'
        orient = 'clockwise'
    if instr == 'B\'':
        face = 'orange'
        orient = 'counter-clockwise'
    if instr == 'D':
        face = 'green'
        orient = 'clockwise'
    if instr == 'D\'':
        face = 'green'
        orient = 'counter-clockwise'

    if instr == 'F2':
        face = 'red'
        orient = 'twice'
    if instr == 'R2':
        face = 'yellow'
        orient = 'twice'
    if instr == 'L2':
        face = 'blue'
        orient = 'twice'
    if instr == 'U2':
        face = 'white'
        orient = 'twice'
    if instr == 'B2':
        face = 'orange'
        orient = 'twice'
    if instr == 'D2':
        face = 'green'
        orient = 'twice'
    ind = ind + 1
    print(str(ind) + '. Turn the '+face+' face '+orient)
    cv2.waitKey(0)

print("Congratulations, you just solved the Rubik cube (or maybe not)!!!!")


cv2.destroyAllWindows()
