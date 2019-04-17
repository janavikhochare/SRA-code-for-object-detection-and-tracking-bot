
import cv2
import numpy as np
from PIL import Image
MIN_MATCH_COUNT=20
i=0
DIST=[]
j=0
SP=[]

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

TrainingImg=cv2.imread("TrainImg.jpg",0)
trainKP,trainDesc=detector.detectAndCompute(TrainingImg,None)

cam=cv2.VideoCapture(0)
while (cam.isOpened()): 
    ret, QueryImgBGR=cam.read()
    if ret:
    	QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    	queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    	matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=TrainingImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
	liss=[np.int32(queryBorder)]
	a=liss[0][0][0][1]
	y=liss[0][0][2][1]
	r=abs((a-y)/2)
	b=((liss[0][0][0][0]+liss[0][0][1][0]+liss[0][0][3][0]+liss[0][0][2][0])/4)
        z=((liss[0][0][0][1]+liss[0][0][1][1]+liss[0][0][3][1]+liss[0][0][2][1])/4)
	width=np.size(QueryImg,1)
	height=np.size(QueryImg,0)
	cv2.line(QueryImgBGR , (320,0),(320,480),(255,0,0),5)
	#print b,z
	cv2.line(QueryImgBGR, (b,z),(b,z),(0,0,255),20)
	#print width/2 ,height/2	
	def distance_to_camera(knownWidth, r):

	
		return ((knownWidth * 551.37) / r)

		#to be included
		# initialize the known distance from the camera to the object, which
		# in this case is 24 inches
	KNOWN_DISTANCE = 10.98

		#to be included after changing the width 
		# initialize the known object width, which in this case, the piece of
		# paper is 12 inches wide
	KNOWN_WIDTH = 3.465


	focalLength = 551.37


	inches = distance_to_camera(KNOWN_WIDTH, r)
	#print inches
	
	
	
	
	DIST.append(inches)
		
	if i!=0:
	
		dist_covered=abs(DIST[i]-DIST[i-1])
		#print dist_covered
		speed=dist_covered*7
		#print  inches ,speed
		SP.append(speed)

		if j>9:
			spe=(SP[j]+SP[j-1]+SP[j-2]+SP[j-3]+SP[j-4]+SP[j-5]+SP[j-6]+SP[j-7]+SP[j-8]+SP[j-9])/10
			print spe
		j=j+1
	i=i+1
	if b<320:
		print "right"
	elif b>320:
		print "left"
	else:
		print "centre"
				
    else:
        print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()	
