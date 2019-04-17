import cv2
import numpy as np
from PIL import Image
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)

#	Motor1A=16
#	Motor1B=18
#	Motor1E=22

#	Motor2A=23
#	Motor2B=21
#	Motor2E=19

#	GPIO.setup(Motor1A,GPIO.OUT)
#	GPIO.setup(Motor1B,GPIO.OUT)
#	GPIO.setup(Motor1E,GPIO.OUT)

#	GPIO.setup(Motor2A,GPIO.OUT)
#	GPIO.setup(Motor2B,GPIO.OUT)
#	GPIO.setup(Motor2E,GPIO.OUT)

MIN_MATCH_COUNT=30
i=0
DIST=[]
cam=cv2.VideoCapture(0)
orb=cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)

TrainingImg=cv2.imread("TrainImg.jpg",0)
kpts1, descs1=orb.detectAndCompute(TrainingImg,None)

while (cam.isOpened()): 
    ret, QueryImgBGR=cam.read()
    if ret:
    	QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
	kpts2, descs2=orb.detectAndCompute(QueryImg,None)
    	bf=cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=False)
	matches = bf.knnMatch(descs2 , descs1 , k=2)
	


    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(kpts1[m.trainIdx].pt)
            qp.append(kpts2[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=TrainingImg.shape[:2]
        trainBorder=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
	liss=[np.int32(queryBorder)]
	a=liss[0][0][0][1]
	y=liss[0][0][2][1]
	r=abs((a+y)/2)
	b=liss[0][0][0][0]
	z=liss[0][0][2][0]
	s=abs((b+z)/2)
	width=np.size(QueryImg,1)
	height=np.size(QueryImg,0)
	cv2.line(QueryImgBGR , (320,0),(320,480),(255,0,0),2)
	cv2.line(QueryImgBGR , (s,r),(s,r),(255,0,0),2)
	
		
	def distance_to_camera(knownWidth, r):

	
		return ((knownWidth * 551.37) / r)

	
	KNOWN_DISTANCE = 10.98

		
	KNOWN_WIDTH = 3.465


	focalLength = 551.37


	inches = distance_to_camera(KNOWN_WIDTH, r)
	
	
	
	
	
#	DIST.append(inches)
		
#	if i!=0:
	
#		dist_covered=abs(DIST[i]-DIST[i-1])
		
#		speed=dist_covered*7
#		if inches ==12:
#			if s>320:
#				GPIO.output(Motor1A,GPIO.HIGH)
#				GPIO.output(Motor1B,GPIO.LOW)
#				GPIO.output(Motor1E,GPIO.HIGH)
#				GPIO.output(Motor2A,GPIO.HIGH)
#				GPIO.output(Motor2B,GPIO.LOW)
#				GPIO.output(Motor2E,GPIO.HIGH)
#			elif s==320:
#				GPIO.output(Motor1A,GPIO.HIGH)
#				GPIO.output(Motor1B,GPIO.LOW)
#				GPIO.output(Motor1E,GPIO.HIGH)
#				GPIO.output(Motor2A,GPIO.HIGH)
#				GPIO.output(Motor2B,GPIO.LOW)
#				GPIO.output(Motor2E,GPIO.HIGH)
#			else:
#				GPIO.output(Motor1A,GPIO.HIGH)
#				GPIO.output(Motor1B,GPIO.LOW)
#				GPIO.output(Motor1E,GPIO.HIGH)
#				GPIO.output(Motor2A,GPIO.HIGH)
#				GPIO.output(Motor2B,GPIO.LOW)
#				GPIO.output(Motor2E,GPIO.HIGH)
#		elif inches<12:
#			GPIO.output(Motor1E,GPIO.LOW)
#			GPIO.output(Motor2E,GPIO.LOW)
#		else:
#			if s>320:
#				GPIO.output(Motor1A,GPIO.HIGH)
#				GPIO.output(Motor1B,GPIO.LOW)
#				GPIO.output(Motor1E,GPIO.HIGH)
#				GPIO.output(Motor2A,GPIO.HIGH)
#				GPIO.output(Motor2B,GPIO.LOW)
#				GPIO.output(Motor2E,GPIO.HIGH)
#			elif s==320:
#				GPIO.output(Motor1A,GPIO.HIGH)
#				GPIO.output(Motor1B,GPIO.LOW)
#				GPIO.output(Motor1E,GPIO.HIGH)
#				GPIO.output(Motor2A,GPIO.HIGH)
#				GPIO.output(Motor2B,GPIO.LOW)
#				GPIO.output(Motor2E,GPIO.HIGH)
#			else:
#				GPIO.output(Motor1A,GPIO.HIGH)
#				GPIO.output(Motor1B,GPIO.LOW)
#				GPIO.output(Motor1E,GPIO.HIGH)
#				GPIO.output(Motor2A,GPIO.HIGH)
#				GPIO.output(Motor2B,GPIO.LOW)
#				GPIO.output(Motor2E,GPIO.HIGH)
				

					
#	i=i+1
	
				
    else:
        print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()	
