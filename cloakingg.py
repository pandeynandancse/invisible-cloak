import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('nandan_invisible.avi' , fourcc, 20.0, (640,480))
time.sleep(2)
background = 0#capturing background
for i in range(30):
    ret, background = cap.read()#capturing image
while(cap.isOpened()):
    ret, img = cap.read()
    
    if not ret:
        break
    #----------------------------------------------------------------------------------------------
    # color detection starting    
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #in hsv there are red on two corner

    # range of red color on hsv scale at first corner
    lower_red = np.array([0,80,40])
    upper_red = np.array([30,255,255])

    # red cloth is white , everything else is black
    mask1 = cv2.inRange(hsv , lower_red , upper_red)

    # range of red color on hsv scale at end/another corner
    lower_red = np.array([150,80,40])
    upper_red = np.array([180,255,255])

    
    mask2 = cv2.inRange(hsv , lower_red , upper_red)
    
    
    mask1 = mask1 + mask2 #OR operation pixel wise so that this mask can detect all red colors that is cloth here
    #color detection performed/completed with mask1 and mask2 and via their operation
    #--------------------------------------------------------------------------------------------------
    # now correct the mask via morphology so that some error in mask can be corrected see example of 'j'
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN ,np.ones((3,3) , np.uint8) , iterations=2)       
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE ,np.ones((3,3) , np.uint8) , iterations=1)
     
    # mask2 will be opposite of mask 1
    # red cloth is black , everything else is white
    mask2 = cv2.bitwise_not(mask1)
    
    # red cloth will be replaced with backgroud image and eveything else in 'img' will be black 
    res1 = cv2.bitwise_and(background, background, mask=mask1)

    # image that is being capturing (img) will be as it is except that red color cloth will be black
    res2 = cv2.bitwise_and(img, img, mask=mask2)

    #both result/output is equally weighted with 1 and 1 . Here 0 is gamma correction , here we are not correcting anything. 
    final_output = cv2.addWeighted(res1 , 1, res2 , 1, 0)
    
    cv2.imshow('Nandan' , final_output)
    k=cv2.waitKey(25)
    if k==25:
        break
        
cap.release()
cv2.destroyAllWindows()

