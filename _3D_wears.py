import cv2, os
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageOps
import matplotlib.pyplot as plt
import sys
def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background
def detectFaceOpenCVHaar(faceCascade, frame,mask1, inHeight=300, inWidth=0):



    frameOpenCVHaar = frame.copy()
    frameHeight = frameOpenCVHaar.shape[0]
    frameWidth = frameOpenCVHaar.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
    frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(frameGray)
    bboxes = []
    for (x, y, w, h) in faces:
        
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                  int(x2 * scaleWidth), int(y2 * scaleHeight)]
        bboxes.append(cvRect)
        cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                      int(round(frameHeight / 150)), 4)








        #разобраться с размерами
       # mask1 = cv2.resize(mask1, (1000, 1000))


        frameOpenCVHaar = overlay_transparent(frameOpenCVHaar, mask1, x, y)


    return frameOpenCVHaar, bboxes,faces

if __name__ == "__main__" :
    image = './photo/czc.jpg'
    image2 = './photo/CCM-LOGO-SNAPBACK-MA-_1_.png'

    img_cv = cv2.imread(image)
    mask = cv2.imread(image2, cv2.IMREAD_UNCHANGED)

    #img_cv = cv2.resize(img_cv, (800, 1000))


    

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")       
    
    outOpencvHaar, bboxes,faces = detectFaceOpenCVHaar(faceCascade, img_cv,mask)   
    print('Faces found: ', len(faces))







    cv2.imshow('res',outOpencvHaar)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    



   # im2 = im2.resize((w, h))
  

  #  im1.paste(im2.convert('RGB'), (x,y), im2)


#im1.show()



   
