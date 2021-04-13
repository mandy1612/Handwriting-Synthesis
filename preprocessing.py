import cv2
import numpy as np
import pytesseract
# import tensorflow.keras
# from tensorflow.keras.models import load_model
import argparse

def getImgName(c):
    if  ord(c)>=ord('0') and ord(c)<=ord('9'):
        return ord(c)-48

    elif ord(c)>=ord('A') and ord(c)<=ord('Z'):
        return ord(c)-55

    elif ord(c)>=ord('a') and ord(c)<=ord('z'):
        return ord(c)-61


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def preprocessing(img,i):
# def preprocessing(img,i,model):

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    binary=gray.copy()
    n=1
    for x in range(n+1):
        start_x=(len(gray)//n)*x
        for y in range(n+1):
            start_y=(len(gray[0])//n)*y
            
            image=gray[start_x:start_x+(len(gray)//n), start_y:start_y+(len(gray[0])//n)]

            ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            binary[start_x:start_x+(len(gray)//n), start_y:start_y+(len(gray[0])//n)]=otsu

    binary2=binary.copy()
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))        
    binary=cv2.dilate(binary,horizontalStructure,iterations=1)

    cv2.imshow('Otsu',binary)
    cv2.waitKey()

    

    # boxes=cv2.Canny(gray,35,100)
    # cv2.imshow('Canny',boxes)
    # cv2.waitKey()
    

    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(15,1))
    # boxes=cv2.erode(boxes,horizontalStructure,iterations=1)
    
    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(30,2))
    # boxes=cv2.dilate(boxes,horizontalStructure,iterations=2)

    # cv2.imshow('E&D',boxes)
    # cv2.waitKey()


    # lines = cv2.HoughLinesP(boxes,1,np.pi/180,150,minLineLength=50,maxLineGap=100)

    # for line in lines: 
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(binary,(x1,y1),(x2,y2),(0),1)

    # cv2.imshow('HoughLines',binary)
    # cv2.waitKey()



    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for k, c in enumerate(contours):
        contours_poly[k] = cv2.approxPolyDP(c, 3, True)
        boundRect[k] = cv2.boundingRect(contours_poly[k])

    heights=[]
    widths=[]

    for m in range(len(contours)):
        heights.append(boundRect[m][3])
        widths.append(boundRect[m][2])

    heights.sort()
    widths.sort()

    print(heights)
    print(widths)

    heightQ1=int(len(heights)*0.07)
    minheight=heights[heightQ1+1]
    # heightQ3=int(len(heights)*0.98)
    # maxheight=heights[heightQ3+1]

    widthQ1=int(len(widths)*0.07)
    minwidth=widths[widthQ1+1]
    # widthQ3=int(len(widths)*0.98)
    # maxwidth=widths[widthQ3+1]

    maxheight=heights[-1]
    maxwidth=widths[-1]


    print(f'height range is [{minheight},{maxheight}]')
    print(f'width range is [{minwidth},{maxwidth}]')

    for k in range(len(contours)):

        height=boundRect[k][3]
        width=boundRect[k][2]
        
        start_x=int(boundRect[k][0])
        start_y=int(boundRect[k][1])


        # p1=(int(boundRect[k][0]), int(boundRect[k][1]))
        # p2=(int(boundRect[k][0]+boundRect[k][2]), int(boundRect[k][1]))
        # p3=(int(boundRect[k][0]+boundRect[k][2]), int(boundRect[k][1]+boundRect[k][3]))
        # p4=(int(boundRect[k][0]), int(boundRect[k][1]+boundRect[k][3]))

        if width<=maxwidth and height<=maxheight and width>=minwidth and height>=minheight:
            # print(boundRect[k][2],boundRect[k][3])
            character=binary2[start_y:start_y+height,start_x:start_x+width]
            character=cv2.resize(character,(28,28))
            # cv2.imshow('char',character)
            # cv2.waitKey()
            cv2.rectangle(img, (int(boundRect[k][0]), int(boundRect[k][1])), \
            (int(boundRect[k][0]+boundRect[k][2]), int(boundRect[k][1]+boundRect[k][3])), (0,255,0), 1)

            

            # temp_character = character.reshape(1,784)
            # temp_character = temp_character.astype('float32')
            # temp_character /= 255

            # out = model.predict(temp_character)
            # c=np.argmax(out)
            # print(c)

            cv2.imwrite("Images\\Font\\Manas\\{out}.jpeg".format(out=k),character)


    cv2.imshow('Counter',img)
    cv2.waitKey()





for i in range(39,40):

    # model = load_model('model.h5')
    img=cv2.imread("Images\\input\\{name}.jpeg".format(name=i))
    img = image_resize(img, width = 600)

    # preprocessing(img,i,model)
    preprocessing(img,i)