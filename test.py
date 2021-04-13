import cv2
import numpy as np


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

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    binary=gray.copy()
    n=20
    for x in range(n+1):
        start_x=(len(gray)//n)*x
        for y in range(n+1):
            start_y=(len(gray[0])//n)*y
            
            image=gray[start_x:start_x+(len(gray)//n), start_y:start_y+(len(gray[0])//n)]

            ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            binary[start_x:start_x+(len(gray)//n), start_y:start_y+(len(gray[0])//n)]=otsu

    # horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))        
    # binary=cv2.dilate(binary,horizontalStructure,iterations=1)

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
        

        # p1=(int(boundRect[k][0]), int(boundRect[k][1]))
        # p2=(int(boundRect[k][0]+boundRect[k][2]), int(boundRect[k][1]))
        # p3=(int(boundRect[k][0]+boundRect[k][2]), int(boundRect[k][1]+boundRect[k][3]))
        # p4=(int(boundRect[k][0]), int(boundRect[k][1]+boundRect[k][3]))

        # if width<=maxwidth and height<=maxheight and width>=minwidth and height>=minheight:
            # print(boundRect[k][2],boundRect[k][3])

        cv2.rectangle(img, (int(boundRect[k][0]), int(boundRect[k][1])), \
        (int(boundRect[k][0]+boundRect[k][2]), int(boundRect[k][1]+boundRect[k][3])), (0,2550), 1)


    cv2.imshow('Counter',img)
    cv2.waitKey()

    # cv2.imwrite("Images\\Output\\contour{name}.jpeg".format(name=i),img)






for i in range(27,28):

    img=cv2.imread("Images\\Input\\{name}.jpeg".format(name=i))
    # img = image_resize(img, width = 600)

