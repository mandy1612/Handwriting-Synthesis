import sys
import cv2
from PIL import Image
import os, os.path
import glob

files = glob.glob(os.path.join('Images/result/temp/*.jpeg')) 
for file in files:
    os.remove(file)

def getImgName(c):
    if  ord(c)>=ord('0') and ord(c)<=ord('9'):
        return ord(c)-48

    elif ord(c)>=ord('A') and ord(c)<=ord('Z'):
        return ord(c)-55

    elif ord(c)>=ord('a') and ord(c)<=ord('z'):
        return ord(c)-61

f=open('input.txt','r')

originalpage = cv2.imread('Images/pageA4.jpeg')

pagelen=len(originalpage)
pagewidth=len(originalpage[0])

fontHeight=28
fontWidth=20

counter=0

x=0     
y=0

page=originalpage.copy()

while True:
    c=f.read(1)
    if not c:
        break

    else:

        if c=='\n':
            x=x+fontHeight
            y=0
        else:
            if c!=' ':
                print(c)
                font=cv2.imread(r'Images\Font\{person}\{name}.jpeg'.format(person=sys.argv[1],name=str(getImgName(c))))
                font=font[:,4:24]
                # cv2.imshow("font",font) 
                # cv2.waitKey()    
                
                try:
                    page[x:x+fontHeight,y:y+fontWidth]=~font          
                except:
                    
                    if y+fontWidth > pagewidth and x<=pagelen-(fontHeight*2):
                        x=x+fontHeight
                        y=0
                        page[x:x+fontHeight,y:y+fontWidth]=~font   

                    elif y+fontWidth > pagewidth and x>pagelen-(fontHeight*2):
                        cv2.imshow("page",page)
                        cv2.waitKey()
                        cv2.imwrite(f"Images/result/temp/res{counter}.jpeg",page)
                        counter=counter+1
                        page=originalpage.copy()
                        x=0
                        y=0
                        page[x:x+fontHeight,y:y+fontWidth]=~font                 

            y=y+fontWidth
# 3508 x 2480
cv2.imwrite(f"Images/result/temp/res{counter}.jpeg",page)
counter=counter+1


#pdf


def imgToPdf(path):
    
    count=0
    for file in os.listdir(path):
        if file.endswith(".jpeg"):
            count+=1
    print(count)
    img0=Image.open(f"Images/result/temp/res{0}.jpeg")
    l=[]
    for i in range(1,count):
        img=Image.open(f"Images/result/temp/res{i}.jpeg")
        
        im=img.convert('RGB')
        l.append(img)
    # l.pop(0)
    print(l)
    img0.save(fr"Images/result/output.pdf",save_all=True, append_images=l)

path=r"Images/result/temp"
imgToPdf(path)