from PIL import Image
import os, os.path

path=r"Images/result/"

count=0
for file in os.listdir(path):
    if file.endswith(".jpeg"):
        count+=1
print(count)
def imgToPdf(path):
    img0=Image.open(f"Images/result/res{0}.jpeg")
    l=[]
    for i in range(0,count):
        img=Image.open(f"Images/result/res{i}.jpeg")
        im=img.convert('RGB')
        l.append(img)
    l.pop(0)
    img0.save(r'Images/result/myImages.pdf',save_all=True, append_images=l)

imgToPdf(path)