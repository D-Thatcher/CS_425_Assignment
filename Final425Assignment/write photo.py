import os
import urllib.request

pth = r"C:\Users\User\Desktop\images2\New Text Document"
svto = r"C:\Users\User\Desktop\images_proper"

def file_len(sv,fname):
    i=-2
    with open(fname) as f:
        for i, l in enumerate(f):
            if(l.strip()[-3:].lower()=="gif"):
                urllib.request.urlretrieve(l.strip(),
                                           os.path.join(sv, str(i) + "img.gif"))
            else:
                print(l.strip())
                urllib.request.urlretrieve(l.strip(),
                                           os.path.join(sv,str(i)+ "img.jpg"))


    return i + 1

print(file_len(svto,pth))
