from PIL import Image
import os, sys


def resize(path, dirs):
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((100,100), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)


def main():
	for i in range(1, 12):
		if (i != 7):
			path = "/Users/clinic1718/Desktop/normFrames80/train/G" + str(i) + "/"
			dirs = os.listdir( path )
			resize(path, dirs)
	for i in range(1, 12):
		if (i == 7):
			continue
		path = "/Users/clinic1718/Desktop/normFrames80/validation/G" + str(i) + "/"
		dirs = os.listdir( path )
		resize(path, dirs)		

main()
#resize()