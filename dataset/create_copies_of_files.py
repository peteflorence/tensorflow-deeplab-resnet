import os
import sys

file_list = sys.argv[1]

copy_dir = "copy"
os.system("mkdir " + copy_dir)

counter = 0
with open(file_list) as f:
    content = f.readlines()
    content = [x.strip() for x in content] 
    for i in content:
    	print i
    	i = i.split()
    	os.system("cp " + i[0] + " " + copy_dir+"/"+str(counter).zfill(8)+"_rgb.png")
    	os.system("cp " + i[0] + " " + copy_dir+"/"+str(counter).zfill(8)+"_labels.png")
    	counter+=1