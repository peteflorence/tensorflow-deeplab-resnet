import os

train_list = []
train_list.append("/home/corl2017/spartan/src/CorlDev/experiments/drill_50_train_scenes.txt.imglist.txt")
train_list.append("/home/corl2017/spartan/src/CorlDev/experiments/drill_25_train_scenes.txt.imglist.txt")
train_list.append("/home/corl2017/spartan/src/CorlDev/experiments/drill_10_train_scenes.txt.imglist.txt")
train_list.append("/home/corl2017/spartan/src/CorlDev/experiments/drill_5_train_scenes.txt.imglist.txt")
train_list.append("/home/corl2017/spartan/src/CorlDev/experiments/drill_2_train_scenes.txt.imglist.txt")
train_list.append("/home/corl2017/spartan/src/CorlDev/experiments/drill_1_train_scenes.txt.imglist.txt")
train_list.append("/home/corl2017/spartan/src/CorlDev/experiments/drill_11_test_scenes.txt.imglist.txt")

for i in train_list:
	os.system("python train.py --not-restore-last --data-list " + i)


