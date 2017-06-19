import os

train_list = []

train_list.append("sixobjects_01_single_train_18_scenes")
train_list.append("sixobjects_02_multi_train_3_scenes")
train_list.append("sixobjects_03_multi_train_18_scenes")
train_list.append("sixobjects_04_mixed_train_21_scenes")
train_list.append("sixobjects_05_mixed_train_36_scenes")


for i in train_list:
	os.system("python inference_dir.py --train_set "+i)