import os

train_list = []
train_list.append("drill_1_train_scenes")
train_list.append("drill_2_train_scenes")
train_list.append("drill_5_train_scenes")
train_list.append("drill_10_train_scenes")
train_list.append("drill_25_train_scenes")
train_list.append("drill_50_train_scenes")
train_list.append("drill_11_test_scenes")

for i in train_list:
	os.system("python inference_dir.py --train_set "+i)