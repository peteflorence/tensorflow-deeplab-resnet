import os

train_list = []
# train_list.append("sixobjects_single_train_18_scenes")
# train_list.append("sixobjects_multi_train_3_scenes")
# train_list.append("sixobjects_multi_train_18_scenes")
# train_list.append("sixobjects_mixed_train_21_scenes")
# train_list.append("sixobjects_mixed_train_36_scenes")
# train_list.append("sixobjects_multi_test_scenes")
# train_list.append("sixobjects_single_test_scenes")

train_list.append("sixobjects_mixed_train_36_scenes_30_00hz")
train_list.append("sixobjects_mixed_train_36_scenes_03_00hz")
train_list.append("sixobjects_mixed_train_36_scenes_00_30hz")
train_list.append("sixobjects_mixed_train_36_scenes_00_03hz")

for i in train_list:
	os.system("python train.py --not-restore-last --data-list " + "/home/corl2017/spartan/src/CorlDev/experiments/"+i+".txt.imglist.txt --snapshot-dir ./snapshots_hz_"+i)


