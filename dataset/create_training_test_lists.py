# This script reads all of the files in ./train/
# that are saved in our CORL17 format
# and outputs a file that lists all files formatted
# for SegNet
#
###  Input: a subdirectories ./train/ and ./test/ with *rgb.png and *labels.png files (generated by preceding CORL17 pipeline)
###  Output: a pair of .txt files list all *.rgb and *.labels pairs (for SegNet training and testing)

import os

def WritePairToFile(rgb_file_name, labels_file_name, target):
	target.write(rgb_file_name)
	target.write(" ")
	target.write(labels_file_name)
	target.write("\n")

def createDatasetList(list_type):
    text_file = list_type + ".txt"
    target = open(text_file, 'w')
    cwd = os.getcwd()
    path_to_folder = cwd + "/" + list_type
    rgb_match = ""
    labels_match = ""
    for root, dirs, files in os.walk(path_to_folder):
        for filename in sorted(files):
            filename_full_path = os.path.join(root, filename)
            print filename_full_path
            
            if filename_full_path.endswith("rgb.png"):
                print "found rgb match"
                rgb_match = filename_full_path

            if filename_full_path.endswith("labels.png") and not filename_full_path.endswith("color_labels.png"):
                print "found labels match"
                labels_match = filename_full_path

            if rgb_match.split("_")[0] == labels_match.split("_")[0]:
                rgb_split = rgb_match.split("_")
                if len(rgb_split)>1 and rgb_split[1] == "rgb.png":
                    print "found a pair"
                    WritePairToFile(rgb_match, labels_match, target)
                    rgb_match = ""
                    labels_match = ""
    target.close()

print "Opening the training set descriptor file..."
createDatasetList("train")

print "Opening the test set descriptor file..."
createDatasetList("test")
