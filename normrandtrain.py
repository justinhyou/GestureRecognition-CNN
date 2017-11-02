"""
run by:
python normrandtrain.py <path-to-frames> <path-to-transcription> <number-to-normalize-to> <randomize-by> <train_size>

note:  
randomize-by has three options: frame, surgeon, surgery
"""


# Create folders for each gesture
import argparse
import os
import random

# Create argument parser
parser = argparse.ArgumentParser(description="Give the frame directory and transcription directories.")
parser.add_argument('frame_path')
parser.add_argument('trans_path')
parser.add_argument('dest')
parser.add_argument('normalize_to')
parser.add_argument('randomize_by')
parser.add_argument('train_size')
args = parser.parse_args()

# Set up variables given arguments
frame_path = args.frame_path
trans_path = args.trans_path
dest = args.dest
normalize_to = args.normalize_to
randomize_by = args.randomize_by
train_size = args.train_size
validation_size = 1.0 - float(train_size)

# Create an output training folder
os.mkdir(os.path.join(dest, "train"))
training_path = dest + "/train/"

# Create an output validation folder
os.mkdir(os.path.join(dest, "validation"))
validation_path = dest + "/validation/"

# Create a subfolder for each gesture
# There are 15 gestures, plus a label for no gesture present
for i in range(11):
    if not (i == 6):
        gesture_folder_name = "G" + str(i+1)
        os.mkdir(os.path.join(training_path, gesture_folder_name))
        os.mkdir(os.path.join(validation_path, gesture_folder_name))

print("Done creating folders for each gesture")
print("Begin copying frames to these folders...")

# Go through all video folders in the frame directory
for folder in os.listdir(frame_path):
    # Check if capture 1, left camera
    if folder[5] == "1" and folder[0] == "b" and folder[3] == "1":
        print("Working on frames for " + folder)

        # Determine the subject and trial number
        subject = folder[0]
        trial_num = folder[3]

        trans_file_path = trans_path
        # Locate the corresponding transcription file
        for root, dirs, files in os.walk(trans_path):
            for file in files:
                if (file[9].lower() == subject) and (file[12] == trial_num):
                    trans_file_path += file
                    break
        
        trans_file = []
        # Read in the transcription file line-by-line
        with open(trans_file_path) as f:
             trans_file = f.readlines()

        # Format lines into list such that we have [begin_frame, end_frame, gesture]
        ref = []
        for line in trans_file:
            x = line.find(" ")
            begin_frame = int(line[:x])
            line = line[x+1:]
            x = line.find(" ")
            end_frame = int(line[:x])
            line = line[x+1:]
            x = line.find(" ")
            gesture_label = line[:x]
            ref += [(begin_frame, end_frame, gesture_label)]

        print("Finished processing transcription file for " + folder)
        print("Begin copying frames to training and validation folders...")

        # Find the gesture for each frame in the directory and move to
        #  the appropriate training subfolder
        for frame in os.listdir(frame_path + folder):            
            # Look for label of frame in transcription folder
            frame_num = int(frame[7:11])

            # Find the range that this frame number falls in under in transcription file
            frame_label = "G12" # Default to no label
            for r in ref:
                if (r[0] <= frame_num) and (frame_num <= r[1]):
                    frame_label = r[2]
                    break

            # First add all frames to train; toss all unlabelled frames
            if not (frame_label == "G12"):
                path_to_current_frame = frame_path + folder + "/" + frame
                subfolder_name = training_path + frame_label + "/"
                os.system("cp " + path_to_current_frame + " " + subfolder_name)

print("Labelling frames complete. All files currently in train dataset.")
print("Begin normalizing class sizes.")

# Normalize
for folder in os.listdir(training_path):
    duplicate_differentiator = 1
    frame_list = os.listdir(training_path + folder)
    num_files = len(frame_list)
    print(folder + " has " + str(num_files) + " examples.")
    
    # Check if examples exist
    if num_files == 0:
        continue
    max_examples_num = int(normalize_to)

    # If max is greater than files existing, upsample. Otherwise,
    # downsample. Files are duplicated or deleted randomly.
    do_upsample = True
    if max_examples_num < num_files:
        do_upsample = False
        print("Downsampling for " + folder)
    else:
        print("Upsampling for " + folder)

    files_needed_to_normalize = abs(max_examples_num  - num_files)
    subfolder_name = training_path + folder + "/"
    for i in range(files_needed_to_normalize):
        file_index = random.randint(0, num_files-1)
        file_to_duplicate = frame_list[file_index]
        if do_upsample:
            new_file_name = file_to_duplicate[:-4] + str(duplicate_differentiator) + ".jpg"
            duplicate_differentiator += 1
            os.system("cp " + subfolder_name + file_to_duplicate + " "  + subfolder_name + new_file_name)
        else:
            frame_list = frame_list[:file_index] + frame_list[file_index+1:]
            num_files -= 1
            os.system("rm " + subfolder_name + file_to_duplicate)
    new_frame_list = os.listdir(training_path + folder)
    if len(new_frame_list) == max_examples_num:
        print(folder + " has been normalized.")

print("Normalization complete.")
print("Begin dividing into train and validation sets randomly by frame.")

# Divide into training and validation folders based on frame,
# surgeon, or surgery            
if randomize_by == "frame":
    print("Randomly selecting by frame.")
    print("Training size is " + train_size + ", and validation " + str(validation_size))
    for folder in os.listdir(training_path):
        for frame in os.listdir(training_path + folder):
            t_or_v = random.uniform(0.0, 1.0)
            if (t_or_v <= validation_size):
                current_path = training_path + folder + "/" + frame
                new_path = validation_path + folder + "/" + frame
                os.system("mv " + current_path + " " + new_path)
elif randomize_by == "surgeon":
    print("Randomly selecting by surgeon.")
    print("Training size is " + train_size + ", and validation " + str(validation_size))
    # Decide which surgeon/surgeons belong in the validation set
    surgeons = ["b", "c", "d", "e", "f", "g", "h", "i"]
    num_surgeons = len(surgeons)
    surgeons_in_validation_set = []
    percent_validation = 0
    while (percent_validation < validation_size):
        surgeon_choice = random.randint(0, num_surgeons-1)
        surgeons_in_validation_set += [surgeons[surgeon_choice]]
        surgeons = surgeons[:surgeon_choice] + surgeons[surgeon_choice+1:]
        num_surgeons -= 1
        percent_validation = len(surgeons_in_validation_set)/8.0
    print("Actual validation size is " + str(percent_validation))
    print("Validates on the following surgeons:")
    for surgeon in surgeons:
        print(surgeon)
    # Take all frames in which validation surgeons were the subject
    # and move to validation folder
    for folder in os.listdir(training_path):
        for frame in os.listdir(training_path + folder):
            if frame[0] in surgeons_in_validation_set:
                current_path = training_path + folder + "/" + frame
                new_path = validation_path + folder + "/" + frame
                os.system("mv " + current_path + " " + new_path)
elif randomize_by == "surgery":
    print("Randomly selecting by surgery.")
    print("Training size is " + train_size + ", and validation " + str(validation_size))
    # Decide which surgery/surgeries belong in the validation set
    surgeries = [["b", [1,2,3,4,5]],["c", [1,2,3,4,5]],["d", [1,2,3,4,5]],
                 ["e", [1,2,3,4,5]],["f", [1,2,3,4,5]],["g", [1,2,3,4,5]],
                 ["h", [1,3,4,5]],["i", [1,2,3,4,5]]]
    num_surgeries = 39
    num_surgeons = len(surgeries)
    surgeries_in_validation_set = []
    percent_validation = 0
    while (percent_validation < validation_size):
        surgeon_choice = random.randint(0, num_surgeons-1)
        surgeon = surgeries[surgeon_choice][0]
        his_surgeries = surgeries[surgeon_choice][1]
        num_surgeries_for_surgeon = len(his_surgeries)
        surgery_choice = random.randint(0, num_surgeries_for_surgeon-1)
        selected_surgery = (surgeon, his_surgeries[surgery_choice])
        surgeries_in_validation_set += [selected_surgery]
        surgeries[surgeon_choice][1] = his_surgeries[:surgery_choice] + his_surgeries[surgery_choice+1:]
        # If all of the surgeon's surgeries are in the validation set,
        # remove him from the possible choices
        if len(surgeries[surgeon_choice][1]) == 0:
            surgeries = surgeries[:surgeon_choice] + surgeries[surgeon_choice+1:]
            num_surgeons -= 1
        percent_validation = len(surgeries_in_validation_set)/39.0
    print("Actual validation size is " + str(percent_validation))
    print("Validates on following surgeries:")
    for surgery in surgeries_in_validation_set:
        print(surgery[0] + str(surgery[1]))
    for folder in os.listdir(training_path):
        for frame in os.listdir(training_path + folder):
            surgeon = frame[0]
            surgery = int(frame[3])
            selected_surgery = (surgeon, surgery)
            if selected_surgery in surgeries_in_validation_set:
                current_path = training_path + folder + "/" + frame
                new_path = validation_path + folder + "/" + frame
                os.system("mv " + current_path + " " + new_path)

print("Datasets are ready.")

