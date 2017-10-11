# Create folders for each gesture
import argparse
import os
import random

# Create argument parser
parser = argparse.ArgumentParser(description="Give the frame directory and transcription directories.")
parser.add_argument('frame_path')
parser.add_argument('trans_path')
parser.add_argument('dest')
args = parser.parse_args()

# Set up variables given arguments
frame_path = args.frame_path
trans_path = args.trans_path
dest = args.dest

# Create an output training folder
os.mkdir(os.path.join(dest, "train"))
training_path = dest + "/train/"

# Create an output validation folder
os.mkdir(os.path.join(dest, "validation"))
validation_path = dest + "/validation/"

# Create a subfolder for each gesture
# There are 15 gestures, plus a label for no gesture present
for i in range(11):
    gesture_folder_name = "G" + str(i+1)
    os.mkdir(os.path.join(training_path, gesture_folder_name))
    os.mkdir(os.path.join(validation_path, gesture_folder_name))

print("Done creating folders for each gesture")
print("Begin copying frames to these folders...")

# Go through all video folders in the frame directory
for folder in os.listdir(frame_path):
    # Check if capture 1, left camera
    if folder[5] == "1":
        print("Working on frames for " + folder)

        # Determine the subject and trial number
        subject = folder[0]
        trial_num = folder[3]

        trans_file_path = trans_path+'/'
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
        # Remove whitespace characters at the end of each line

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
        for frame in os.listdir(frame_path+'/'+folder):            
            # Look for label of frame in transcription folder
            frame_num = int(frame[7:11])

            # Find the range that this frame number falls in under in transcription file
            frame_label = "G12" # Default to no label
            for r in ref:
                if (r[0] <= frame_num) and (frame_num <= r[1]):
                    frame_label = r[2]
                    break

            # For now, toss all unlabelled frames
            if not (frame_label == "G12"):
                # Randomly decide whether data point will be put in training data
                #  or validation data. Currently randomized by frame (alternatively,
                #  we could randomize by surgeon or surgery)
                t_or_v = random.uniform(0.0,1.0)

                path_to_current_frame = frame_path+ "/" + folder + "/" + frame
                # Currently, 80-20 for training to validation
                if (t_or_v <= 0.8):
                    # Copy the frame to the appropriate subfolder in training
                    #  given its label
                    subfolder_name = training_path + "/"+ frame_label + "/"
                    os.system("cp " + path_to_current_frame + " " + subfolder_name)
                else:
                    subfolder_name = validation_path+ "/" + frame_label + "/"
                    os.system("cp " + path_to_current_frame + " " + subfolder_name)    


