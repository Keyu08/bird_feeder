#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--topK", type=int, default=1, help="show the topK number of class predictions (default: 1)")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the recognition network
net = imageNet(args.network, sys.argv)

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
#                 input_blob="input_0", output_blob="output_0")

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)
font = cudaFont(size=15)



# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # classify the image and get the topK predictions
    # if you only want the top class, you can simply run:
    #   class_id, confidence = net.Classify(img)
    predictions = net.Classify(img, topK=args.topK)

    total_label_height = len(predictions) * font.GetSize()
    starting_y = img.shape[0] - total_label_height - 44
    starting_y2 = img.shape[0] - total_label_height - 22
    starting_y3 = img.shape[0] - total_label_height


    # draw predicted class labels
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        confidence *= 100.00

        print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")

        if classID == 0:
            print('You can feed it fruits, grains, or insects')
            font.OverlayText(img, text=f"You can feed it fruits, grains,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"or insects", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 1:
            print('You can feed it seeds (Nyjer seed is a favorite), or sunflower hearts and chips')
            font.OverlayText(img, text=f"You can feed it seeds", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(Nyjer seed is a favorite), or", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"sunflower hearts and chips", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 2:
            print('You can feed it earthworms, insects, berries, or fruits (raisins, chopped apples)')
            font.OverlayText(img, text=f"You can feed it earthworms,", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"insects, berries, or fruits", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(raisins, chopped apples)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 3:
            print('You can feed it insects, seeds, berries, suet, or peanut butter')
            font.OverlayText(img, text=f"You can feed it insects, seeds,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"berries, suet, or peanut butter", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 4:
            print('You can feed it seeds, peanuts, berries, or suet')
            font.OverlayText(img, text=f"You can feed it seeds, peanuts,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"berries, or suet", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 5:
            print('You can feed it seeds, suet, or insects')
            font.OverlayText(img, text=f"You can feed it seeds, suet,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"or insects", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 6:
            print('You can feed it seeds, or insects (its favorite food)')
            font.OverlayText(img, text=f"You can feed it seeds, or insects", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(its favorite food)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 7:
            print('You can feed it insects, cracked corn, or seeds (millet, sunflower seeds...)')
            font.OverlayText(img, text=f"You can feed it insects,", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"cracked corn, or seeds", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(millet, sunflower seeds...)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 8:
            print('You can feed it insects, suet, berries, peanuts, or seeds (sunflower seeds)')
            font.OverlayText(img, text=f"You can feed it insects, suet,", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"berries, peanuts, or seeds", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(sunflower seeds)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 9:
            print('You can feed it suet, berries, or insects')
            font.OverlayText(img, text=f"You can feed it suet, berries,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"or insects", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 10:
            print('You can feed it suet, peanuts, berries, insects, nuts, or seeds (sunflower seeds)')
            font.OverlayText(img, text=f"You can feed it suet, peanuts,", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"berries, insects, nuts, or seeds", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(sunflower seeds)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 11:
            print('You can feed it buds, berries, or seeds (sunflower seeds, safflower seeds, nyjer seeds)')
            font.OverlayText(img, text=f"You can feed it buds, berries,", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"or seeds (sunflower seeds,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"safflower seeds, nyjer seeds)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 12:
            print('You can feed it grains, insects, cracked corn, or seeds (millet, sunflower seeds)')
            font.OverlayText(img, text=f"You can feed it grains, insects,", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"cracked corn, or seeds", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(millet, sunflower seeds)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 13:
            print('You can feed it insects, berries, seeds (sunflower seeds, safflower seed), grains, buds, or fruits')
            font.OverlayText(img, text=f"You can feed it insects, berries,", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40) 
            font.OverlayText(img, text=f"seeds (sunflower seeds, safflower", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"seeds), grains, buds, or fruits", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 14:
            print('You can feed it seeds (millet), peanuts, or grains (cracked corn)')
            font.OverlayText(img, text=f"You can feed it seeds (millet),", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"peanuts, or grains (cracked corn)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 15:
            print('You can feed it insects, berries, or seeds (sunflower seeds, nyjer seeds)')
            font.OverlayText(img, text=f"You can feed it insects, berries,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"or seeds (sunflower & nyjer seeds)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 16:
            print('You can feed it insects, nuts, suet, or seeds (sunflower seeds)')
            font.OverlayText(img, text=f"You can feed it insects, nuts,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"suet, or seeds (sunflower seeds)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 17:
            print('You can feed it seeds (sunflower seeds), fruits, berries, or insects')
            font.OverlayText(img, text=f"You can feed it seeds", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(sunflower seeds), fruits,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(berries, or insects", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 18:
            print('You can feed it insects, seeds (sunflower seeds, millet), or cracked corn')            
            font.OverlayText(img, text=f"You can feed it insects, seeds", 
                            x=5, y=starting_y + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"(sunflower seeds, millet),", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"or cracked corn", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 19:
            print('You can feed it insects, suet, peanuts, or seeds (sunflower seeds)')
            font.OverlayText(img, text=f"You can feed it insects, suet,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"peanuts, or seeds (sunflower seeds)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
        if classID == 20:
            print('You can feed it insects, suet, peanuts, or seeds (sunflower seeds)')
            font.OverlayText(img, text=f"You can feed it insects, suet,", 
                            x=5, y=starting_y2 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)
            font.OverlayText(img, text=f"peanuts, or seeds (sunflower seeds)", 
                            x=5, y=starting_y3 + n * (font.GetSize()),
                            color=font.White, background=font.Gray40)

        font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}, ", 
                         x=5, y=5 + n * (font.GetSize()),
                         color=font.White, background=font.Gray40)

                         
    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
