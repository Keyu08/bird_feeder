# Bird Feeding AI

***A Bird Species Detector and a Feeding Recommendation Model***

This model is designed to identify various bird species based on their characteristics. It utilizes a Resnet-18 model pre-trained on ImageNet and further trained using transfer learning. The goal of this AI model is to assist in identifying different bird species that are frequently encountered in our surroundings and to suggest appropriate types of food to attract and nourish them (if desired). Many bird enthusiasts and backyard birdwatchers face this challenge of identifying birds and providing the right sustenance to attract them. However, with Bird Feeding AI, they can finally *wing* it with confidence when it comes to both identification and dining recommendations!

> ![image](https://github.com/Keyu08/bird_feeder/assets/141778196/b716ed17-260a-4dd7-8434-6bf3c1dc6273) ![image](https://github.com/Keyu08/bird_feeder/assets/141778196/2a566acd-0d62-49c8-868d-30e9643dbaa9) ![image](https://github.com/Keyu08/bird_feeder/assets/141778196/10209143-ee3a-442f-b240-cdc675fdcc45) ![image](https://github.com/Keyu08/bird_feeder/assets/141778196/8e9910d1-9534-4f0c-99f0-ef913cdca6c5) ![image](https://github.com/Keyu08/bird_feeder/assets/141778196/465552aa-912b-40a5-b57f-50a1552cbe3e) ![image](https://github.com/Keyu08/bird_feeder/assets/141778196/9d388696-490a-4f9e-b924-a4be7895a999)

## The Algorithm
This project utilizes a resetnet18 model that was retrained with 21 different sets of data (21 types of bird). After training the model for 166 epochs and exporting it in ONNX format, it was able to identify, with a slightly modified version of imagenet.py [^1], the name of a bird and the food that can be fed to it. Although making small mistakes ever so often, overall, the model is still rather accurate. 

[^1]:It was modified to give as output not only the name of the bird and the confidence, but also the bird's diet (the different foods that we can feed it) (See example pictures above).

The 21 sets of data are (bird classes): 
1. American_Crow
2. American_Goldfinch
3. American_Robin
4. Black_Capped_Chickadee
5. Blue_Jay
6. Brown_Headed_Cowbird
7. Common_Grackle
8. Dark_Eyed_Junco
9. Downy_Woodpecker
10. European_Starling
11. Hairy_Woodpecker
12. House_Finch
13. House_Sparrow
14. Male_Northern_Cardinal
15. Mourning_Dove
16. Purple_Finch
17. Red_Bellied_Woodpecker
18. Red_Winged_Blackbird
19. Song_Sparrow
20. Tufted_Titmouse
21. White_Breasted_Nuthatch

The dataset used to retrain: https://www.kaggle.com/datasets/otherkirby/rhode-island-backyard-birds

## Running this project
**Initiating VS Code**

1. Click on the small blue icon at the bottom left of your screen to access the SSH menu.
2. Select + Add New SSH Host to add a new host.
3. Enter ssh nvidia@x.x.x.x, replacing x.x.x.x.x with the IP address you usually use in Putty or terminal to connect to the Nano.
4. Pick the first configuration file.
5. Click Connect in the prompted window.
6. Choose Linux as the operating system when asked.
7. If you're asked to continue, click Continue.
8. You'll be asked for a password after connecting to the Nano. Input your Nano password and hit Enter.
9. Select Open Folder and navigate to jetson-inference. Input your password again if required.
10. Click Yes, I trust the authors to access and start working on your projects in this directory.
## Preparing the Dataset
1. Navigate to jetson-inference/python/training/classification/data.
2. Put the downloaded dataset into the Nano using Filezilla.
3. Extract the dataset ZIP file.

**Check if the following steps are already done or not. If not, follow those steps:**
1. Inside jetson-inference/python/training/classification/data, create a new folder called bird_classification. Inside bird_classification, add three folders: test, train, val. Also add a file named labels.txt.
2. In the train directory inside bird_classification, create 21 folders for the 21 sets of data for each bird class (the 21 sets are above, in the algorithm part).
3. Copy these folders to the val and test directories.
4. Distribute (If not already organized) the images from your ZIP file among these folders, with 80% in the train folder, 10% in the val folder, and 10% in the test folder for each bird type.
## Running the Docker Container
1. Go to the jetson-inference folder and run ./docker/run.sh.
2. Once inside the Docker container, navigate to jetson-inference/python/training/classification.
## Training the Neural Network
1. Run the training script with the following command: python3 train.py --model-dir=models/bird_classification data/bird_classification --epoch=1. You can change the number of epoch as you wish. This process may take quite some time. 
2. You can stop the process at any time using Ctl+C and resume it later using the --resume and --epoch-start flags.
## Export the model in ONNX format
1. Once you finished training, make sure you are still in the docker container and in jetson-inference/python/training/classification.
2. Run this script: python3 onnx_export.py --model-dir=models/bird_classification
## Testing the Trained Network on Images
1. Exit the Docker container by pressing Ctrl + D in the terminal.
2. On your Nano, navigate to jetson-inference/python/training/classification.
3. Check if the model exists on the Nano by executing ls models/bird_classification/. You should see a file named resnet18.onnx.
4. Set the NET and DATASET variables: NET=models/bird_classification DATASET=data/bird_classification
5. Run this command to see how the model works on an image from the test folder: python3 imagenet_mod.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/PICK_A_CLASS/PICK_AN_IMAGE.jpg PICK_A_NAME_FOR_THE_IMAGE.jpg. Keep in mind that you are able to choose the bird class by replacing PICK_A_CLASS with the class, you are able to pick any test image of that class by changing PICK_AN_IMAGE.jpg, and are able to change the name of the output image name by changing PICK_A_NAME_FOR_THE_IMAGE.jpg.
6. To view the image output, find and open PICK_A_NAME_FOR_THE_IMAGE.jpg (or the changed output image name)

https://youtu.be/6ufqk7kHM4k Video Link

[View a video explanation here](video link)
