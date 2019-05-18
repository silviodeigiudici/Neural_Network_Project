# Neural_Network_Project
This is the project of neural network wrote by Marco Morella, Fortunato Tocci and Silvio Dei Giudici.

There are three main files in this repo called: 
-targeted_attack.py; 
-non-targeted_attack_imagenet.py;
-non-targeted_attack.py;

These files run the same version of the algorithm with some differences for type of attacks (targeted, non-targeted) and a special case file for imagenet, because it needs some dedicated code to extract the dataset's infos.

In all the files there is a delimited zone in the end of the file used to change the parameters of the algorithm, for example the number of pixels, number of images to test, which network run ecc.

The repo isn't provided of the files containing weights of the networks, you need to download them: https://drive.google.com/drive/folders/1m_oDfM64GSjwmfvN-xoUgr8kgGHalCAj. Then you can put them in 'networks/[network_name]' folder.
