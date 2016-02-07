Deep Learning Models for multi-speaker Text-to-speech systems with different English accents (British and Scottish).

This is a brief guide to run the codes developed for addressing the problem of acoustic modelling of multi-speaker DNN based TTS systems. The overall project consists on a novel multi-speaker DNN architecture for the acoustic realisation of TTS systems trained with a multiple speaker corpora of 10 English Native speakers. Moreover, it includes an alternative approach to train this DNN architecture, which is based on averaging the learning of each speaker during the training stage. 

An extensive discussion of the work developed within this project  is included in the “dissertation.pdf” file.

 
All scripts are written in Python; Theano is the DL framework used to train DNNs. 

Files included are:


1.binary_io.py
2.C00.py
3.C00_Multi.py
4.C01_Multi.py
5.C02.py
6.C03_Multi_Microsoft.py
7.layers.py
8.Main_multispeaker_1speaker_accent.py
9.Main_multispeaker_2speakers_accent.py
10.Main_multispeaker_3speakers_accent.py
11.Main_multispeaker_4speakers_accent.py
12.Main_multispeaker_5speakers_accent.py
13.Main_proposed_multispeaker_1speaker_accent.py
14.Main_proposed_multispeaker_2speakers_accent.py
15.Main_proposed_multispeaker_3speakers_accent.py
16.Main_proposed_multispeaker_4speakers_accent.py
17.Main_proposed_multispeaker_5speakers_accent.py
18.Main_speaker_dependent.py
19.Multi_3_512_6.py


N.B. Scripts are run in GPU mode using GPU servers from UoE, so some setting steps need to be performed before running any script in UoE GPU servers.   

	1. We need to connect to a GPU server: This is made by open a terminal and type ssh lazar
	2. After that we need to move to the directory where the files are located. We can make this set by using cd <directory>
	3. Then we need to type the following lines in order to set the GPU mode of our server:

﻿export PATH=/usr/lib64/qt-3.3/bin:/usr/lpp/mmfs/bin:/usr/local/bin/:/opt/cuda-6.5.19/bin:/opt/matlab-R2014a/bin/:/usr/local/sbin:/usr/bin:/bin:/opt/sicstus-4.0.1/bin
THEANO_FLAGS=cuda.root=/opt/cuda-6.5.19,mode=FAST_RUN,device=gpu0,floatX=float32
export THEANO_FLAGS
        
        5. We also need to type our USERNAME and PASSWORD to get access to the Eddie server. We can type this information in scripts "C00.py" line  73 and "C00_Multi.py" line 99.

	4. After that we can run any of the scripts listed above and obtain the results we have presented in the dissertation.pdf file. As an example, if we want to run the file "myscript.py" we need to type in the terminal python myscript.py



-The file ''Main_speaker_dependent.py'' build a speaker dependent model either for Scottish or English accent. In order to set the type of accent we need to define the input as  "Scottish" or "English" in the line 404 of this code. Moreover, if we want to modify the configuration of the DNN architecture we can modify the array 'hidden_layers_sizes' located in the line 226 of the same code. 

As an example, if we want to define a DNN model of 3 layers with 128 nodes each, we set the array hidden_layers_sizes as:
	hidden_layers_sizes = [128,128,128]


-The file "Multi_3_512_6.py" is the code we developed for building the naive multi-speaker DNN model. This particular script is set to train a 3-speaker dependent model with Scottish accent. The architecture is formed by 6 layers of 512 nodes each. If we want to modify this script to set 1 to 5 speakers we can modify line 404 and 408 and change the actual number of speakers 3 for the number we are interested to test of this approach. Moreover, we can also set either a model with English accent or Scottish accent by modifying the same line 404 and type either "Scottish" or "English" accent.
	

-The files ''Main_multispeaker_<X>speaker_accent.py'' contain the code for running the multi-speaker DNN model implemented and tested by Yuchen Fan et al (2015). This code test the performance of the DNN model either for 2,4,6,8 or 10 speakers. 

Thus, if we want to run a 2-speaker DNN model we should run the script  ''Main_multispeaker_1speaker_accent.py'' as we have referenced the name of the file to the number of speakers per accent.

-The files "Main_proposed_multispeaker_<X>speakers_accent.py" contain the code developed for running our proposed multi-speaker DNN architecture using the averaged weights approach for training our model. This code test the performance of the DNN model either for 2,4,6,8 or 10 speakers. 

Thus, if we want to run a 2-speaker DNN model we should run the script  ''Main_proposed_multispeaker_1speaker_accent.py'' as we have referenced the name of the file to the number of speakers per accent.



 




