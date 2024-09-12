# Cyber Threat Detection Using Artificial Intelligence 


Student Names:

Muhammed Joharji    2037729

Muhannad Al-Jaddawi 2036459

Hashem Baroom       2037062

------------------------------------------------------

In response to the accelerating sophistication of cyber threats, this research endeavors to enhance cybersecurity through the implementation of artificial intelligence (AI) techniques, specifically focusing on the domain of anomaly detection. 
The project’s primary objective is to develop strong AI models capable of effectively identifying unusual patterns and potential cyber threats within network logs.

![CS499_G#9_Poster](https://github.com/user-attachments/assets/42e79fe0-d2be-4954-968b-dadf5336b8a9)

------------------------------------------------------

The dataset used on the code can be downloaded through this link: https://drive.google.com/file/d/105OEx5gLW6Pjws1HLLr1CgCjhX-gSUH1/view?usp=drive_link (43 MB Rar file). 

Open "AvastDataset" and copy "RealDataset" and paste it and overrite "RealDataset" folder inside the project folders. You can also refer to Kaggle website to read about the dataset from here: https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis/data?select=CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv

There are various simple steps you have to follow to be able to run the code. *Note: please refer to Hardware Recommendation from the report in Chapter 3, because the code might not run properly due to this or will take longer to run, Please have patient.

You can download training dump files for deep learning and machine learning where you can use trained data to test thier result without waiting for training process through this link: https://drive.google.com/file/d/1CQJl-K0MxiN4yuW4gOhJSgR2Hft3i4_T/view?usp=drive_link

Python Version: 3.11 (base)

# Pre-requisites: 
You need to install all libraries and tools needed and they are as follows: 
 - Jupyter from Anaconda Navigator: https://www.anaconda.com/products/navigator , you can use Visual Studio Code with jupyter extension but that requires installation of all libraries.

Remember: installing python libraries in Jupyter Notebook using !pip and for Visual Studio Code using %pip 

Libraries needed: 
 - Scikit-Learn 
 - Matplotlib
 - Seaborn
 - Tensorflow
 - Keras
 - Pandas
 - Numpy
 - xgboost
 - joblib

 Please install all these libraries to be able to run the code.

------------------------------------------------------

# Files explanation:
 - Threat Detection.ipynb is the main part for prediction. 
 - ModelTraining.ipynb is the training file for all models with save functionality.
 - preprocessing_data.py is a python file that has function that preprocess the data of same structure automatically.

# Folders Explanation:
 - SplitDatasets folder, contains all the splits that are going to be used for training purposes.
 - TrainingDumpData folder, contains all the files for loading the models. It consists of .joblib and .weights which are the data of training model.
 - functions folder, contains the python files that are used as functions.
 - RealDataset folder, contains all the datasets used for training and testing purposes.

# How to run?:

 To run the program you need a dataset first of same structure you download the same dataset used through the link given previously.

 1- Run ThreatDetection.ipynb till Cell 17 (Before XGBoost Model). 
 2- After that Run ModelTraining.ipynb (This may take a while). 

 The reason behind manual runs is to showcase the process of training and how each epochs upgrades.

 3- Continue running ThreatDetection.ipynb till the end to see the result and analysis. (To understand what the analysis showcases, please refer to the final repor / documentation file).

Note: In case of adding training dump files, you are only required to run ThreatDetection.ipynb.

You can check The Final Report of the Project: "Full Documentation.pdf".

Thank you!
