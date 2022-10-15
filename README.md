# Assignment 1 - distributed in Github Repo e4040-2022Fall-assign1
The assignment is distributed as several jupyter notebooks and a number of directories and subdirectories in utils.

# Students need to follow the instructions below, and they also need to edit the README.md such that key information is shown in it - right after this line
TODO students to add text summarizing the key (high level) modifications that they did to solve the assignment

# Summary of the modifications

1. Task 1:
     * Part-0(Basic):
        * Changes in the task1-basic_classifiers.ipynb.
        * Appended the bias dimension of ones in X_train, X_val, X_test, X_dev, X_train_binary, X_val_binary, X_dev_binary.
        
     * Part 1: Logistic Regression Classifier
        * Changes in the ./utils/classifiers/logistic_regression.py
        * Implemented logistic_regression_loss_naive function via regress loops
        * Implemented logistic_regression_loss_vectorized function via matrix multiplications
        * Implemented sigmoid function via numpy
        
     * Part 2: Softmax Classifier
        * Changes in the ./utils/classifiers/softmax.py
        * Implemented softmax_loss_naive function via regress loops
        * Implemented softmax_loss_vectorized function via matrix multiplications
        * Implemented softmax function via numpy
        * Implemented onehot function via numpy
        * Implemented cross_entropy function via matrix multiplications
      
      * Part 3: Train your classifiers
        * Changes in the ./utils/classifiers/basic_classifiers.py
        * Implemented Stochastic Gradient Descent (mini-batch)in the step function of the BasicClassifier class.
        * Implemented logistic prediction in the predict function of the LogisticRegression class.
        * Implemented softamx prediction in the predict function of the Softmax class.

2. Task 2:
     * Part 1: Basic Layers:
        * Changes in the utils.layer_funcs.py
        * Added all the required code for the affine, Relu, Softmax.
        * Changes in the utils/layer_utils.py
        * Added all the required code for the AffineLayer, DenseLayer.
        
     * Part 2: Two Layer Network
        * Changes in the utils/classifiers/twolayernet.py
        * Added all the required code for the TwoLayerNet such SGD in the step function.
        * Accuracy of TwoLayerNet is around 0.8478
        * Plotted the accuracy history vs Epoch
        * Changes in the utils.classifiers.mlp.py
        * Added all the required code for the MLP such SGD with momentum in the step function.
        * Accuracy of TwoLayerNet is around 0.8728
        * Plotted the accuracy history vs Epoch

3. Task 3:
     * Part 1: Tensorflow MLP
        * Created 4-layer MLP with epoch = 100, hidden_dim1 = 256, hidden_dim2 = 128, hidden_dim3 = 64
        * Accuracy of MLP is around 0.7839
        * Build MLP with tf.keras.models.Sequential
        
     * Part 2: t-SNE 
        * Visualized data that is passed through MLP via tsne with perplexity = 25
        * Cost of t-SNE is around 0.1211
        * Visualized data that is passed through MLP via tsne with perplexity = 50
        * Cost of t-SNE is around 0.0778



4. Task 4:
        

# Detailed instructions how to submit this assignment/homework:
1. The assignment will be distributed as a github classroom assignment - as a special repository accessed through a link
2. A students copy of the assignment gets created automatically with a special name - students have to rename the repo per instructions below
3. The solution(s) to the assignment have to be submitted inside that repository as a set of "solved" Jupyter Notebooks, and several modified python files which reside in directories/subdirectories
4. Three files/screenshots need to be uploaded into the directory "figures" which prove that the assignment has been done in the cloud


## (Re)naming of the student repository (TODO students) 
INSTRUCTIONS for naming the student's solution repository for assignments with one student:
* This step will require changing the repository name
* Students MUST use the following name for the repository with their solutions: e4040-2022Fall-assign1-UNI (the first part "e4040-2022Fall-assign1" will probably be inherited from the assignment, so only UNI needs to be added) 
* Initially, the system will give the repo a name which ends with a  student's Github userid. The student MUST change that name and replace it with the name requested in the point above
* Good Example: e4040-2022Fall-assign1-zz9999;   Bad example: e4040-2022Fall-assign1-e4040-2022Fall-assign1-zz9999.
* This change can be done from the "Settings" tab which is located on the repo page.

INSTRUCTIONS for naming the students' solution repository for assignments with more students, such as the final project. Students need to use a 4-letter groupID): 
* Template: e4040-2022Fall-Project-GroupID-UNI1-UNI2-UNI3. -> Example: e4040-2022Fall-Project-MEME-zz9999-aa9999-aa0000.


# Organization of this directory

```
.
├── Assignment1_intro.ipynb
├── README.md
├── figures
│   ├── tw2834_gcp_work_example_screenshot_1.png
│   ├── tw2834_gcp_work_example_screenshot_2.png
│   └── tw2834_gcp_work_example_screenshot_3.png
├── requirements.txt
├── task1-basic_classifiers.ipynb
├── task2-mlp_numpy.ipynb
├── task3-mlp_tensorflow.ipynb
├── task4-questions.ipynb
└── utils
    ├── classifiers
    │   ├── basic_classifiers.py
    │   ├── logistic_regression.py
    │   ├── mlp.py
    │   ├── softmax.py
    │   └── twolayernet.py
    ├── display_funcs.py
    ├── features
    │   ├── pca.py
    │   └── tsne.py
    ├── layer_funcs.py
    ├── layer_utils.py
    └── train_funcs.py

5 directories, 25 files
```