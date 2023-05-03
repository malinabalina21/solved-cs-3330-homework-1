Download Link: https://assignmentchef.com/product/solved-cs-3330-homework-1
<br>



Data Processing and Memory Augmented Neural Networks




<h1>Overview</h1>

In this assignment we will be looking at meta-learing for few shot classification. You will

<ul>

 <li>Learn how to process and partition data for meta learning problems, where training is done over a distribution of training tasks <em>p</em>(T ).</li>

 <li>Implement and train memory augmented neural networks, which meta-learn through a recurrent network</li>

 <li>Analyze the learning performance for different size problems</li>

 <li>Experiment with model parameters and explore how they improve performance.</li>

</ul>

We will be working with the Omniglot dataset [1], a dataset for one-shot learning which contains 1623 different characters from 50 different languages. For each character there are 20 28×28 images. We are interested in training models for <em>K</em>-shot, <em>N</em>-way classification, that is, we want to train a classifier to distinguish between <em>N </em>previously unseen characters, given only <em>K </em>labeled examples of each character.

<strong>Submission</strong>: To submit your homework, submit one pdf report and one zip file to GradeScope, where the report will contain answers to the deliverables listed below and the zip file contains your code (hw1.py, load data.py) with the filled in solutions. <strong>Code Overview: </strong>The code consists of two files

<ul>

 <li>load data.py: Contains code to load batches of images and labels</li>

 <li>py: Contains the network architecture/loss functions and training script.</li>

</ul>

There is also the omniglot resized folder which contains the data. <em>You should not modify this folder.</em>

<strong>Dependencies: </strong>We expect code in Python 3.5+ with Pillow, scipy, numpy, tensorflow installed.

<h1>Problem 1: Data Processing for Few-Shot Classification</h1>

Before training any models, you must write code to sample batches for training. Fill in the sample batch function in the DataGenerator class in the load data.py file. The class

<h2>        X1 , Y1                  X2 , Y2                         XN*K , YN*K                       X1 , 0               X2 , 0                       XN , 0</h2>

Figure 1: Feed <em>K </em>labeled examples of each of <em>N </em>classes through network with memory. Then feed final set of <em>N </em>examples and optimize to minimize loss.

already has variables defined for batch size batch size (<em>B</em>), number of classes num classes (<em>N</em>), and number of samples per class num samples per class (<em>K</em>). Your code should

<ol>

 <li>Sample <em>N </em>different classes from either the specified train, test, or validation folders.</li>

 <li>Load <em>K </em>images per class and collect the associated labels</li>

 <li>Format the data and return two numpy matrices, one of flattened images with shape[<em>B,K,N,</em>784] and one of one-hot labels [<em>B,K,N,N</em>]</li>

</ol>

Helper functions are provided to (1) take a list of folders and provide paths to image files/labels, and (2) to take an image file path and return a flattened numpy matrix.

<h1>Problem 2: Memory Augmented Neural Networks [2, 3]</h1>

We will be attempting few shot classification using memory augmented neural networks. The idea of memory augmented networks is to use a classifier with recurrent memory, such that information from the <em>K </em>examples of unseen classes informs classification through the hidden state of the network.

The data processing will be done as in SNAIL [3]. Specifically, during training, you sample batches of <em>N </em>classes, with <em>K</em>+1 samples per batch. Each set of labels and images are concatenated together, and then all <em>K </em>of these concatenated pairs are sequentially passed through the network. Then the final example of each class is fed through the network (concatenated with 0 instead of the true label). The loss is computed between these final outputs and the ground truth label, which is then backpropagated through the network. <strong>Note</strong>: The loss is <em>only </em>computed on the last set of <em>N </em>classes.

The idea is that the network will learn how to encode the first <em>K </em>examples of each class into memory such that it can be used to enable accurate classification on the <em>K </em>+ 1th example. See Figure 1.

In the hw1.py file:

<ol>

 <li>Fill in the call function of the MANN class to take in image tensor of shape [<em>B,K </em>+</li>

</ol>

1<em>,N,</em>784] and a label tensor of shape [<em>B,K </em>+ 1<em>,N,N</em>] and output labels of shape [<em>B,K </em>+ 1<em>,N,N</em>]. The layers to use have already been defined for you in the init function. <em>Hint: Remember to pass zeros, not the ground truth labels for the final N examples.</em>

<ol start="2">

 <li>Fill in the function called loss function which takes as input the [<em>B,K </em>+ 1<em>,N,N</em>] labels and [<em>B,K </em>+ 1<em>,N,N</em>] and computes the cross entropy loss.</li>

</ol>

<strong>Note</strong>: Both of the above functions will need to backpropogated through, so they need to be written in differentiable tensorflow.

<h1>Problem 3: Analysis</h1>

Once you have completed problems 1 and 2, you can train your few shot classification model.

For example run python hw1.py –num classes=2 –num samples=1 –meta batch size=4 to run 1-shot, 2-way classification with a batch size of 4. You should observe both the train and testing loss go down, and the test accuracy go up.

Now we will examine how the performance varies for different size problems. Train models for the following values of <em>K </em>and <em>N</em>.

<em>K </em>= 1, <em>N </em>= 2

<em>K </em>= 1, <em>N </em>= 3

<em>K </em>= 1, <em>N </em>= 4

<em>K </em>= 5, <em>N </em>= 4

For each configuration, submit a plot of the test accuracy over iterations. Note your observations.

<h1>Problem 4: Experimentation</h1>

<ul>

 <li>Experiment with one parameter of the model that affects the performance of the model,such as the type of recurrent layer, size of hidden state, learning rate, number of layers. Show learning curves of how the test success rate of the model changes on 1-shot, 3-way classification as you change the parameter. Provide a brief rationale for why you chose the parameter and what you observed in the caption for the graph.</li>

 <li><strong>Extra Credit: </strong>You can now change the MANN architecture however you want (including adding convolutions). Can you achieve over 60% test accuracy on 1-shot, 5-way classification?</li>

</ul>

<h1>References</h1>

<ul>

 <li>Brenden M. Lake, Ruslan Salakhutdinov, and Joshua B. Tenenbaum. Human-level concept learning through probabilistic program induction. <em>Science</em>, 350(6266):1332–1338, 2015.</li>

 <li>Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In Maria Florina Balcan and Kilian Q. Weinberger, editors, <em>Proceedings of The 33rd International Conference on Machine Learning</em>, volume 48 of <em>Proceedings of Machine Learning Research</em>, pages 1842–1850, New York, New York, USA, 20–22 Jun 2016. PMLR.</li>

 <li>Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. Meta-learning with temporal convolutions. <em>CoRR</em>, abs/1707.03141, 2017.</li>

</ul>