# PyTorch-Arrow-Training
Using Python, Pytorch, MatPlotLib, Numpy, and OpenCV to train a neural network to recognize digits in various image compositions.
Completed for the AIT500 course at Seneca Polytechnic during the Winter 2025 semester.
Images obtained through the public domain from Kaggle: https://www.kaggle.com/datasets/jithinnambiarj/directions?resource=download

Project Overview
--
Dataset Processing
---
The dataset I used was a compilation of >3500 arrow images of varying styles and sizes (retrieved from https://www.kaggle.com/datasets/jithinnambiarj/directions?resource=download). I used this dataset to train the neural network for image recognition. I ended up using two folders: one folder (“arrow_images”) for the training data; one folder containing a set of arrows that I removed from the dataset to be used solely for testing (“unseen_arrow_images”). The latter folder thus contains arrows that the model was never trained on.
Other processing was writing a Python script that simply renamed the image files to contain their labels (“forward”, “backward”, “left”, and “right”) so that my code would properly identify the arrow type depicted in a given image.
Lastly, I ensured all images were made 88*88 pixels using the cv2 (OpenCV) library.

Neural Network Architecture
---
The neural network model I constructed is comprised of:
* 7,744 neurons in the input layer (88x88 pixels)
* 128 neurons in the first hidden layer (self.fc1)
* 64 neurons in the second hidden layer (self.fc2)
* 4 neurons in the output layer (reflecting number of all possible arrow types)

Training and Evaluation
---
The training and evaluation procedure was as follows:
* Loaded 3,065 training images and 282 testing images (testing images not present in training data)
* Split the training images into one training group using 80% of images (2,452) and a second group using the remaining 20% (613 images) for validation
* PyTorch DataLoader configured to load a batch size of 4 images per iteration
* Ran 25 training epochs, applying ReLu function to hidden layers
* Applied CrossEntropyLoss and Adam optimizer
* Metrics logged after each epoch
After configuration, the model learned by first training itself on the training images, then testing itself against the test (unseen) images.

Results and Accuracy
--
The end result of this neural network’s training process was quite fruitful. The model scored an average recognition metric of 89.36%, with subsequent runs yielding scores anywhere between ~85.5% to ~90.5%. The output is pictured below:
![epoch training results showing an average accuracy of 89.36%](https://github.com/Adam-Beik/PyTorch-Arrow-Training/blob/main/training_output.jpg)
*Figure 1: Neural network accuracy output after 25 epochs.*
The resulting graphs of loss-over-epochs and accuracy-over-epochs are pictured below:
![graphs of loss-over-epochs and accuracy-over-epochs, respectively, accuracy output](https://github.com/Adam-Beik/PyTorch-Arrow-Training/blob/main/arrow_training_curves.png)
*Figure 2: Graphs depicting loss-over-epochs and accuracy-over-epochs of the model, respectively.* 
