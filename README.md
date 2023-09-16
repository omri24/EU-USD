# EU-USD

## The CSV files

training set = EU price, in every minute, since 2001, unill may 2023.
testing set = EU price, in every minute in june and july 2023.
l1, l2,...,l6 = the weights of the NN- layers

## The algorithm
This algorihm separates the datasets into groups of 72 samples (which is about 1/20 of a day). Then it estimates the average of the next group, from all the data samples of the current group, using a NN.

## The results

Below are the predictions in compare to real values. There is a big bias for this algorithm, in the sense of big difference between the average of the predictions and the average of the real values that the algorithm tries to predict.

![image](https://github.com/omri24/EU-USD/assets/115406253/23f6bf3f-288f-4b61-a64c-f6314c6822b5)

Below are the predictions of the algorithm, after shifting all the red dots up, such that now the average of the red dots is equal to the average of the blue dots.
This action "manually" fixes the bias, but doesn't change the shapes that the red dots and the blue dots create. 
In other words, the shapes of the "discrete curves" that the points create remain unchanged. 
After this fix, it is possible to see that the shapes created by the blue and the red dots overlap each other, in parts of the plot.

![estimations after bias fix](https://github.com/omri24/EU-USD/assets/115406253/2591ee57-a3ca-4a68-954e-b6b87f698193)



--- MORE INFORMATION ---

To reduce the bias of the estimation, use the bias reduction algorithm, although the improvement is not significant.
inorder to use the bias reduction algorithm, one must run the data exportion option of this algorithm (last dialogue). 

Methods used to reduce overfitting:

1. The saved weights are the "first run" of the code after setting it completly, without "choosing" convenient parameters that work well specifically for the testing set. They are saved only to reduce the running time.
2. The data in the testing file doesn't exist in the training file at all.
