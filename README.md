# EU-USD
--- GENERAL INFORMATION ABOUT THE CSV FILES---

training set = EU price, in every minute, since 2001, unill may 2023.
testing set = EU price, in every minute in june and july 2023.

l1, l2,...,l6 = the weights of the NN- layers

--- GENERAL INFORMATION ABOUT THE ALGORITHM ---

This algorihm separates the datasets into groups of 72 samples (which is about 1/20 of a day). Then it estimates the average of the next group, from all the data samples of the current group, using a NN.

--- INFORMATION ABOUT THE IMAGES OF THE RESULTS, PLEASE READ THIS PART ---

The "raw estimations" image shows the estimated values (red) in compare to the real values (blue). 
It is clear that there is a big bias (bias here is the difference between the averages of the predictions and the real values).

The "estimations after bias fix" image shows the predictions and the real values, after adding the bias size to each predictions (this action can be depicted by "moving" the entire predictions data such that now the average of predictions is the average of the real data, but the shapes that the datapoints create are unchanged). 
It is possible to see that the shape of the graphs is similar.

--- MORE INFORMATION ---

To reduce the bias of the estimation, use the bias reduction algorithm, although the improvement is not significant.
inorder to use the bias reduction algorithm, one must run the data exportion option of this algorithm (last dialogue). 

Methods used to reduce overfitting:

1. The saved weights are the "first run" of the code after setting it completly, without "choosing" convenient parameters that work well specifically for the testing set. They are saved only to reduce the running time.
2. The data in the testing file doesn't exist in the training file at all.
