import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
"""
Params that worked well: mode = 1, n_samples = 72, time_bound = 80000, modulu_param = 140, learning_rate = 0.0001,
learning_rate_decay_parameter = 1, optimizer = Adam, loss_breaker = 0.00001, min_epoch_to_break = 10,
hidden_size1 = 200   # the size of hidden layer 1
hidden_size2 = 500   # the size of hidden layer 2
hidden_size3 = 700   # the size of hidden layer 3
hidden_size4 = 500   # the size of hidden layer 4
hidden_size5 = 200   # the size of hidden layer 5
"""

# This line prevents randomness in the results
torch.manual_seed(0)

# Hyper- parameters
n_samples = 72  # number of time samples used for each prediction. 1440 time samples == 1 day
future_n = 100  # future time point to predict. 480 time samples == 8 hours [irrelevant for average mode]
time_bound = 80000  # ignore number of samples in each epoch. if n_samples = 1440, then 365 here means ignore 1st year
modulu_param = 140  # use only half, third, etc ot the samples in training phase
# if n_samples = 720, modulu_param = 14 -> 1 week

input_size = n_samples   # input size for the FNN.
hidden_size1 = 200   # the size of hidden layer 1
hidden_size2 = 500   # the size of hidden layer 2
hidden_size3 = 700   # the size of hidden layer 3
hidden_size4 = 500   # the size of hidden layer 4
hidden_size5 = 200   # the size of hidden layer 5
num_epochs = 30  # the number of 'loops' on the entire dataset (in each loop, we discard time_bound items from the data)
output = 1  # of the FNN
learning_rate = 0.0001
learning_rate_decay_parameter = 1  # reduces the leaning rate in exp way, look below. set to 1 to cancel this effect
loss_breaker = 0.00001   # if the loss reaches this value, training breaks immediately
min_epoch_to_break = 10   # the minimal epoch number in which the loss_breaker can apply

low_bound_for_bias = 6000  # irrelevant in this algorithm
high_bound_for_bias = 7000  # irrelevant in this algorithm

batch_size_train = 1   # irrelevant in this algorithm
batch_size_test = 1   # irrelevant in this algorithm

# This part chooses GPU if possible and CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activation_mode = input("Select activation mode: 1 = average estimation,  2 = point estimation: ")
if (activation_mode != "1") and (activation_mode != "2"):
    raise ValueError("activation mode is different from 1 and from 2")

class EuroDataset(Dataset):

    def __init__(self, file_name):
        csv = np.loadtxt(file_name, delimiter=",", dtype=np.float32, skiprows=1)  # use chopped csv version
        price_points = csv[:,[2]]
        points_to_remove = price_points.shape[0] % n_samples  # removing them to allow reshape to the raw data
        d = int(price_points.shape[0] / n_samples)
        x_w_extra_col = np.reshape(price_points[points_to_remove:, :], (n_samples, d))
        if activation_mode == "2":
            y_w_extra_val = x_w_extra_col[[future_n], :]
        if activation_mode == "1":
            y_w_extra_val = np.mean(x_w_extra_col, axis=0, dtype=np.float32, keepdims=True)
        self.x = torch.from_numpy(x_w_extra_col[:, :-1])
        self.y = torch.from_numpy(y_w_extra_val[:, 1:])
        self.n = x_w_extra_col.shape[1]
        self.m = y_w_extra_val.shape[1]

    def __len__(self):
        if (self.m != self.n):
            raise IndexError("There are " + str(self.n) + " inputs (x for NN) but " + str(self.m) + " outputs (y)")
        else:
            return self.n

    def __getitem__(self, index):
        return (self.x[:, index], self.y[:, index])

# The Datasets for this algorithm:
train_dataset = EuroDataset(file_name="training_set.csv")
test_dataset = EuroDataset(file_name="testing_set.csv")

to_check_errors = input("Would you like to check if there are errors in the datasets? (1 + enter = yes): ")
if to_check_errors == "1":
    if activation_mode == "2":
        # Error is defined by: dataset[i][1] != dataset[i+1][0][future_n],
        # which means, the prediction isn't in the correct place of the next item in the dataset
        counter = 0
        for i in range (len(train_dataset) - 2):
            if train_dataset[i + 1][0][future_n] != train_dataset[i][1]:
                counter += 1
        print("errors for training_dataset = " + str(counter))
        counter = 0
        for i in range (len(test_dataset) - 2):
            if test_dataset[i + 1][0][future_n] != test_dataset[i][1]:
                counter += 1
        print("errors for testing_dataset = " + str(counter))

    if activation_mode == "1":
        # Error is defined by: abs(dataset[i][1] - mean(dataset[i+1][0])) > dataset[i][1] * 0.0001
        # which means, the prediction isn't in the average of the next item in the dataset
        counter = 0
        for i in range(len(train_dataset) - 2):
            if abs(train_dataset[i][1].item() - torch.mean(train_dataset[i + 1][0], dtype=torch.float32).item()) > \
                    train_dataset[i][1].item() * 0.00001:
                counter += 1
        print("errors for training_dataset = " + str(counter))
        counter = 0
        for i in range(len(test_dataset) - 2):
            if abs(test_dataset[i][1].item() - torch.mean(test_dataset[i + 1][0], dtype=torch.float32).item()) > \
                    test_dataset[i][1].item() * 0.00001:
                counter += 1
        print("errors for testing_dataset = " + str(counter))

# FNN structure:
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.l3 = nn.Linear(hidden_size2, hidden_size3)
        self.tanh = nn.Tanh()
        self.l4 = nn.Linear(hidden_size3, hidden_size4)
        self.exp = nn.ELU()
        self.l5 = nn.Linear(hidden_size4, hidden_size5)
        self.l6 = nn.Linear(hidden_size5, output)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.exp(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.exp(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.l6(out)
        return out


model = NeuralNet(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2,
                  hidden_size3=hidden_size3, hidden_size4=hidden_size4, hidden_size5=hidden_size5,
                  output=output).to(device)

# Loss (cost function) and optimizer
criterion = nn.L1Loss()  # For this algorithm, L1 norm is the most reasonable loss func
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # try using different optimizers


# Train the model
to_train = input("Would you like to train the NN? (1 + enter == yes, enter == no): ")
if to_train == "1":
    optimizer_call = 0
    for epoch in range(num_epochs):
        if (epoch > min_epoch_to_break) and (loss <= loss_breaker):
            break
        for i, (present_data, future_data) in enumerate(train_dataset):
            if (i > time_bound) and (i % modulu_param == 0):
                present_data = present_data.to(device)
                future_data = future_data.to(device)

                # Forward pass
                model_out = model(present_data)
                loss = criterion(model_out, future_data)
                if loss <= loss_breaker:
                    break

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_call = optimizer_call + 1
                learning_rate = learning_rate * (learning_rate_decay_parameter ** optimizer_call)

                if i % 100 == 0:
                    print("Epoch = " + str(epoch + 1) + ", step = " + str(i + 1) + ", loss = " + str(loss.item()))

    predictions_lst = []
    values_tried_to_predict = []
    for i, (present_data, future_data) in enumerate(train_dataset):
        if (i > low_bound_for_bias) and (i < high_bound_for_bias):
            present_data = present_data.to(device)
            future_data = future_data.to(device)
            model_out = model(present_data)
            predictions_lst = predictions_lst + [model_out.item()]
            values_tried_to_predict = values_tried_to_predict + [future_data.item()]

    bias = (sum(values_tried_to_predict) / len(values_tried_to_predict)) - (sum(predictions_lst) / len(predictions_lst))
    to_save_params = input("Would you like to save the calculated NN parameters? (1 + enter == yes, enter == no): ")
    if to_save_params == "1":
        l1_np = model.l1.weight.detach().cpu().numpy()
        l2_np = model.l2.weight.detach().cpu().numpy()
        l3_np = model.l3.weight.detach().cpu().numpy()
        l4_np = model.l4.weight.detach().cpu().numpy()
        l5_np = model.l5.weight.detach().cpu().numpy()
        l6_np = model.l6.weight.detach().cpu().numpy()
        np.savetxt("l1.csv", l1_np, delimiter=",")
        np.savetxt("l2.csv", l2_np, delimiter=",")
        np.savetxt("l3.csv", l3_np, delimiter=",")
        np.savetxt("l4.csv", l4_np, delimiter=",")
        np.savetxt("l5.csv", l5_np, delimiter=",")
        np.savetxt("l6.csv", l6_np, delimiter=",")
    print("Bias from Training = " + str(bias))
    print("Training phase- End\n")

# Test the model
x_for_scatter = []
y_for_scatter = []
real_values = []
if to_train == "1":
    print("Training phase was calculated- would you like to test the NN with the new weights?")
    how_to_test = input("Test the new weights- 1 + enter. Test weights from csv files- 2 + enter. Skip test- enter: ")
    if how_to_test == "2":
        l1_np = np.loadtxt("l1.csv", dtype=np.float32, delimiter=",")
        l2_np = np.loadtxt("l2.csv", dtype=np.float32, delimiter=",")
        l3_np = np.loadtxt("l3.csv", dtype=np.float32, delimiter=",")
        l4_np = np.loadtxt("l4.csv", dtype=np.float32, delimiter=",")
        l5_np = np.loadtxt("l5.csv", dtype=np.float32, delimiter=",")
        l6_np = np.loadtxt("l6.csv", dtype=np.float32, delimiter=",", ndmin=2)
        model.l1.weight = nn.parameter.Parameter(torch.from_numpy(l1_np))
        model.l2.weight = nn.parameter.Parameter(torch.from_numpy(l2_np))
        model.l3.weight = nn.parameter.Parameter(torch.from_numpy(l3_np))
        model.l4.weight = nn.parameter.Parameter(torch.from_numpy(l4_np))
        model.l5.weight = nn.parameter.Parameter(torch.from_numpy(l5_np))
        model.l6.weight = nn.parameter.Parameter(torch.from_numpy(l6_np))
    if (how_to_test == "1") or (how_to_test == "2"):
        with torch.no_grad():
            total_error = 0
            n_predictions = 0
            for i, (present_data, future_data) in enumerate(test_dataset):
                present_data = present_data.to(device)
                future_data = future_data.to(device)
                model_out = model(present_data)
                n_predictions += 1
                total_error += abs((model_out - future_data).item()) / future_data.item()  # relative error
                if activation_mode == "2":
                    x_for_scatter = x_for_scatter + [(i + 1) * n_samples + future_n]
                if activation_mode == "1":
                    x_for_scatter = x_for_scatter + [i + 1]
                y_for_scatter = y_for_scatter + [model_out.item()]
                real_values = real_values + [future_data.item()]
            mean_error = total_error / n_predictions
            print("NN output:")
            print("amount of predictions = " + str(n_predictions) + ", total relative error = " + str(total_error) +
                  ", mean relative error = " + str(mean_error * 100.0) + "%")

    all_test_data = np.loadtxt("testing_set.csv", delimiter=",", dtype=np.float32)
    real_data_to_plot = all_test_data[:, 2]
    average_euro_price = np.average(real_data_to_plot)
    predictions_average = sum(y_for_scatter) / len(y_for_scatter)
    real_values_lst_average = sum(real_values) / len(real_values)
    print("\nAverage euro price = " + str(average_euro_price))
    print("Average of the predictions = " + str(predictions_average))
    print("Average of the values that the model tries to predict = " + str(real_values_lst_average))
    print("Bias of the model in the testing phase = " + str(abs(real_values_lst_average - predictions_average)))
    if activation_mode == "1":
        plt.scatter(x=x_for_scatter, y=y_for_scatter, c="r", label="Predictions")
        plt.scatter(x=x_for_scatter, y=real_values, c="b", label="Real data")
        plt.legend()
        plt.show()
        plt.scatter(x=x_for_scatter, y=y_for_scatter, c="r", label="Predictions")
        plt.legend()
        plt.show()
        plt.scatter(x=x_for_scatter, y=real_values, c="b", label="Real data")
        plt.legend()
        plt.show()
        plt.scatter(x=x_for_scatter, y=real_values, c="b", label="Real data")
        plt.scatter(x=x_for_scatter, y=[i + real_values_lst_average - predictions_average for i in y_for_scatter],
                    c="r", label="Predictions- normalized to real values")
        plt.legend()
        plt.show()
    if activation_mode == "2":
        scaling_factor = 1
        # scaling_factor = average_euro_price / predictions_average   # to "get over" the bias when plotting
        # set scaling_factor to 1 for plotting the NN output without any changes
        plt.scatter(x=x_for_scatter, y=[i * scaling_factor for i in y_for_scatter], c="r")
        plt.plot(real_data_to_plot)
        plt.show()
        plt.scatter(x=x_for_scatter, y=[i * scaling_factor for i in y_for_scatter], c="r")
        plt.show()

if to_train != "1":
    q = "Training phase skipped- testing phase will use data from csv files. test- 1 + enter. skip test- enter: "
    to_test = input(q)
    if to_test == "1":
        l1_np = np.loadtxt("l1.csv", dtype=np.float32, delimiter=",")
        l2_np = np.loadtxt("l2.csv", dtype=np.float32, delimiter=",")
        l3_np = np.loadtxt("l3.csv", dtype=np.float32, delimiter=",")
        l4_np = np.loadtxt("l4.csv", dtype=np.float32, delimiter=",")
        l5_np = np.loadtxt("l5.csv", dtype=np.float32, delimiter=",")
        l6_np = np.loadtxt("l6.csv", dtype=np.float32, delimiter=",", ndmin=2)
        model.l1.weight = nn.parameter.Parameter(torch.from_numpy(l1_np))
        model.l2.weight = nn.parameter.Parameter(torch.from_numpy(l2_np))
        model.l3.weight = nn.parameter.Parameter(torch.from_numpy(l3_np))
        model.l4.weight = nn.parameter.Parameter(torch.from_numpy(l4_np))
        model.l5.weight = nn.parameter.Parameter(torch.from_numpy(l5_np))
        model.l6.weight = nn.parameter.Parameter(torch.from_numpy(l6_np))
        with torch.no_grad():
            total_error = 0
            n_predictions = 0
            for i, (present_data, future_data) in enumerate(test_dataset):
                present_data = present_data.to(device)
                future_data = future_data.to(device)
                model_out = model(present_data)
                n_predictions += 1
                total_error += abs((model_out - future_data).item()) / future_data.item()  # relative error
                if activation_mode == "2":
                    x_for_scatter = x_for_scatter + [(i + 1) * n_samples + future_n]
                if activation_mode == "1":
                    x_for_scatter = x_for_scatter + [i + 1]
                y_for_scatter = y_for_scatter + [model_out.item()]
                real_values = real_values + [future_data.item()]
            mean_error = total_error / n_predictions
            print("NN output:")
            print("amount of predictions = " + str(n_predictions) + ", total relative error = " + str(total_error) +
                  ", mean relative error = " + str(mean_error * 100.0) + "%")

    all_test_data = np.loadtxt("testing_set.csv", delimiter=",", dtype=np.float32)
    real_data_to_plot = all_test_data[:, 2]
    average_euro_price = np.average(real_data_to_plot)
    predictions_average = sum(y_for_scatter) / len(y_for_scatter)
    real_values_lst_average = sum(real_values) / len(real_values)
    print("\nAverage euro price = " + str(average_euro_price))
    print("Average of the predictions = " + str(predictions_average))
    print("Average of the values that the model tries to predict = " + str(real_values_lst_average))
    print("Bias of the model in the testing phase = " + str(abs(real_values_lst_average - predictions_average)))
    if activation_mode == "1":
        plt.scatter(x=x_for_scatter, y=y_for_scatter, c="r", label="Predictions")
        plt.scatter(x=x_for_scatter, y=real_values, c="b", label="Real data")
        plt.legend()
        plt.show()
        plt.scatter(x=x_for_scatter, y=y_for_scatter, c="r", label="Predictions")
        plt.legend()
        plt.show()
        plt.scatter(x=x_for_scatter, y=real_values, c="b", label="Real data")
        plt.legend()
        plt.show()
        plt.scatter(x=x_for_scatter, y=real_values, c="b", label="Real data")
        plt.scatter(x=x_for_scatter, y=[i + real_values_lst_average - predictions_average for i in y_for_scatter],
                    c="r", label="Predictions- normalized to real values")
        plt.legend()
        plt.show()
    if activation_mode == "2":
        scaling_factor = 1
        # scaling_factor = average_euro_price / predictions_average   # to "get over" the bias when plotting
        # set scaling_factor to 1 for plotting the NN output without any changes
        plt.scatter(x=x_for_scatter, y=[i * scaling_factor for i in y_for_scatter], c="r")
        plt.plot(real_data_to_plot)
        plt.show()
        plt.scatter(x=x_for_scatter, y=[i * scaling_factor for i in y_for_scatter], c="r")
        plt.show()

run_data_exporter = input("\nWould you like to export the data? (1 + enter == yes, enter == no): ")
if run_data_exporter == "1":  # this section exports the estimations and real values, for the bias fixing algorithm
    train_arr = np.zeros(shape=(int(len(train_dataset)) - 1, 2), dtype=np.float32)
    test_arr = np.zeros(shape=(int(len(test_dataset)) - 1, 2), dtype=np.float32)
    for i, (present_data, future_data) in enumerate(train_dataset):
        model_out = model(present_data)
        train_arr[i, 0] = model_out
        train_arr[i, 1] = future_data
    for i, (present_data, future_data) in enumerate(test_dataset):
        model_out = model(present_data)
        test_arr[i, 0] = model_out
        test_arr[i, 1] = future_data
    np.savetxt("exported_train.csv", train_arr, delimiter=",")
    np.savetxt("exported_test.csv", test_arr, delimiter=",")
