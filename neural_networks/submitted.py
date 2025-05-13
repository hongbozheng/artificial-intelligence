# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import torch
import torch.nn as nn

device = torch.device("cuda")

"""
1.  Build a neural network class.
"""
class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        # image input size 31x31 channels = 3 RGB
        self.conv3x3 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.conv1x1_1_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, padding=0, stride=1)
        self.conv3x3_1 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv1x1_1_2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0, stride=1)
        self.conv1x1_sc_1 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0, stride=1)

        self.conv1x1_1_3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, stride=1)
        # self.conv3x3_1_ = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)
        # self.conv1x1_1_4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0, stride=1)
        # self.conv1x1_sc_1_ = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=0, stride=1)

        self.conv1x1_2_1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=2)
        self.conv3x3_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv1x1_2_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=1)
        self.conv1x1_sc_2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=2)

        self.conv1x1_2_3 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, stride=1)

        self.conv1x1_3_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, stride=2)
        self.conv3x3_3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.conv1x1_3_2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=1)
        self.conv1x1_sc_3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=0, stride=2)

        self.conv1x1_3_3 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0, stride=1)

        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(8,8))
        self.fc = torch.nn.Linear(in_features=128*8*8, out_features=5)
        # self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(4,4))
        # self.fc = torch.nn.Linear(in_features=256*4*4, out_features=5)
        # self.dropout = torch.nn.Dropout(p=0.5)
        # raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x = torch.reshape(input=x, shape=(-1,3,31,31))  # 3x31x31
        x = self.conv3x3(x)                             # downsize by 2,    32x16x16
        x = self.relu(x)
        # --- residual block 1 ---
        # resblk 1 x 1
        shortcut_1 = self.conv1x1_sc_1(x)
        x = self.conv1x1_1_1(x)                         # 32x16x16
        x = self.relu(x)
        x = self.conv3x3_1(x)                           # 32x16x16
        x = self.relu(x)
        x = self.conv1x1_1_2(x)                         # 64x16x16
        x += shortcut_1
        x = self.relu(x)
        # resblk 1 x 2
        shortcut_1 = x
        x = self.conv1x1_1_3(x)                         # 32x16x16
        x = self.relu(x)
        x = self.conv3x3_1(x)                           # 32x16x16
        x = self.relu(x)
        x = self.conv1x1_1_2(x)                         # 64x16x16
        x += shortcut_1
        x = self.relu(x)
        # resblk 1 x 3
        shortcut_1 = x
        x = self.conv1x1_1_3(x)                         # 32x16x16
        x = self.relu(x)
        x = self.conv3x3_1(x)                           # 32x16x16
        x = self.relu(x)
        x = self.conv1x1_1_2(x)                         # 64x16x16
        x += shortcut_1
        x = self.relu(x)
        # resblk 1 x 4
        shortcut_1 = x
        x = self.conv1x1_1_3(x)                         # 32x16x16
        x = self.relu(x)
        x = self.conv3x3_1(x)                           # 32x16x16
        x = self.relu(x)
        x = self.conv1x1_1_2(x)                         # 64x16x16
        x += shortcut_1
        x = self.relu(x)
        # resblk 1 x 5
        shortcut_1 = x
        x = self.conv1x1_1_3(x)                         # 32x16x16
        x = self.relu(x)
        x = self.conv3x3_1(x)                           # 32x16x16
        x = self.relu(x)
        x = self.conv1x1_1_2(x)                         # 64x16x16
        x += shortcut_1
        x = self.relu(x)
        # -----------------------
        # --- residual block 2 ---
        # resblk 2 x 1
        shortcut_2 = self.conv1x1_sc_2(x)               # downsize by 2, 128x8x8
        x = self.conv1x1_2_1(x)                         # 64x8x8
        x = self.relu(x)
        x = self.conv3x3_2(x)                           # 64x8x8
        x = self.relu(x)
        x = self.conv1x1_2_2(x)                         # 128x8x8
        x += shortcut_2
        x = self.relu(x)
        # resblk 2 x 2
        shortcut_2 = x
        x = self.conv1x1_2_3(x)                         # 64x8x8
        x = self.relu(x)
        x = self.conv3x3_2(x)                           # 64x8x8
        x = self.relu(x)
        x = self.conv1x1_2_2(x)                         # 128x8x8
        x += shortcut_2
        x = self.relu(x)
        # resblk 2 x 3
        shortcut_2 = x
        x = self.conv1x1_2_3(x)                         # 64x8x8
        x = self.relu(x)
        x = self.conv3x3_2(x)                           # 64x8x8
        x = self.relu(x)
        x = self.conv1x1_2_2(x)                         # 128x8x8
        x += shortcut_2
        x = self.relu(x)
        # resblk 2 x 4
        shortcut_2 = x
        x = self.conv1x1_2_3(x)                         # 64x8x8
        x = self.relu(x)
        x = self.conv3x3_2(x)                           # 64x8x8
        x = self.relu(x)
        x = self.conv1x1_2_2(x)                         # 128x8x8
        x += shortcut_2
        x = self.relu(x)
        # resblk 2 x 5
        shortcut_2 = x
        x = self.conv1x1_2_3(x)                         # 64x8x8
        x = self.relu(x)
        x = self.conv3x3_2(x)                           # 64x8x8
        x = self.relu(x)
        x = self.conv1x1_2_2(x)                         # 128x8x8
        x += shortcut_2
        x = self.relu(x)
        # -----------------------
        # # --- residual block 3 ---
        # # resblk 3 x 1
        # shortcut_3 = self.conv1x1_sc_3(x)               # downsize by 2, 256x4x4
        # x = self.conv1x1_3_1(x)                         # 128x4x4
        # x = self.relu(x)
        # x = self.conv3x3_3(x)                           # 128x4x4
        # x = self.relu(x)
        # x = self.conv1x1_3_2(x)                         # 256x4x4
        # x += shortcut_3
        # x = self.relu(x)
        # # resblk 3 x 2
        # shortcut_3 = x  # 256x4x4
        # x = self.conv1x1_3_3(x)  # 128x4x4
        # x = self.relu(x)
        # x = self.conv3x3_3(x)  # 128x4x4
        # x = self.relu(x)
        # x = self.conv1x1_3_2(x)  # 256x4x4
        # x += shortcut_3
        # x = self.relu(x)
        # # resblk 3 x 3
        # shortcut_3 = x  # 256x4x4
        # x = self.conv1x1_3_3(x)  # 128x4x4
        # x = self.relu(x)
        # x = self.conv3x3_3(x)  # 128x4x4
        # x = self.relu(x)
        # x = self.conv1x1_3_2(x)  # 256x4x4
        # x += shortcut_3
        # x = self.relu(x)
        # # resblk 3 x 4
        # shortcut_3 = x  # 256x4x4
        # x = self.conv1x1_3_3(x)  # 128x4x4
        # x = self.relu(x)
        # x = self.conv3x3_3(x)  # 128x4x4
        # x = self.relu(x)
        # x = self.conv1x1_3_2(x)  # 256x4x4
        # x += shortcut_3
        # x = self.relu(x)
        # # resblk 3 x 5
        # shortcut_3 = x  # 256x4x4
        # x = self.conv1x1_3_3(x)  # 128x4x4
        # x = self.relu(x)
        # x = self.conv3x3_3(x)  # 128x4x4
        # x = self.relu(x)
        # x = self.conv1x1_3_2(x)  # 256x4x4
        # x += shortcut_3
        # x = self.relu(x)
        # # -----------------------
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        y = self.fc(x)

        return y
        raise NotImplementedError("You need to write this part!")
        ################## Your Code Ends here ##################


"""
2. Train your model.
"""
def fit(train_dataloader, test_dataloader, epochs):
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
        loss_fn:            your selected loss function
        optimizer:          your selected optimizer
    """
    
    # Create an instance of NeuralNet, don't modify this line.
    model = NeuralNet()


    ################# Your Code Starts Here #################
    """
    2.1 Create a loss function and an optimizer.

    Please select an appropriate loss function from PyTorch torch.nn module.
    Please select an appropriate optimizer from PyTorch torch.optim module.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    # make sure you use weight decay !
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
    # raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################


    """
    2.2 Train loop
    """
    for epoch in range(epochs):
        print("Epoch #", epoch)
        train(train_dataloader, model, loss_fn, optimizer)  # You need to write this function.
        test(test_dataloader, model, loss_fn)  # optional, to monitor the training progress
    return model, loss_fn, optimizer


"""
3. Backward propagation and gradient descent.
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################
    model = model.to(device)

    for (x, labels) in train_dataloader:
        x = x.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            y = model.forward(x)
            # print("[INFO]: Output shape ", y.size())
            (_, pred) = torch.max(y, 1)
            loss = loss_fn(y, labels)
            loss.backward()
            optimizer.step()
    # raise NotImplementedError("You need to write this part!")
    ################## Your Code Ends here ##################


def test(test_dataloader, model, loss_fn):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions
        loss_fn:            loss function
    """

    # test_loss = something
    # print("Test loss:", test_loss)
