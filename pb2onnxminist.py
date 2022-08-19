import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n1 = 1
        self.n2 = 10
        self.n3 = 20
        self.conv1 = nn.Conv2d(self.n1, self.n2, kernel_size=5)
        self.conv2 = nn.Conv2d(self.n2, self.n3, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, y, z):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        result = [self.func(x), self.funcy(y), self.funcz(z)]
        return result
        # if y and not z:
        #     return [F.log_softmax(x, dim=1), self.generate(y), None]
        # if not y and z:
        #     return [F.log_softmax(x, dim=1), None, self.decode(z)]
        # if y and z:
        #     return [F.log_softmax(x, dim=1), self.generate(y), self.decode(z)]
        # return [F.log_softmax(x, dim=1), None, None]
        # return [F.log_softmax(x, dim=1), self.generate(y), self.decode(z)]
    
    def func(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def funcy(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    def funcz(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    def generate(self, x):
        return 'generate function！'
    
    def decode(self, x):
        return 'decode function!'


if __name__ == "__main__":
    from torch.autograd import Variable

    # Load the trained model from file
    trained_model = Net()
    trained_model.load_state_dict(torch.load('output/mnist.pth'))

    # Export the trained model to ONNX
    dummy_input1 = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
    dummy_input2 = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
    dummy_input3 = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model

    example_output = trained_model(dummy_input1, dummy_input2, dummy_input3)
    print(example_output)

    torch.onnx.export(trained_model, 
                    (dummy_input1, dummy_input2, dummy_input3),
                    "output/mnist.onnx", 
                    input_names=["x", "y", "z"],
                    output_names=["output1", "output2", "output3"])



