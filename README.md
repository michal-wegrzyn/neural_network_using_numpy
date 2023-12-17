# Neural Network Implementation using NumPy

## Activation and loss functions

Some popular activation and loss functions are implemented. After `from NeuralNetwork import NeuralNetwork as NN` you can access them with `NN.AFunc.func_name` or `NN.LFunc.func_name` . Use `d_func_name` for derivative and `func_name_pair` for tuple `(func_name, d_func_name)` .

## Test and experiment with the neural network

In `/examples` directory there are examples of using the neural network on some popular datasets that are stored in `/datasets` .

```bash
git clone https://github.com/michal-wegrzyn/neural_network_using_numpy.git
cd neural_network_using_numpy
pip install -r requirements.txt
cd examples
python mnist.py
```

## Dataset augmentation

The dataset may be too small to effectively train the neural network. In `DatasetAugmentation.py` there are functions that can help you augment it.

Use `shift` function to create more data by shifting e.g. images of digits from the MNIST dataset by a few pixels in different directions. 

You can also use `flip` function e.g. for the Fashion MNIST dataset to expand it with symmetrical (flipped) images.