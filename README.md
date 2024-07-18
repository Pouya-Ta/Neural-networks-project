# Neural-networks-project
### Project Overview

This project focuses on building and training neural networks using the PyTorch library. The primary tasks involve working with different types of neural networks to solve classification problems on well-known datasets and implementing custom neural network components.

#### Tasks and Goals

1. **Fully Connected Neural Network with MNIST**:
   - **Objective**: Train a fully connected neural network to classify handwritten digits from the MNIST dataset.
   - **Goal**: Achieve an accuracy of over 97%.
   - **Implementation**: The training process is documented in the `mnist-linear.ipynb` file.

2. **Convolutional Neural Network with MNIST**:
   - **Objective**: Train a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset.
   - **Goal**: Achieve an accuracy of over 97%.
   - **Implementation**: The training process is documented in the `mnist-cnn.ipynb` file.

3. **Brain Tumor Detection**:
   - **Objective**: Use transfer learning with a pre-trained ResNet model to classify brain tumor images.
   - **Goal**: Achieve an accuracy of over 73%.
   - **Implementation**: The process is documented in the `Tumor_Project.ipynb` file, using transfer learning techniques.

#### Key Components

1. **PyTorch Library**:
   - PyTorch is used for building and training the neural networks. It is selected for its popularity and ease of use in the deep learning community.

2. **Datasets**:
   - **MNIST Dataset**: Used for digit classification tasks.
   - **Brain Tumor Dataset**: Contains labeled images of brain tumors, used for the transfer learning task.

3. **Model Training and Evaluation**:
   - The project includes various Jupyter notebooks (`.ipynb` files) which document the step-by-step process of training and evaluating the models.
   - Tasks involve setting up the data pipeline, defining the network architecture, training the model, and evaluating its performance.

4. **Custom Components**:
   - Implementation of different types of neural networks (fully connected, convolutional).
   - Use of advanced techniques like transfer learning to leverage pre-trained models for new tasks.

### Instructions for Running the Project

1. **Setup**:
   - Ensure you have Python and PyTorch installed.
   - Additional dependencies are listed in the respective Jupyter notebooks.

2. **Execution**:
   - Clone the repository and navigate to the project directory.
   - Open the Jupyter notebooks and follow the instructions provided to run the training and evaluation scripts.

3. **Evaluation**:
   - Each notebook includes cells for evaluating the trained models. Follow these cells to obtain accuracy metrics and visualize the results.

This project is a comprehensive demonstration of using neural networks for image classification tasks, leveraging the power of PyTorch and transfer learning techniques.
