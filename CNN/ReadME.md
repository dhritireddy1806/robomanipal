The steps are:
-Load the MNIST Dataset
-Normalize the Image Pixel Values
-Initialize Convolution Filters
-Perform Convolution Operation
-Apply ReLU Activation
-Perform Max Pooling
-Flatten the Feature Maps
-Initialize Fully Connected Layer Weights
-Forward Pass Through Hidden Layer
-Forward Pass Through Output Layer
-Apply Softmax Activation
-Convert Labels to One-Hot Encoding
-Compute Output Layer Error (Cross-Entropy Gradient)
-Backpropagate Through Output Layer
-Backpropagate Through Hidden Layer
-Compute Dense Layer Gradients
-Update Dense Layer Weights and Biases
-Backpropagate to Flatten Layer
-Reshape Gradient to Pooling Layer Shape
-Perform Max Pooling Backpropagation
-Perform ReLU Backpropagation
-Compute Convolution Filter Gradients
-Update Convolution Filters
-Repeat for All Training Samples (Epoch Training)
-Evaluate Training Accuracy
-Perform Forward Pass on Test Data
-Evaluate Test Accuracy

⁠What a convolution is?
A)A convolution is an operation where a small matrix called a kernel/filter slides over an image and computes a weighted sum at every location.

⁠How kernels extract features?
A)A kernel (filter) acts like a pattern detector. Different filters respond strongly to different image features
vertical edge filter :Produces large values where vertical edges exist.
horizontal edge filter:Detects horizontal edges.
During training, CNNs automatically learn useful filters instead of manually defining them.

⁠Why padding is used?
A) It is necessaru coz it reserves image size, retains edge information and allows deeper networks

⁠How stride affects output dimensions?
A)If the strides are large then:Smaller output, Faster computation, Less detailed information

⁠Why pooling is used?
A)Poolin reduces spatial dimentions. It reduces computation and overfitting. Even if a feature moves slightly , pooling captures it

⁠The role of fully connected layers
A)Fully connected layers perform classification and convolution layers act as feature extracters

⁠How CNNs learn spatial features from images?
A)A traditional neural network sees:784 numbers and loses the image structure. CNNs preserve spatial relationships.It has 3 sets of layers: Erly, middle and deep
Early layer: identifies edges, lines and corners
Middle layer:combines simple patterns to form curves, loops and shapes
Deep layers: recognizes complex structures like eyes, faces , objects etc

⁠How CNNs differ from traditional Neural Networks
A)Since in a traditional NN , each neuron is connected to every input so it cause certain problems
Huge number of parameters
Ignores image structure
More prone to overfitting

while CNN has a concept of weight sharing which means that each neuron goes thru all the points contatining a specific feature, basically the same filter is reused across the entire image. its benefits include:
Far fewer parameters
Faster training
Better at images
Preserves spatial information