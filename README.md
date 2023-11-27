# Closest_celeb_match
trained based on Siamese networks with stored data on the latest bollywood actors.
Achieved accuracy of 97% and works even in low light conditions.
Real time face encoding matching with celeb dataset.

## 1.Facerec.py
face recognition system using PyTorch and the facenet_pytorch library. Here's an explanation of the code:

### Libraries Imported
facenet_pytorch: Library for face detection and recognition.
torch, torchvision: PyTorch libraries for machine learning.
cv2: OpenCV library for computer vision.
os, glob, numpy: Standard Python libraries for file operations and array manipulation.
PIL.Image: Library for image handling.

### Loading Pre-trained Models and Data
data=torch.load('data.pt',map_location=torch.device('cpu')): Loads previously saved face encodings and associated names.
faces, names = data[0], data[1]: Extracts face encodings and names from the loaded data.
mtcnn = MTCNN(): Initializes an MTCNN (Multi-Task Cascaded Convolutional Networks) for face detection.
res = InceptionResnetV1(pretrained='vggface2').eval(): Initializes a pre-trained InceptionResnetV1 model for face recognition using the VGGFace2 dataset.
res = torch.load("flayer_model.pt", map_location=torch.device('cpu')): Loads a previously trained face recognition model.

## Facerec Class
### Initialization:
known_face_encodings, known_face_names: Lists storing known face encodings and their associated names.
frame_resizing: Scale factor for resizing frames.

### load_encoding_images Method:
Loads images from the specified path and processes each image for face detection and encoding.
Uses MTCNN to detect faces in the image and InceptionResnetV1 to encode them.
Compares the encodings with existing known face encodings. If a new face is detected, it adds its encoding and associated name to the list of known faces.

### detect_known_faces Method:
Accepts a frame (image) as input and detects faces using MTCNN.
Encodes the detected face using the pre-trained face recognition model (InceptionResnetV1).
Compares the encoding of the detected face with known face encodings to recognize and identify the person.
Returns face locations, recognized face names, and their respective confidence scores.

The code seems to manage face detection, encoding, and recognition using pre-trained models and stored face encodings. The Facerec class encapsulates the functionalities related to face recognition.


## 2.Match_celeb.ipynb
implementation of face recognition in a video using the Facerec class previously defined. 

### Libraries Imported:
cv2, os, glob, numpy, time, torch: Libraries for various operations like image processing, file handling, array manipulation, timing, and PyTorch functionalities.
Facerec: Custom class, presumably containing face recognition methods.
MTCNN, InceptionResnetV1 from facenet_pytorch: Modules for face detection and recognition.

### Initialization:
fr = Facerec(): Instantiates an object of the Facerec class for face recognition operations.
path = "data": Defines a path where data (presumably images and encodings) is stored.
vid = cv2.VideoCapture("C:\\Users\\nchakri\\Downloads\\test1.mp4"): Opens a video file for processing.

### Face Recognition Loop:
mtcnn = MTCNN(): Initializes an MTCNN object for face detection.
flag = -1: Initializes a flag used for timing purposes.
t = time.time(): Records the start time for frame processing.

### Frame Processing Loop:
Enters a continuous loop for processing frames from the video.
_, img = vid.read(): Reads a frame from the video.
boxs = mtcnn.detect(img): Detects faces in the frame using MTCNN.
Checks if it's been more than 2 seconds (time.time() - t > 2) and flag is -1. If so, tries to detect known faces in the frame (fr.detect_known_faces(img)).

### Face Recognition and Visualization
If faces are detected (type(boxs) != type(None)), the script tries to recognize known faces.
Draws bounding boxes around detected faces and embeds the recognized face images onto the original frame.
cv2.putText() adds text labels with the recognized face names.
Displays the modified frame in a window using cv2.imshow().
If the 'Esc' key is pressed (cv2.waitKey(1) == 27), the loop breaks, and the video stream is released and windows are closed.

The real-time face recognition on a video stream, recognizing known faces and displaying their names as labels on the video frames. It uses pre-trained models for face detection and recognition along with the custom Facerec class to manage face recognition operations.


## 3.Reset.ipynb
This code creates a nested list containing two empty lists. Then it uses PyTorch's torch.save() function to save this list into a file named "data.pt" in a serialized format.

### Breaking it down:
a = [[], []]: Initializes a list a containing two empty lists as its elements. It's a nested list structure with two empty lists inside the outer list.
torch.save(a, "data.pt"): Saves the list a into a file named "data.pt" using PyTorch's torch.save() function. This function serializes the object a (in this case, the nested list) and saves it into the specified file path ("data.pt").

After running this code, the file "data.pt" will contain the serialized representation of the nested list a. This serialization allows you to save Python objects, such as lists, tensors, or models, into files so that they can be loaded and used later without losing their structure or content.


## 4.Siamese.ipynb
This script appears to create a Siamese neural network for face recognition using a contrastive loss function and the facenet_pytorch library for face embeddings.

### Importing Libraries
The script imports necessary libraries including PyTorch, torchvision, OpenCV (cv2), numpy, PIL, and modules/classes from facenet_pytorch.

### Preparing Pre-trained Models
Initializes MTCNN and loads a pre-trained InceptionResnetV1 model (model=InceptionResnetV1(pretrained='vggface2').eval()).
Sets the requires_grad property for specific layers in the model to enable gradient computation (for i in model.last_linear.parameters(): i.requires_grad = True, etc.).

### Dataset Preparation
Defines a SiameseNetworkDataset class inheriting from Dataset, used to prepare image pairs for the Siamese network.
Defines transformations (transform) including resizing and converting images to tensors.

### Loading Image Data
Uses torchvision.datasets.ImageFolder to load images from a specified directory.
Creates a SiameseNetworkDataset object (sdata) using the loaded image data.

### Data Loader
Uses DataLoader to create a data loader (data_loader) for the prepared SiameseNetworkDataset.
Iterates through the data loader and visualizes sample image pairs using matplotlib.

### Contrastive Loss and Optimization
Defines a ContrastiveLoss class inheriting from torch.nn.Module to compute the contrastive loss between pairs of embeddings.
Sets up an Adam optimizer and initializes variables for storing loss history (counter, loss_history).

### Training Loop
Runs a training loop over 10 epochs.
Inside the loop, iterates through the data loader batches and performs forward pass, loss computation, and backward propagation.
Updates the network's parameters using the optimizer and records loss values.

### Model Saving
Finally, saves the trained Siamese network (net) as "flayer_model.pt" using torch.save().

This code trains a Siamese network using face image pairs, optimizing it to minimize a contrastive loss function that measures the similarity between embeddings of pairs of images. The trained model is then saved for later use in face recognition or verification tasks.


## 5.encoding.ipynb
This snippet of code is intended to perform some facial recognition tasks using the Facerec class from a file/module named Facerec.py. Let me break down what each part of this code is likely doing:

### Importing Libraries
cv2, os, glob, numpy, time: Libraries for various operations like image processing, file handling, array manipulation, and timing.

### Loading Paths
path = 'data': Sets the variable path to the directory named 'data'.
dir_paths = os.listdir(path): Retrieves a list of all files and directories in the 'data' directory.

### Initializing Facerec Class and Encoding Images
fr = Facerec(): Initializes an instance of the Facerec class, presumably containing methods related to facial recognition tasks.
fr.load_encoding_images(path): Invokes the load_encoding_images() method from the Facerec class to load and encode images located in the 'data' directory.

This code snippet, in essence, intends to utilize the Facerec class to load images from a specific directory ('data') and perform facial encoding or recognition tasks. The load_encoding_images() method within the Facerec class is likely responsible for loading images, detecting faces, and encoding them for later recognition purposes.
