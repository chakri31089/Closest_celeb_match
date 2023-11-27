# Closest_celeb_match
trained based on Siamese networks with stored data on the latest bollywood actors.
Achieved accuracy of 97% and works even in low light conditions.
Real time face encoding matching with celeb dataset.

## Facerec.py
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

The code manage face detection, encoding, and recognition using pre-trained models and stored face encodings. The Facerec class encapsulates the functionalities related to face recognition.


