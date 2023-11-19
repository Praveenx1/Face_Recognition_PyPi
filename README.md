# Face_Recognition_PyPi
 Recognize and manipulate faces from Python or from the command line with the world’s simplest face recognition library.


Face Recognition

Welcome to the Face Recognition project, a Python library that allows you to recognize and manipulate faces easily. This project leverages the power of dlib's state-of-the-art face recognition built with deep learning. The model boasts an impressive accuracy of 99.38% on the Labeled Faces in the Wild benchmark.
Features. Recognize and manipulate faces from Python or via the command line. Simple and easy-to-use face recognition library. Utilizes dlib's powerful face recognition capabilities.Achieves a remarkable accuracy of 99.38% on the Labeled Faces in the Wild benchmark.

Installation
INSTALL THE requirements.txt file


    Python 3.3+ or Python 2.7
    macOS or Linux (Windows not officially supported, but might work)

Installing on macOS or Linux

Before installing this library, make sure you have dlib already installed with Python bindings. You can follow the instructions on how to install dlib from source on macOS or Ubuntu.

Once dlib is installed, you can install the face_recognition module from PyPI using pip3 (or pip2 for Python 2):

bash

    pip3 install face_recognition

Installing on Windows

Although Windows is not officially supported, helpful users have posted instructions on how to install this library on Windows. Check out @masoudr’s Windows 10 installation guide for a step-by-step guide on installing dlib and face_recognition.
Usage

This project provides a simple face_recognition command line tool that allows you to perform face recognition on a folder of images from the command line. Explore the various options and parameters to customize your face recognition experience. 
Run
   python face.py
This will give the face recognised image of the image provided 
Run 
  python face_video.py
THis will recognise image from webcame or USBcam
This project was taken from https://pypi.org/project/face-recognition/ 

License

This project is licensed under the MIT License.

Feel free to contribute, open issues, or submit pull requests. Happy face recognizing! 😊
