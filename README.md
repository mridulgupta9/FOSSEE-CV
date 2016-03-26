# FOSSEE-CV  
# task 1
faces_trained.xml- It contains trained xml files for faces - label=0 is for ellen degeneres' face else label=1;

facerec_fisherfaces.cpp- In line 55 address of csv file for faces data is given which then trains the model and saves it as faces_trained.xml

line 55:
string fn_csv = string("C:/Users/MRIDUL/Desktop/fossee/task1/faces.csv");
here change the address to your csv file and generate your own trained data.


task1.cpp- In line 18, address of a picture is given, all the faces are then detected and then ellen degeneres' face is recognised and her eyes are detected. Accuracy of this step depends on the amount of images given for the training of the model and hence can be improved by providing more images. We can also train an MLP for the same.

line 18:
img = imread("oscarSelfie.jpg", CV_LOAD_IMAGE_UNCHANGED);
here give the address of the image in your computer and then run the program.
it will look like:-
img = imread("c:/users/path/to/file", CV_LOAD_IMAGE_UNCHANGED);


All other details have been provided in the form of comments.
