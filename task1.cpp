#include<bits/stdc++.h>
#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
using namespace cv;
using namespace std;
Mat img; Mat templ; Mat result;
const char* image_window = "Source Image";
const char* result_window = "Result window";


int main()
{


    img = imread("oscarSelfie.jpg", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "oscarSelfie.jpg" and store it in 'img'
    
    Mat orig=img.clone();


    CascadeClassifier face_cascade;
    face_cascade.load( "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml" );    // put the address of the file as in your system
    CascadeClassifier eye_cascade;
    eye_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_eye.xml");          // put the address of the file as in your system
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( img, faces, 1.01, 2, 0|CV_HAAR_SCALE_IMAGE, Size(15, 15) );


    for( int i = 0; i < faces.size(); i++ )
    {
        faces[i].x*=0.95;faces[i].y*=0.95;faces[i].width*=1.2;faces[i].height*=1.2;  //to get complete face detected region is increased and then sent to predictor function
        //draws rectangle around faces
        rectangle( img, Point(faces[i].x,faces[i].y), Point( faces[i].x + faces[i].width , faces[i].y + faces[i].height ), Scalar(0,0,255), 4, 8, 0 );
        Mat face = orig(faces[i]);

        Mat imgres=Mat::zeros(100,100,CV_32F);         //will store a resized image of size 100*100 to be sent to predictor function

        resize(face,imgres,imgres.size(),0,0);
        imwrite("face.jpg",imgres);
        //imshow("H0",orig);
        //waitKey(0);
        Mat newface=imread("face.jpg",0);
        Ptr<FaceRecognizer> model = createFisherFaceRecognizer();      //creates own model and 
        model->load("faces_trained.xml");                              //imports trained xml
        int predictedLabel = model->predict(newface);                  //predicts the label for sent image 
        if(predictedLabel==0)                                          //if face is of ellen degenres then eyes are detected.


        {


            std::vector<Rect> eyes;
            faces[i].x/=0.95;faces[i].y/=0.95;faces[i].width/=1.2;faces[i].height/=1.2;    //eyes are detected for original face sizes
            face = orig(faces[i]);
            eye_cascade.detectMultiScale(face, eyes, 1.1, 2, CV_HAAR_SCALE_IMAGE, cv::Size(20,20));
            for(int j=0;j<eyes.size();j++)
            {

                // draws rectangle around detected eyes
                rectangle( img, Point(faces[i].x+eyes[j].x,faces[i].y+eyes[j].y), Point(faces[i].x+ eyes[j].x + eyes[j].width , faces[i].y+eyes[j].y + eyes[j].height ), Scalar(0,0,255), 4, 8, 0 );
                
                cout<<"BGR value at center of eye "<<j+1<<" is ";
                int a=faces[i].y+eyes[j].y+eyes[j].width/2;int c=faces[i].x+eyes[j].x+eyes[j].height/2;    //centre coordinates of eyes
                //cout<<a<<endl<<c<<endl;
                int b = img.at<cv::Vec3b>(a,c)[0];
                int g = img.at<cv::Vec3b>(a,c)[1];
                int r = img.at<cv::Vec3b>(a,c)[2];
                cout<<b<<" "<<g<<" "<<r<<endl;
            }
        }
    }
    imshow( result_window, img );
    waitKey(0);




    return 0;
}

