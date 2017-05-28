#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

void read_csv(const string &filename, vector<Mat> &images,
                     vector<int> &labels)
{
    ifstream file(filename.c_str(), ifstream::in);
    string line, path, label;
    while(getline(file, line))
    {
        stringstream stream(line);
        getline(stream, path, ';');
        getline(stream, label);
        if(!path.empty() && !label.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(label.c_str()));
        }
    }
}

int main() {
    const string FACE_CASCADE_PATH =
            "/home/konstantin/QtProgs/FaceRecognition/FaceRecognition/"
            "haarcascade_frontalface_alt.xml";
    const string CSV_PATH = "/home/konstantin/QtProgs/FaceRecognition/"
                            "FaceRecognition/faces.csv";
    vector<Mat> images;
    vector<int> labels;
    read_csv(CSV_PATH, images, labels);
    int imWidth = images[0].cols;
    int imHeight = images[0].rows;
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);
    CascadeClassifier haar_cascade;
    haar_cascade.load(FACE_CASCADE_PATH);
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    Mat frame;
    for(;;) {
        cap >> frame;
        Mat original = frame.clone();
        Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        for(size_t i = 0; i < faces.size(); ++i)
        {
            Rect faceRec = faces[i];
            Mat face = gray(faceRec);
            cv::resize(face, face, Size(imWidth, imHeight));
            int prediction = model->predict(face);
            string text;
            if(prediction == 0)
            {
                text = "Prediction = Kostya";
            }
            else
            {
                text = "Prediction = Nastya";
            }
            rectangle(original, faceRec, Scalar(255, 0, 0), 2);
            putText(original, text, Point(faceRec.x - 10, faceRec.y - 10),
                    FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 2.0);
        }
        imshow("face_recognizer", original);
        int key = waitKey(1);
        if(key == 27)
        {
            break;
        }
    }
    return 0;
}
