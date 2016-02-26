#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/video/tracking.hpp"
#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

#define NUM_FRAME 20
#define EXPAND_COEFFICIENT 2 //Expand_coefficient>1

void haarDetector(Mat& img, CascadeClassifier& cascade, vector<Point2f>& prePoints, vector<Point2f>& centerPoints);

void findMotorcycle(VideoCapture& cam, vector<Point2f>& prePoints, vector<Point2f>& centerPoints, Rect& rect, Mat& firstGray, int num_Frame);