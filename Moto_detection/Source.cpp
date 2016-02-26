#include "moto_Detection.h"

string cascadeName = "wheel_haartraining.xml";

int main(int argc, char** argv)
{
	double t;
	Mat source, frame, grayFrame;
	namedWindow("result", 1);
	vector<Point2f> prePoints, nextPoints, centerPoints;

	//Load casecade classifier
	CascadeClassifier cascade;
	if (!cascade.load(cascadeName))
	{
		cerr << "ERROR: Could not load classifier cascade" << endl;
		return -1;
	}

	//VideoCapture cap(0); // open the default camera
	VideoCapture cap("motofilm3.mp4");
	if (!cap.isOpened()) // check if we succeeded
		return -1;

	cout << "In capture ..." << endl;
	for (;;)
	{
		//cap >> source;
		cap.read(source);
		//Setting frame size
		int r_height = source.rows;
		int r_width = source.cols / 3;
		Rect rect(r_width, 0, r_width, r_height);

		t = (double)cvGetTickCount();//Compute time consumption
		source(rect).copyTo(frame);
		cvtColor(frame, grayFrame, CV_BGR2GRAY);//convert to grayimage

		haarDetector(grayFrame, cascade, prePoints, centerPoints);
		if (prePoints.size() > 0)
			findMotorcycle(cap, prePoints, centerPoints, rect, grayFrame, NUM_FRAME);
		else
		{
			imshow("result", frame);
			if (waitKey(30) >= 0) break;
		}
		t = (double)cvGetTickCount() - t;
		cout << "Time consume " << t / ((double)cvGetTickFrequency()*1000.) << "ms" << endl;
	}
	return 0;
}