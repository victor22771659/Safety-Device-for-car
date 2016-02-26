#include "moto_Detection.h"

void haarDetector(Mat& img, CascadeClassifier& cascade, vector<Point2f>& prePoints, vector<Point2f>& centerPoints)
{
	//Clear previous tracking points
	prePoints.clear();
	centerPoints.clear();

	vector<Rect> faces;
	equalizeHist(img, img);
	
	//Classifier detecting
	cascade.detectMultiScale(img, faces,
		1.1, 2, 0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		| CV_HAAR_SCALE_IMAGE
		,
		Size(30, 30));

	//Save tracking points
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++)
	{
		Mat smallImgROI;
		smallImgROI = img(*r);
		prePoints.push_back(Point(cvRound(r->x+r->width*0.4), cvRound(r->y+r->height*0.4)));
		prePoints.push_back(Point(cvRound(r->x + r->width*0.5), cvRound(r->y+r->height*0.4)));
		prePoints.push_back(Point(cvRound(r->x + r->width*0.4), cvRound(r->y + r->height*0.5)));
		prePoints.push_back(Point(cvRound(r->x + r->width*0.5), cvRound(r->y + r->height*0.6)));
		prePoints.push_back(Point(cvRound(r->x + r->width*0.6), cvRound(r->y + r->height*0.5)));
		prePoints.push_back(Point(cvRound(r->x + r->width*0.6), cvRound(r->y + r->height*0.4)));
		prePoints.push_back(Point(cvRound(r->x + r->width*0.4), cvRound(r->y + r->height*0.6)));
		prePoints.push_back(Point(cvRound(r->x + r->width*0.6), cvRound(r->y + r->height*0.6)));
		centerPoints.push_back(Point(cvRound(r->x + r->width*0.5), cvRound(r->y + r->height*0.5)));
	}
}

void findMotorcycle(VideoCapture& cam, vector<Point2f>& prePoints, vector<Point2f>& centerPoints, Rect& rect,Mat& firstGray,int num_Frame)
{
	Mat source,frame,preGrayFrame,nextGrayFrame;
	Size winSize(15, 15);
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.03);
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> firstPoints,nextPoints,Points;
	vector<double> vSum_nextDistance,vSum_iniDistance;
	double sum_nextDistance = 0, sum_iniDistance=0;
	size_t m = 0,n=0,q=0,k=0;
	firstGray.copyTo(preGrayFrame);
	while (n != prePoints.size())
	{
		Points.push_back(prePoints[n]);
		n++;
	}
	cout << "new trackpoints" << endl;
	q = k=0;

	for (int i = 0; i < NUM_FRAME; i++)
	{	
		nextPoints.clear();
		cam >> source;//read next frame
		source(rect).copyTo(frame);
		cvtColor(frame, nextGrayFrame, CV_BGR2GRAY);//convert to grayimage
		m = 0;
		calcOpticalFlowPyrLK(preGrayFrame, nextGrayFrame, Points, nextPoints, status, err, winSize,
			3, termcrit, 0, 0.001);
		//Draw first points
		if (i==0)
		{
			n = 0;
			while (n != nextPoints.size())
			{
				firstPoints.push_back(nextPoints[n]);
				n++;
			}
		}
		m = 0;
		while (m != firstPoints.size())
		{
			circle(frame, firstPoints[m], 3, Scalar(255, 0, 0), -1, 8);
			m++;
		}
		//Draw tracking points
		m = 0;
		while (m != nextPoints.size())
		{
			circle(frame, nextPoints[m], 3, Scalar(0, 255, 0), -1, 8);
			m++;
		}

		//compute sumation of distance
		q = k=0;
		while (q != nextPoints.size())
		{
			sum_nextDistance += norm(nextPoints[q] - centerPoints[k]);
			q++;
			if (q>0)
			{
				if (q % 8 == 0)
				{
					if (i <= 0)
					{
						vSum_iniDistance.push_back(sum_nextDistance);
					}
					else
					if (sum_nextDistance > vSum_iniDistance[k] * EXPAND_COEFFICIENT)
						cout << "alarm" << endl;
					sum_nextDistance = 0;
					k++;
				}
			}
		}
		imshow("result", frame);
		if (i == 0)
			imwrite("original.png", frame);
		if (i == 8)
			imwrite("oneFrame.png", frame);

		//copy nextTracking points and nextGrayFrame to previous ones
		memcpy(&Points[0], &nextPoints[0], nextPoints.size()*sizeof(Point2f));
		nextGrayFrame.copyTo(preGrayFrame);
		if (waitKey(30) >= 0) break;
	}
}