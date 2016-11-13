#include <opencv2\opencv.hpp>
#include <iostream>
#include "MarkerDetector.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	namedWindow("MBAR", WINDOW_AUTOSIZE);
	int cameraNum = 0;
	if (argc > 1) {
		cameraNum = atoi(argv[1]);
	}
	VideoCapture camera;
	camera.open(cameraNum);
	if (!camera.isOpened()) {
		cerr << "Camera is not opened" << endl;
		exit(1);
	}


	while (true) {
		Mat cameraFrame;
		camera >> cameraFrame;
		if (cameraFrame.empty()) {
			cerr << "Could not load frame" << endl;
			exit(1);
		}
		Mat displayedFrame(cameraFrame.size(), CV_8UC3);
		displayedFrame = cameraFrame;
		imshow("MBAR", displayedFrame);
		if (waitKey(33) == 27)
			break;

	}
	return EXIT_SUCCESS;

}