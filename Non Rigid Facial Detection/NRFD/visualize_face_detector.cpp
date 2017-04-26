#include "face_detector.h"

int main() {
	ft_data data = load_ft<ft_data>("annotation.yml");
	shape_model smodel = load_ft<shape_model>("shape_model.yml");
	patch_models pmodels = load_ft<patch_models>("patch_models.yml");
	face_detector detector = load_ft<face_detector>("face_detector.yml");
	detector.reference = pmodels.reference;
	
	cv::VideoCapture capture;
	capture.open(0);
	cv::Mat frame;
	cv::namedWindow("Show Frame");
	while (true) {
		capture >> frame;
		cv::Mat grayFrame;
		if (frame.channels() != 1) cv::cvtColor(frame, grayFrame, CV_BGR2GRAY);
		else grayFrame = frame;
		std::vector<cv::Point2f> pts = detector.detect(grayFrame);
		pmodels.calc_peaks(grayFrame, pts);
		//smodel.calc_params(pts);
		//pts = smodel.calc_shape();
		for (cv::Point2f pt : pts){
			cv::circle(grayFrame, pt, 3, cv::Scalar(255, 0, 0));
		}
		cv::imshow("Show Frame", grayFrame);
		if(cv::waitKey(30) == 27) break;
		
	}
}