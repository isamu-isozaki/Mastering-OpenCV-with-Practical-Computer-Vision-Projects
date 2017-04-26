#include "shape_model.h"

int main() {
	float c = 3.0;//for clamping

	shape_model smodel = load_ft<shape_model>("shape_model.yml");
	int n = smodel.V.rows / 2;
	float scale = smodel.calc_scale(smodel.V.col(0), 200);
	float tranx = n * 150.0 / smodel.V.col(2).dot(cv::Mat::ones(2 * n, 1, CV_32F));//n * 150/sum of transformation in x direction for all n points
	float trany = n * 150.0 / smodel.V.col(3).dot(cv::Mat::ones(2 * n, 1, CV_32F));//n * 150/sum of transformation in y direction for all n points

	std::vector<float> val;
	for (int i = 0; i < 50; i++) val.push_back(float(i) / 50);//0 to 1
	for (int i = 0; i < 50; i++) val.push_back(float(50 - i) / 50);//1 to 0
	for (int i = 0; i < 50; i++) val.push_back(-float(i) / 50);//0 to -1
	for (int i = 0; i < 50; i++) val.push_back(-float(50 - i) / 50);//-1 to 0

	cv::Mat img(300, 300, CV_8UC3); cv::namedWindow("shape model");

	std::vector <cv::Point2f> detected_pts;
	
	if (detected_pts.size() == 0)
		while (true) {
			for (int k = 4; k < smodel.V.cols; k++) {
				for (int j = 0; j < val.size(); j++) {
					cv::Mat p = cv::Mat::zeros(smodel.V.cols, 1, CV_32F);
					p.at<float>(0) = scale;
					p.at<float>(2) = tranx;
					p.at<float>(3) = trany;
					p.at<float>(k) = scale * val[j] * c * std::sqrt(smodel.e.at<float>(k));//parameter for eigen vectors -> The animation will first show large transformations and proceed to show transformations with small eigen values
					
					p.copyTo(smodel.p); img = cv::Scalar::all(255);
					std::vector<cv::Point2f> q = smodel.calc_shape();
					smodel.draw_shape(img, q);
					cv::imshow("shape model", img);
					if (cv::waitKey(10) == 'q') return 0;
				}
			}
		}
	else {
		smodel.calc_params(detected_pts, cv::Mat(), c);
		std::vector<cv::Point2f> q = smodel.calc_shape();
		smodel.draw_shape(img, q);
		cv::imshow("shape model", img);
		if (cv::waitKey(10) == 'q') return 0;
	}
}