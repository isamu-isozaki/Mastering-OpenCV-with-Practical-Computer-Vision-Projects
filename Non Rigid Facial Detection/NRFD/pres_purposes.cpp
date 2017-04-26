#include "patch_models.h"

int main() {
	cv::Size rect_size(250, 350);
	cv::Mat patch, img;
	ft_data data = load_ft<ft_data>("annotation.yml");
	patch_models pmodels = load_ft<patch_models>("patch_models.yml");

	

	cv::namedWindow("Tracking win");
	cv::VideoCapture cap;
	cap.open(0);
	/*
	for (int i = 0; i < pmodels.patches.size(); i++) {
		img = data.get_image(i, 0);
		patch = pmodels.patches[i].calc_response(img);
		cv::imshow("Tracking win", patch);
		cv::waitKey(30);
	}*/
	
	for (int i = 0; i < pmodels.patches.size(); i++) {
		cap >> img;
		patch = pmodels.patches[i].calc_response(img);
		cv::imshow("Tracking win", patch);
		cv::waitKey(30);
	}
}