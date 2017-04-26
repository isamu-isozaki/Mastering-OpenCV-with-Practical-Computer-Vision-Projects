#include "patch_models.h"

int main() {
	int width = 100;
	int psize = 22;
	int ssize = 22;

	ft_data data = load_ft<ft_data>("annotation.yml");
	
	shape_model smodel = load_ft<shape_model>("shape_model.yml");
	smodel.p = cv::Mat::zeros(smodel.V.cols, 1, CV_32F);//shape where the values of all parameters are 0
	
	smodel.p.fl(0) = smodel.calc_scale(smodel.V.col(0), width);//Because V.col(0) is the average shape
	std::vector<cv::Point2f> r = smodel.calc_shape();

	patch_models pmodel; pmodel.train(data, r, cv::Size(psize, psize), cv::Size(ssize, ssize));
	save_ft<patch_models>("patch_models.yml", pmodel);
}