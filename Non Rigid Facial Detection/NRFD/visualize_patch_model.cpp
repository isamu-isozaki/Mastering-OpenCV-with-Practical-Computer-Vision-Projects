#include "patch_models.h"

int main() {
	cv::Size rect_size(250, 350);
	ft_data data = load_ft<ft_data>("annotation.yml");
	shape_model smodel = load_ft<shape_model>("shape_model.yml");
	patch_models pmodels = load_ft<patch_models>("patch_models.yml");

	cv::namedWindow("Tracking win");
	
	cv::VideoCapture capture;
	capture.open(0);
	int n = pmodels.reference.rows / 2;
	std::vector<cv::Point2f> points;//contrains all the points for the model loc
	std::vector<cv::Point2f> init_points;//contrains all the points foqr the model loc
	float scale;//the scale factor the reference image will need to move
	float xmin = smodel.V.col(0).fl(0), xmax = smodel.V.col(0).fl(0);
	for (int i = 0; i < n; i++) {
		xmin = std::min(xmin, smodel.V.col(0).fl(2 * i));
		xmax = std::max(xmax, smodel.V.col(0).fl(2 * i));
	}
	scale = float(rect_size.width) / (xmax - xmin);
	cv::Mat frame;
	cv::Mat output;
	capture >> frame;
	frame.convertTo(output, CV_32F, 1.0 / 255.0);
	cv::cvtColor(output, output, CV_RGB2GRAY);
	int middle_x = frame.cols / 2, middle_y = frame.rows / 2;

	smodel.p = cv::Mat::zeros(smodel.V.cols, 1, CV_32F);

	smodel.p.fl(0) = scale;
	smodel.p.fl(2) = n * middle_x / smodel.V.col(2).dot(cv::Mat::ones(2 * n, 1, CV_32F));
	smodel.p.fl(3) = n * middle_y / smodel.V.col(3).dot(cv::Mat::ones(2 * n, 1, CV_32F));

	points = smodel.calc_shape();
	init_points = points;
	while (1) {
		capture >> frame;
		frame.convertTo(output, CV_32F, 1.0 / 255.0);
		cv::cvtColor(output, output, CV_RGB2GRAY);
		cv::Mat outline = cv::Mat::zeros(frame.size(), CV_32F);
		cv::Rect face_rect(cv::Point2f(middle_x - rect_size.width / 2., middle_y - rect_size.height / 2.), rect_size);
		cv::rectangle(outline, face_rect, cv::Scalar(255, 0, 0));
		for (int i = 0; i < points.size(); i++) {
			cv::Point2f point = points[i];		
			cv::circle(outline, point, 1, cv::Scalar(255, 0, 0));
			for (int j = 0; j < data.connections[i].size(); j++) {
				cv::line(outline, point, points[data.connections[i][j]], cv::Scalar(255, 0, 0));
			}
		}
		cv::addWeighted(output, 0.8, outline, 0.7, 0, output);
		cv::Mat face_area = frame(face_rect);
		points = pmodels.calc_peaks(face_area, points, cv::Size(44, 44));
		
	
		smodel.calc_params(points);
		points = smodel.calc_shape();
		/*
		if (smodel.p.fl(0) < 0.6 * scale || scale*1.6 < smodel.p.fl(0)) {
			std::cout << "re-adjusted" << std::endl;
			points = init_points;
		}*/
		
		cv::imshow("Tracking win", output);
		char wait = cv::waitKey(100);
		if (wait == 'q')
			break;
		if (wait == 'a')
			points = init_points;
	}
}