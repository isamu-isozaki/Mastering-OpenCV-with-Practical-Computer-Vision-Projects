#include "Marker.h"

float perimeter(std::vector<cv::Point2f> points) {
	float perimeter_squared = 0;
	for (int c = 0; c < points.size(); c++) {
		cv::Point2f v = points[(c + 1) % 4] - points[c];
		perimeter_squared += v.dot(v);
	}
	return perimeter_squared;
}

cv::Mat rotate(cv::Mat id, int degrees) {
	cv::Mat output;
	switch (degrees)
	{
	case 90:
		cv::transpose(id, output);
		cv::flip(output, output, 1);//check
	case -90:
		cv::transpose(id, output);
		cv::flip(output, output, 0);
	case 180:
		cv::flip(output, output, 1);
		cv::flip(output, output, 0);
	default:
		std::cerr << "Please enter a valid value" << std::endl;
		exit(1);
	}
	return output;
}

std::pair<int, int> hammDistMarker(cv::Mat checkValid) {
	int id[5][5] = {
		{ 1, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 0 },
		{ 1, 0, 1, 1, 1 },
		{ 1, 0, 1, 1, 1 },
		{ 1, 0, 1, 1, 1 }
	};
	const cv::Mat ids[4] = {
		cv::Mat(5, 5, CV_8UC1, &id),
		rotate(cv::Mat(5, 5, CV_8UC1, &id), 90),
		rotate(cv::Mat(5, 5, CV_8UC1, &id), 180),
		rotate(cv::Mat(5, 5, CV_8UC1, &id), -90)
	};
	int sum = 0;
	std::pair<int, int> minDist(0, 5);
	for (int r = 0; r < 4; r++){
		for (int y = 0; y < 5; y++){
			for (int x = 0; x < 5; x++){
				sum += checkValid.at<uchar>(y, x) == ids[r].at<uchar>(y, x) ? 0 : 1;
			}
		}
		if (minDist.second == 5 || minDist.first > sum) {
			minDist.first = sum;
			minDist.second = r;
		}

	}
	return minDist;
}