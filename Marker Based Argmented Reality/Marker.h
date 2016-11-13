#ifndef OPEN_CV_
#include <opencv2\opencv.hpp>
#endif

class Marker
{
public:
	std::vector<cv::Point2f> points;
private:
};


float perimeter(std::vector<cv::Point2f>);
std::pair<int, int> hammDistMarker(cv::Mat);
cv::Mat rotate(cv::Mat, int);