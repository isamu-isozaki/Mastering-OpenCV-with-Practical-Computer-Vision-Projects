#include "Pattern.h"
//#ifndef PATTERN
//#define PATTERN

Pattern::Pattern() {

}

Pattern::Pattern(const cv::Mat& data, const cv::Mat& frame, const cv::Size& size, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& desctiptors, const std::vector < cv::Point2f>& point2d, const std::vector<cv::Point3f>& point3d)
	:data(data), frame(frame), size(size), keypoints(keypoints), descriptors(descriptors), point2d(point2d), point3d(point3d)
{
	
}
//#endif