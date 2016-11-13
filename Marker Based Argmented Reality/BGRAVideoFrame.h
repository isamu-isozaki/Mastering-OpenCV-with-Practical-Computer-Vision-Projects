#ifndef OPEN_CV
#include <opencv2\opencv.hpp>
#endif
class BGRAVideoFrame
{
public:
	BGRAVideoFrame();
	BGRAVideoFrame(cv::Mat);
	cv::Mat BGRA;
};

BGRAVideoFrame::BGRAVideoFrame(cv::Mat img) : BGRA(img){}
