#ifndef OPEN_CV
#define OPEN_CV

#include <opencv2\opencv.hpp>

#endif


class annotation
{
public:
	annotation();
	std::string im_name;
	std::vector<cv::Point2f> points;
	static std::vector<int> symmetry;
	static std::vector<Vec2i.data> connections;

	~annotation();
};

