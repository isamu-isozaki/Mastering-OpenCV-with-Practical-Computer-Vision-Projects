#ifndef OPEN_CV
#define OPEN_CV

#include <opencv2\opencv.hpp>

#endif

class Vec2i
{
public:
	Vec2i();
	Vec2i(const std::vector<int>&);
	std::vector<int> data = std::vector<int>(2);
};

