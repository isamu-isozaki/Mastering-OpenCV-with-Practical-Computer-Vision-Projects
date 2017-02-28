#ifndef OPEN_CV
#define OPEN_CV

#include <opencv2\opencv.hpp>

#endif

#include "fileStorage.h"
#include <fstream>

class ft_data
{
public:
	ft_data();
	ft_data(const std::string& csv, const std::vector<int>& symmetry, std::vector<std::vector<int>> connections);//import pre-annotated data
	std::vector<int> symmetry;
	std::vector <std::vector<int> > connections;//connections of features(annotations in forming a face, each point connects to two other points.
	std::vector<std::vector<cv::Point2f> > points;//annotations

	void write(cv::FileStorage& fs) const;//save to yml file
	void read(const cv::FileStorage& node);//read from the file to the variable

	void display_img(const int& idx, const int& flag);//display image with annotions on it with the flipping and color option
private:
	std::vector<std::string> imnames;//image names

	cv::Mat get_image(const int& idx, const int& flag);//loads an image at a specified index and optionally mirrors it
	std::vector<cv::Point2f> get_points(const int& idx, const bool& flipped);//get points corresponding to the image
	void rm_incomplete_samples();//remove samples with no corresponding annotations
};

void write(cv::FileStorage &fs, const ft_data x);

void read(const cv::FileStorage& node, ft_data& x, const ft_data& default);