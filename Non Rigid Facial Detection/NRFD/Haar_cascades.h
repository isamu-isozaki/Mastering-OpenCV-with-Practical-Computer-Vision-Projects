#include "patch_models.h"

class Haar_cascades
{
public:
	Haar_cascades();
	void train(const ft_data &data);
	std::vector<std::string> pos_samples_loc;
	std::vector<cv::Mat> pos_samples;
	std::vector<std::string> neg_samples_loc;
	std::vector<cv::Mat> neg_samples;
private:
	cv::Rect getBouding(const ft_data& data, const int &idx, const int &flag);
};

