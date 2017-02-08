#ifndef OPEN_CV
#define OPEN_CV

#include <opencv2\opencv.hpp>

#endif

#include "ft_data.h"

int main() {
	ft_data face_data("../muct/landmarks/muct76-opencv.csv");
	std::cout << face_data.imnames.size() << std::endl;

	//save_ft<ft_data>("annotation.yml", face_data);//save to yml file

	face_data.display_img(0, 2);
}