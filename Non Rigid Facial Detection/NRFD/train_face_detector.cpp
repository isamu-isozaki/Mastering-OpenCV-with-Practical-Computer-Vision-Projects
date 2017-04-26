#include "face_detector.h"

//for the cascade file
/*
Intel License Agreement
For Open Source Computer Vision Library
Copyright(C) 2000, Intel Corporation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met :
*Redistribution's of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistribution's in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and / or other materials provided with the distribution.
* The name of Intel Corporation may not be used to endorse or promote products
derived from this software without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort(including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

int main() {
	ft_data data = load_ft<ft_data>("annotation.yml");
	shape_model smodel = load_ft<shape_model>("shape_model.yml");
	face_detector detector;

	smodel.p = cv::Mat::zeros(smodel.V.cols, 1, CV_32F);//assign 0 to all elements of p
	smodel.p.fl(0) = 1.0;
	std::vector<cv::Point2f> points = smodel.calc_shape();
	cv::Mat pts = pts_to_mat(points);
	detector.train(data, "./haarcascades/haarcascade_frontalface_default.xml", pts);
	save_ft<face_detector>("face_detector.yml", detector);
}