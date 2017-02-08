#include "ft_data.h"


ft_data::ft_data()
{
}

ft_data::ft_data(const std::string& csv) {//creates annotation from csv
	std::ifstream file(csv);//thanks http://answers.opencv.org/question/55210/reading-csv-file-in-opencv/!
	try{
		if (file.good() == false) throw(std::runtime_error("File does not exist at " + csv));
	}
	catch(std::runtime_error error) {
		std::cout << error.what() << std::endl;
	}
	std::cout << "Csv file detected" << std::endl;
	std::string current_line;
	std::vector<std::vector<std::string>> all_data;

	while (std::getline(file, current_line)) {
		std::vector<std::string> vals;
		std::stringstream current_stream(current_line);
		std::string single_val;

		while (std::getline(current_stream, single_val, ',')) {
			vals.push_back(single_val);
		}
		all_data.push_back(vals);
	}

	std::vector<std::string> params = all_data[0];

	all_data.erase(all_data.begin());//only the data no param

	for (int i = 0; i < all_data.size(); i++) {
		//thanks http://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c for the tip on how to find whether a file exists
		std::vector<std::string> line = all_data[i];
		std::string img_loc = "../muct/pictures/jpg/" + line[0] + ".jpg";
		std::ifstream img(img_loc);
		if (img.good()) {//if the image exists
			img.close();//no need so close the stream
			imnames.push_back(img_loc);//push the image location to imnames
			std::vector<cv::Point2f> face_points;//points of annotation for each face
			for (int j = 0; j < line.size(); j++) {
				if (params[j] == "tags") continue;//if the parameter to be evaluated is tags, skip it
				cv::Point2f point;//to store each point, in the csv file points go in the order x00,y00,x01,y01,x02,y02
				if (params[j].find('y') != std::string::npos) {// thanks http://stackoverflow.com/questions/2340281/check-if-a-string-contains-a-string-in-c, if the letter y could be found
					int x_val = atoi(line[j - 1].c_str());//x will be at index -1 of that
					int y_val = atoi(line[j].c_str());
					if ((x_val == 0) || (y_val == 0)) {//if there is no points with 0, 0, since those will be the points that will be invalid
						imnames.erase(imnames.begin() + i);//remove image name
						all_data.erase(all_data.begin() + i);
						i--;
						goto next_iter;//since I could not find any other way out of it
					}
					else {
						point = cv::Point2f(x_val, y_val);//create the point and
						face_points.push_back(point);//push it to points
					}					
				}
				else//if it is an x coordinate
					continue;
			}
			if (face_points.size() != 0)
				points.push_back(face_points);//ignored if there is even 1 x or y coordinate adnormality
		}
		else {//if the image does not exist
			continue;
		}
	next_iter://go to next iteration literally
		continue;//just for clarity
	}

}

void ft_data::write(cv::FileStorage& fs) const {//save to yml file
	assert(fs.isOpened());

	fs << "{" << "annotations" << points << "img_names" << imnames << "connections" << connections << "symmetry" << symmetry << "}";
}

void write(cv::FileStorage &fs, const ft_data x) {
	x.write(fs);
}

void ft_data::read(const cv::FileStorage& fs) {//read from yml file
	assert(fs["annotations"].type() == cv::FileNode::MAP);
	assert(fs["img_names"].type() == cv::FileNode::MAP);
	assert(fs["connections"].type() == cv::FileNode::MAP);
	assert(fs["symmetry"].type() == cv::FileNode::MAP);
	fs["annotations"] >> points;
	fs["img_names"] >> imnames;
	fs["connections"] >> connections;
	fs["symmetry"] >> symmetry;
}

void read(const cv::FileStorage& fs, ft_data& x, const ft_data& default) {
	if (fs["annotations"].empty()) x = default;
	if (fs["img_names"].empty()) x = default;
	//if (fs["connections"].empty()) x = default;
	//if (fs["symmetry"].empty()) x = default;
	else x.read(fs);
}

cv::Mat ft_data::get_image(const int& idx, const int& flags){
	if (idx < 0 || (idx >= (int)imnames.size())) return cv::Mat();
	cv::Mat img, im;
	if (flags < 2)
		img = cv::imread(imnames[idx], 0);//grayscale
	else
		img = cv::imread(imnames[idx], 1);//3 channel
	if (flags % 2 != 0)
		cv::flip(img, im, 1);//flip over the y axis
	else
		im = img;
	return im;
}

std::vector<cv::Point2f> ft_data::get_points(const int& idx, const bool& flipped) {//index and bool for flip
	if (idx < 0 || (idx >= (int)imnames.size())) return std::vector<cv::Point2f>();
	try {
		std::vector<cv::Point2f> p = points[idx];
		if (p == std::vector<cv::Point2f>()) {
			throw(std::runtime_error("Image has not been annotated at index " + idx));//if the image has not been annotated throw an error
		}
		if (flipped) {
			cv::Mat im = this->get_image(idx, 0);//or can pass image width as variable
			int n = p.size();
			std::vector<cv::Point2f> q(n);
			for (int i = 0; i < n; i++) {
				q[i].x = im.cols - 1 - p[symmetry[i]].x;//why -1 and why not just q[i].x = p[symmetry[i]].x
				q[i].y = p[symmetry[i]].y;
			}
			return q;
		}
		else {
			return p;
		}
	}
	catch (std::runtime_error error) {
		std::cout << error.what() << std::endl;
		return std::vector<cv::Point2f>();
	}
	
	
}

void ft_data::rm_incomplete_samples() {
	int n = 0, N = points.size();
	for (int i = 0; i < N; i++) n = std::max(n, int(points[i].size()));//selects the maximum size of the points and assign it to n, it ought to be the cannonical sample
	for (int i = 0; i < N; i++) {
		if (int(points[i].size()) != n){
			points.erase(points.begin() + i);
			imnames.erase(imnames.begin() + i);
			i--;//without this, the next loop will skip one element
		}
		else {
			int j = 0;
			for (; j < n; j++) {
				if ((points[i][j].x <= 0) || (points[i][j].y <= 0)) break;//if the value given to a coordinate is negative or 0, j will be smaller than n because it may possibly be due to ambiguity
			}

			if (j < n) {//if j is smaller than n
				points.erase(points.begin() + i);
				imnames.erase(imnames.begin() + i);
				i--;
			}
		}
	}

}

void ft_data::display_img(const int& idx, const int& flags) {
	cv::Mat output;
	cv::namedWindow("display_annotations");

	cv::Mat img = get_image(idx, flags);
	std::vector<cv::Point2f> points = get_points(idx, bool(flags-2));

	int type = 0;

	if (flags < 2) {//depending on get_image function
		type = CV_8UC1;
	}
	else {
		type = CV_8UC3;
	}

	cv::Mat overlay = cv::Mat::zeros(img.size(), type);

	std::cout << "point nums: " << points.size() << std::endl;

	for (cv::Point2f point : points) {
		cv::circle(overlay, point, 5, cv::Scalar(0, 250, 0), 6);//plot the points
		
	}

	cv::addWeighted(img, 0.7, overlay, 0.7, 0, output, type);
	cv::imshow("display_annotations", overlay);
	cv::waitKey(0);
	cv::imshow("display_annotations", output);
	cv::waitKey(0);//wait for user consent 

	cv::destroyWindow("display_annotations");
	output.release();
	overlay.release();
	img.release();
}
