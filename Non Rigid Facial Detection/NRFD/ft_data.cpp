#include "ft_data.h"


ft_data::ft_data()
{
	
}


ft_data::ft_data(const std::string& csv, const std::vector<int>& symmetry, std::vector<std::vector<int>> connections): symmetry(symmetry), connections(connections) {//creates annotation from csv
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

	fs << "annotations" << "[";//this encases all of these
	for (std::vector<cv::Point2f> anno_points : points){
		fs << "[";//each annotation
		for (cv::Point2f point : anno_points) {
			fs << point;//for each point, they already make "[]" on both sides of them for some reason
		}
		fs << "]";
	}
	fs << "]" << "img_names" << imnames << "connections" << "[";

	for (std::vector<int> connection : connections) {
		fs << "[";
		for (int connected_pt : connection) {
			fs << connected_pt;
		}
		fs << "]";
	}
	fs << "]" << "symmetry" << symmetry;
}

void write(cv::FileStorage &fs, const ft_data x) {
	x.write(fs);
}

void ft_data::read(const cv::FileStorage& fs) {//read from yml file
	assert(fs["annotations"].type() == cv::FileNode::SEQ);
	assert(fs["img_names"].type() == cv::FileNode::SEQ);
	assert(fs["connections"].type() == cv::FileNode::SEQ);
	assert(fs["symmetry"].type() == cv::FileNode::SEQ);
	
	//get annotations
	cv::FileNode fs_annotations = fs["annotations"];
	cv::FileNodeIterator anno_iter = fs_annotations.begin(), anno_end = fs_annotations.end();
	for (; anno_iter != anno_end; anno_iter++) {
		cv::FileNodeIterator point_iter = (*anno_iter).begin(), point_end = (*anno_iter).end();
		std::vector<cv::Point2f> anno_points;//vector of points in annotation
		for (; point_iter != point_end; point_iter++) {
			cv::Point2f point;//point in annotation
			*point_iter >> point;//get point
			anno_points.push_back(point);
		}
		points.push_back(anno_points);
	}

	//get img names
	cv::FileNode fs_img_names = fs["img_names"];
	
	cv::FileNodeIterator img_name_iter = fs_img_names.begin(), img_name_end = fs_img_names.end();
	
	for (; img_name_iter != img_name_end; img_name_iter++) {
		std::string imname;
		*img_name_iter >> imname;
		imnames.push_back(imname);
	}

	//get connections
	cv::FileNode fs_connections = fs["connections"];

	cv::FileNodeIterator fs_connections_iter = fs_connections.begin(), fs_connections_end = fs_connections.end();

	for (; fs_connections_iter != fs_connections_end; fs_connections_iter++) {
		cv::FileNodeIterator fs_connection_iter = (*fs_connections_iter).begin(), fs_connection_end = (*fs_connections_iter).end();
		std::vector<int> connection;
		for (; fs_connection_iter != fs_connection_end; fs_connection_iter++) {
			int connection_idx = 0;
			(*fs_connection_iter) >> connection_idx;
			connection.push_back(connection_idx);
		}
		connections.push_back(connection);
	}

	//get symmetrical indices
	cv::FileNode fs_symmetry = fs["symmetry"];
	
	cv::FileNodeIterator symmetry_iter = fs_symmetry.begin(), symmetry_end = fs_symmetry.end();
	
	for (; symmetry_iter != symmetry_end; symmetry_iter++) {
		int symmetry_idx = 0;
		*symmetry_iter >> symmetry_idx;
		symmetry.push_back(symmetry_idx);
	}
}

void read(const cv::FileStorage& fs, ft_data& x, const ft_data& default) {
	if (fs["annotations"].empty()) x = default; 
	if (fs["img_names"].empty()) x = default;
	if (fs["connections"].empty()) x = default;
	if (fs["symmetry"].empty()) x = default;
	x.read(fs);
	
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
				q[i].x = im.cols - 1 -  p[symmetry[i]].x;//find the point that is corresponding to the given point and flip it -> corresponds to img flip
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
	std::vector<cv::Point2f> points_curr = get_points(idx, (flags%2 ? false: true));

	int type = 0;

	if (flags < 2) {//depending on get_image function
		type = CV_8UC1;
	}
	else {
		type = CV_8UC3;
	}
	cv::Mat overlay = cv::Mat::zeros(img.size(), type);

	for (int idx = 0; idx < points_curr.size(); idx++) {
		//cv::circle(overlay, points_curr[idx], 3, cv::Scalar(0, 250, 0), -1);//plot the points
		
		for (int connect_idx : connections[idx]) {
				cv::line(overlay, points_curr[idx], points_curr[connect_idx], cv::Scalar(0, 250, 0), 1, 8);
		}


		cv::putText(overlay, std::to_string(idx), points_curr[idx], cv::FONT_HERSHEY_PLAIN, 0.5, cv::Scalar(250, 250, 0), 0.8);
	}

	//cv::circle(overlay, points_curr[19], 5, cv::Scalar(0, 250, 0), 6);

	cv::addWeighted(img, 0.7, overlay, 0.7, 0, output, type);

	//just in case it is too small
	//cv::Size dst_size(output.size().width * 2, output.size().height * 2);
	//cv::resize(output, output, dst_size);

	cv::imshow("display_annotations", output);
	cv::waitKey(0);//wait for user consent 

	cv::destroyWindow("display_annotations");
	output.release();
	overlay.release();
	img.release();
}

int ft_data::n_images() {
	return imnames.size();
}