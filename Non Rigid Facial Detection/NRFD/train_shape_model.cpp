#include "shape_model.h"

int main() {
	float frac = 0.95;//for PCA for smodel.train, if it describes this portion of the variation correctly, stop there
	int kmax = 20;

	bool mirror = true;

	ft_data face_data;
	face_data = load_ft<ft_data>("annotation.yml");

	std::vector<std::vector<cv::Point2f>> points;

	for (int i = 0; i < int(face_data.points.size()); i++){
		points.push_back(face_data.get_points(i, false));
		if (mirror) points.push_back(face_data.get_points(i, true));
	}

	shape_model smodel; smodel.train(points, face_data.connections, frac, kmax);

	save_ft<shape_model>("shape_model.yml", smodel);
}