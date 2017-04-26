#include "ft_data.h"

int main() {
	int idx;
	int flag = 3;
	std::cin >> idx;
	ft_data face_data;
	face_data = load_ft<ft_data>("annotation.yml");

	face_data.display_img(idx, flag); // for checking whether it works
}