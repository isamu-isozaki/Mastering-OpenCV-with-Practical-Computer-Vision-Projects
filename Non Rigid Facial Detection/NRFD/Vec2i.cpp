#include "Vec2i.h"


Vec2i::Vec2i()
{

}

Vec2i::Vec2i(const std::vector<int>& assign_data) {
	try{
		if (assign_data.size() == 2)
			data = assign_data;
		else
			throw(std::runtime_error("vector is invalid"));
	}
	catch(std::runtime_error error) {
		std::cout << error.what() << std::endl;
	}
	
}
