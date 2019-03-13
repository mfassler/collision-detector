#ifndef __DARKNET_HPP_
#define __DARKNET_HPP_

extern "C" {
#include <darknet.h>
}

#include <opencv2/core.hpp>



// Same as in darknet.h:
struct _bbox {
	float x, y, w, h;
};


class Darknet {
private:
	float thresh = 0.5;
	float hier_thresh = 0.5;
	float nms = 0.45;
	network *dn_net;

public:
	Darknet(const char*, const char*, const char*, int, int, int);
	void load_image(cv::Mat);
	detection *detect_scale(int*);

	image dn_image_c;
};


#endif // __DARKNET_HPP_

