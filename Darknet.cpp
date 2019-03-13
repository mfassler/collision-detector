
#include "Darknet.hpp"
#include <sys/time.h>
//#include <unistd.h>

void printtimeofday() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	printf("ts: %ld.%06ld\n", tv.tv_sec, tv.tv_usec);

}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

void convert_cvMat_to_image(cv::Mat mat, image dn_image) {
	int w = mat.cols;
	int h = mat.rows;
	int c = mat.channels();

	int step_h = w*c;
	int step_w = c;
	int step_c = 1;

	int y, x, k;
	int index1, index2;

	for (y=0; y<h; ++y) {
		for (x=0; x<w; ++x) {
			for (k=0; k<c; ++k) {

				index1 = k*w*h + y*w + x;
				index2 = step_h*y + step_w*x + step_c*(2-k);

				dn_image.data[index1] = ((unsigned char*)mat.data)[index2]/255.;
			}
		}
	}
}



void crop_cvMat_to_image2(cv::Mat mat, image dn_image, int top, int left) {
	int ww = mat.cols;
	int hh = mat.rows;
	int cc = mat.channels();

	int w = dn_image.w;
	int h = dn_image.h;
	int c = dn_image.c;

	int step_h = ww*cc;
	int step_w = cc;
	int step_c = 1;

	int y, x, k;
	int index1, index2;

	for (y=0; y<h; ++y) {
		for (x=0; x<w; ++x) {
			for (k=0; k<c; ++k) {

				index1 = k*w*h + y*w + x;
				index2 = step_h*(y+top) + step_w*(x+left) + step_c*(2-k);
				dn_image.data[index1] = ((unsigned char*)mat.data)[index2]/255.;
			}
		}
	}
}



Darknet::Darknet(const char* cfgfile, const char* wtsfile, const char* metafile, int width, int height, int channels) {

	dn_net = load_network((char*)cfgfile, (char*)wtsfile, 0);

	set_batch_network(dn_net, 1);  // <---  TODO:  what does this do?
	srand(2222222);

	dn_image_c = make_image(width, height, channels);
}


void Darknet::load_image(cv::Mat im) {
	convert_cvMat_to_image(im, dn_image_c);
}

detection* Darknet::detect_scale(int *nboxes) {

	//printf("convert...\n");
	//printtimeofday();

	//printf("predict...\n");
	//printtimeofday();
	network_predict_image(dn_net, dn_image_c);

	//printf("get boxes...\n");
	//printtimeofday();
	detection *dets = get_network_boxes(
		dn_net, dn_image_c.w, dn_image_c.h, thresh, hier_thresh, 0, 1, nboxes
	);

	if (nms) {
		do_nms_obj(dets, *nboxes, 1, nms);
	}

	return dets;
}




