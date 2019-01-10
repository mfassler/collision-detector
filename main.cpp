/*
 * collision-detector
 *
 * Using an RGB-D (depth) camera, detect people and measure the distance to them.
 *
 * Copyright 2019 Mark Fassler
 * Licensed under the GPLv3
 *
 */

#include <stdio.h>
#include <sys/time.h>

#include "librealsense2/rs.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

extern "C" {
#include <darknet.h>
}



using namespace std;


// The pre-trained neural-network for people detection:
const char* nn_cfgfile = "darknet/cfg/yolov3-tiny.cfg";
const char* nn_weightfile = "darknet/yolov3-tiny.weights";
const char* nn_meta_file = "darknet/cfg/coco.data";


int main(int argc, char* argv[]) {

	struct timeval tv;
	int _DEPTH_WIDTH = 640;
	int _DEPTH_HEIGHT = 360;
	int _FPS = 30;
	int _RGB_WIDTH = 1280;
	int _RGB_HEIGHT = 720;

	float _MAX_DISPLAY_DISTANCE_IN_METERS = 3.0;
	float _YELLOW_DISTANCE_IN_METERS = 2.0;
	float _RED_DISTANCE_IN_METERS = 1.0;

	// TODO:
	// if (config_file) {
	//    load_config_file();
	//    change_width_height_and_fps_settings();
	// }


	// ---------------------------------------------------------
	//  BEGIN:  Start RealSense
	// ---------------------------------------------------------
	rs2::context ctx;
	rs2::pipeline pipe;
	rs2::config cfg;

	cfg.enable_stream(RS2_STREAM_COLOR, _RGB_WIDTH, _RGB_HEIGHT, RS2_FORMAT_BGR8, _FPS);
	cfg.enable_stream(RS2_STREAM_DEPTH, _DEPTH_WIDTH, _DEPTH_HEIGHT, RS2_FORMAT_Z16, _FPS);

	printf("Starting the Intel RealSense driver...\n");
	auto profile = pipe.start(cfg);
	printf("...started.  (I hope.)\n");

	auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
	auto depth_scale = depth_sensor.get_depth_scale();
	printf("Depth scale: %.6f\n", depth_scale);

	rs2::align align(RS2_STREAM_COLOR);

	int YELLOW_DISTANCE = (int) (_YELLOW_DISTANCE_IN_METERS / depth_scale);
	int RED_DISTANCE = (int)  (_RED_DISTANCE_IN_METERS / depth_scale);

	// ---------------------------------------------------------
	//  END:  Start RealSense
	// ---------------------------------------------------------



	// ---------------------------------------------------------
	//  BEGIN:  Start Darknet
	// ---------------------------------------------------------
	network *dn_net = load_network((char*)nn_cfgfile, (char*)nn_weightfile, 0);

	// TODO:  what does this do?
	set_batch_network(dn_net, 1);
	srand(2222222);

	metadata dn_meta = get_metadata((char*)nn_meta_file);

	// ---------------------------------------------------------
	//  END:  Start Darknet
	// ---------------------------------------------------------


	//cv::namedWindow("RealSense", cv::WINDOW_NORMAL);

	while (true) {

		rs2::frameset frames = pipe.wait_for_frames();
		auto processed = align.process(frames);
		rs2::video_frame vid_frame = processed.get_color_frame();
		rs2::depth_frame dep_frame = processed.get_depth_frame();

		//double frame_ts = vid_frame.get_timestamp();

		auto raw_rgb_data = vid_frame.get_data();


		// -----------------------------------------------------------------
		// We have to copy the RGB frame into Darknet-land for NN detection:
		image dn_image = make_image(_RGB_WIDTH, _RGB_HEIGHT, 3);
		int h = _RGB_HEIGHT;
		int w = _RGB_WIDTH;
		int c = 3;
		int step_h = w*c;
		int step_w = c;
		int step_c = 1;

		int i, j, k;
		int index1, index2;

		for (i=0; i<h; ++i) {
			for (k=0; k<c; ++k) {
				for (j=0; j<w; ++j) {

					index1 = k*w*h + i*w + j;
					index2 = step_h*i + step_w*j + step_c*k;

					dn_image.data[index1] = ((unsigned char*)raw_rgb_data)[index2]/255.;
				}
			}
		}


		// ------------------------------------------------------------
		// We have to copy the RGB frame into OpenCV-land for display:
		cv::Mat imRGB(cv::Size(vid_frame.get_width(), vid_frame.get_height()), CV_8UC3, (void*)raw_rgb_data, cv::Mat::AUTO_STEP);
		cv::Mat imD(cv::Size(dep_frame.get_width(), dep_frame.get_height()), CV_16UC1, (void*)dep_frame.get_data(), cv::Mat::AUTO_STEP);

		cv::Vec3b px_red = cv::Vec3b(0, 0, 255);
		cv::Vec3b px_yellow = cv::Vec3b(0, 255, 255);

		for (i=0; i<imD.rows; ++i) {
			for (j=0; j<imD.cols; ++j) {
				// A little counter-intuitive, but:
				// The depth image has "shadows".  The Intel librealsense2 driver interprets
				// shadows as distance == 0.  But we will change that to distance=max, so that
				// everything else will ignore the shadows:
				if (imD.at<uint16_t>(i, j) < 20) {
					imD.at<uint16_t>(i, j) = 65535;
				}
			}
		}

		// White bounding box around the "object-too-close" detector:
		int bbox_x_min = 20000;
		int bbox_x_max = 0;
		int bbox_y_min = 20000;
		int bbox_y_max = 0;
		int depth_min = 200000;  // in integer depth_scale units
		int all_depth_min = 200000;  // in integer depth_scale units


		// ------------------------------------------------------------
		//    BEGIN:  Neural-network people-detector
		// ------------------------------------------------------------
		// We have to copy the RGB frame into OpenCV-land for display:
		// Python version doesn't use this?
		//image sized = letterbox_image(dn_image, dn_net->w, dn_net->h);
		// float *X = sized.data;
		// network_predict(net, X);

		float thresh = 0.5;
		float hier_thresh = 0.5;
		float nms = 0.45;

		network_predict_image(dn_net, dn_image);
		int nboxes = 0;
		detection *dets = get_network_boxes(dn_net, dn_image.w, dn_image.h, thresh, hier_thresh, 0, 1, &nboxes);

		if (nms) {
			do_nms_obj(dets, nboxes, dn_meta.classes, nms);
		}

		float half_w = _RGB_WIDTH / 2.0;
		float half_h = _RGB_HEIGHT / 2.0;

		for (i=0; i<nboxes; ++i) {
			//for (j=0; j<dn_meta.classes; ++j) {
			j = 0;  // class #0 is "person", the only category we care about
				if (dets[i].prob[j] > 0.0) {

					float x_center = dets[i].bbox.x * _RGB_WIDTH;
					float y_center = dets[i].bbox.y * _RGB_HEIGHT;
					float width = dets[i].bbox.w * _RGB_WIDTH;
					float height = dets[i].bbox.h * _RGB_HEIGHT;

					int x_min = (int) (x_center - dets[i].bbox.w * half_w);
					int x_max = (int) (x_min + width);
					int y_min = (int) (y_center - dets[i].bbox.h * half_h);
					int y_max = (int) (y_min + height);

					if (x_min < 0) {
						x_min = 0;
					}
					if (x_max >= _RGB_WIDTH) {
						x_max = _RGB_WIDTH - 1;
					}
					if (y_min < 0) {
						y_min = 0;
					}
					if (y_max >= _RGB_HEIGHT) {
						y_max = _RGB_HEIGHT - 1;
					}

					// Find the closest point within this box:
					bbox_x_min = 20000;
					bbox_x_max = 0;
					bbox_y_min = 20000;
					bbox_y_max = 0;
					depth_min = 200000;  // in integer depth_scale units
					for (int ii=y_min; ii<y_max; ++ii) {
						for (int jj=x_min; jj<x_max; ++jj) {
							if (imD.at<uint16_t>(ii, jj) < depth_min) {
								depth_min = imD.at<uint16_t>(ii, jj);
							}
						}
					}
					if (depth_min < all_depth_min) {
						all_depth_min = depth_min;
					}

					cv::Rect r = cv::Rect(x_min, y_min, width, height);

					if (depth_min < RED_DISTANCE) {
						cv::rectangle(imRGB, r, cv::Scalar(0,0,200), 11);
					} else if (depth_min < YELLOW_DISTANCE) {
						cv::rectangle(imRGB, r, cv::Scalar(0,230,230), 9);
					}
				}
			//}
		}

		free_image(dn_image);
		free_detections(dets, nboxes);

		float closest = all_depth_min * depth_scale;
		if (closest < _MAX_DISPLAY_DISTANCE_IN_METERS) {
			char textBuffer[255];
			sprintf(textBuffer, "%.02f m", closest);
			auto font = cv::FONT_HERSHEY_SIMPLEX;
			cv::putText(imRGB, textBuffer, cv::Point(19,119), font, 4, cv::Scalar(0,0,0), 8);  // black shadow
			cv::putText(imRGB, textBuffer, cv::Point(10,110), font, 4, cv::Scalar(255,255,255), 8);  // white text
		}

		//cv::setWindowProperty("RealSense", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
		cv::imshow("RealSense", imRGB);
		cv::waitKey(1);

		//gettimeofday(&tv, NULL);
		//printf("ts: %ld.%06ld\n", tv.tv_sec, tv.tv_usec);
	}

	return 0;
}


