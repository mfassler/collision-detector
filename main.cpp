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
#include <unistd.h>

#include <thread>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <librealsense2/rs.hpp>

#include "Darknet.hpp"


using namespace std;


// The pre-trained neural-network for people detection:
const char* nn_cfgfile = "darknet/cfg/yolov3-tiny.cfg";
const char* nn_weightfile = "darknet/yolov3-tiny.weights";
const char* nn_meta_file = "darknet/cfg/coco.data";


double gettimeofday_as_double() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    double ts = tv.tv_sec + (tv.tv_usec / 1000000.0);

    return ts;
}



int nboxes;
detection* dets;
std::mutex mtx_a;

void worker_thread(Darknet mDarknet) {
	int _nboxes = 0;
	detection* _dets;

	//double ts0, ts1, deltaT;
	//ts0 = gettimeofday_as_double();

	while (true) {
		_dets = mDarknet.detect_scale(&_nboxes);

		mtx_a.lock();
		free_detections(dets, nboxes);
		nboxes = _nboxes;
		dets = _dets;
		mtx_a.unlock();

		//ts1 = gettimeofday_as_double();
		//deltaT = ts1 - ts0;
		//ts0 = ts1;
		//printf("NN FPS: %.03f\n", 1.0/deltaT);
	}
}


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
	Darknet mDarknet(nn_cfgfile, nn_weightfile, nn_meta_file, _RGB_WIDTH, _RGB_HEIGHT, 3);

	// Run darknet in a different thread, because it is a bit slower than the framerate:
	std::thread worker(worker_thread, mDarknet);

	// ---------------------------------------------------------
	//  END:  Start Darknet
	// ---------------------------------------------------------


	cv::namedWindow("RealSense", cv::WINDOW_NORMAL);

	while (true) {

		rs2::frameset frames = pipe.wait_for_frames();
		auto processed = align.process(frames);
		rs2::video_frame vid_frame = processed.get_color_frame();
		rs2::depth_frame dep_frame = processed.get_depth_frame();

		//double frame_ts = vid_frame.get_timestamp();

		auto raw_rgb_data = vid_frame.get_data();
		// Copy the Realsense frames into OpenCV-land:
		cv::Mat imD(cv::Size(dep_frame.get_width(), dep_frame.get_height()), CV_16UC1, (void*)dep_frame.get_data(), cv::Mat::AUTO_STEP);
		cv::Mat imRGB(cv::Size(vid_frame.get_width(), vid_frame.get_height()), CV_8UC3, (void*)raw_rgb_data, cv::Mat::AUTO_STEP);

		// copy the input RGB into Darknet:
		mDarknet.load_image(imRGB);

		for (int i=0; i<imD.rows; ++i) {
			for (int j=0; j<imD.cols; ++j) {
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


		struct _bbox bboxes_a[25];
		int i, j;

		// Copy the bounding boxes provided from the neural-network
		mtx_a.lock();
		int count = 0;
		for (i=0; i<nboxes; ++i) {
			if (count > 24) {
				break;
			}
			//for (j=0; j<dn_meta.classes; ++j) {
			j = 0;  // class #0 is "person", the only category we care about
			if (dets[i].prob[j] > 0.2) {
				bboxes_a[count].x = (dets[i]).bbox.x;
				bboxes_a[count].y = (dets[i]).bbox.y;
				bboxes_a[count].w = (dets[i]).bbox.w;
				bboxes_a[count].h = (dets[i]).bbox.h;
				count++;
			}

			//}
		}
		mtx_a.unlock();


		float half_w = _RGB_WIDTH / 2.0;
		float half_h = _RGB_HEIGHT / 2.0;

		for (i=0; i<nboxes; ++i) {
			//for (j=0; j<dn_meta.classes; ++j) {
			j = 0;  // class #0 is "person", the only category we care about
				if (dets[i].prob[j] > 0.0) {

					float x_center = bboxes_a[i].x * _RGB_WIDTH;
					float y_center = bboxes_a[i].y * _RGB_HEIGHT;
					float width = bboxes_a[i].w * _RGB_WIDTH;
					float height = bboxes_a[i].h * _RGB_HEIGHT;

					int x_min = (int) (x_center - bboxes_a[i].w * half_w);
					int x_max = (int) (x_min + width);
					int y_min = (int) (y_center - bboxes_a[i].h * half_h);
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


		float closest = all_depth_min * depth_scale;
		if (closest < _MAX_DISPLAY_DISTANCE_IN_METERS) {
			char textBuffer[255];
			sprintf(textBuffer, "%.02f m", closest);
			auto font = cv::FONT_HERSHEY_SIMPLEX;
			cv::putText(imRGB, textBuffer, cv::Point(19,119), font, 4, cv::Scalar(0,0,0), 8);  // black shadow
			cv::putText(imRGB, textBuffer, cv::Point(10,110), font, 4, cv::Scalar(255,255,255), 8);  // white text
		}

		cv::setWindowProperty("RealSense", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
		cv::imshow("RealSense", imRGB);
		cv::waitKey(1);

		//gettimeofday(&tv, NULL);
		//printf("ts: %ld.%06ld\n", tv.tv_sec, tv.tv_usec);
	}

	return 0;
}


