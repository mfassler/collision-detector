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

#include "CDNeuralNet.hpp"

#ifdef USE_NETWORK_DISPLAY
#include "UdpSender.hpp"
#endif // USE_NETWORK_DISPLAY

using namespace std;


// The pre-trained neural-network for people detection:
const char* nn_weightfile = "darknet/yolov3-tiny.weights";
const char* nn_cfgfile = "darknet/cfg/yolov3-tiny.cfg";
const char* nn_meta_file = "darknet/cfg/coco.data";


double gettimeofday_as_double() {
	struct timeval tv;
	gettimeofday(&tv, NULL);

	double ts = tv.tv_sec + (tv.tv_usec / 1000000.0);

	return ts;
}



cv::Mat imRGB;


void worker_thread(CDNeuralNet _cdNet) {
	while (true) {
		if (imRGB.cols > 10) {
			_cdNet.detect(imRGB);
		} else {
			usleep(5000);
		}
	}
}

// Command-line options:
std::string keys =
	"{ help   h | | Print help message. }"
	"{ serial   | | choose specific RealSense by serial number}";

int main(int argc, char* argv[]) {

	cv::CommandLineParser parser(argc, argv, keys);

	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	const std::string serialNumber = parser.get<cv::String>("serial");

	struct timeval tv;
	int _DEPTH_WIDTH = 848;
	int _DEPTH_HEIGHT = 480;
	int _FPS = 30;
	int _RGB_WIDTH = 848;
	int _RGB_HEIGHT = 480;

	float _MAX_DISPLAY_DISTANCE_IN_METERS = 3.0;
	float _YELLOW_DISTANCE_IN_METERS = 2.0;
	float _RED_DISTANCE_IN_METERS = 1.0;

#ifdef USE_NETWORK_DISPLAY
	int local_id = 0;
	if (argc < 2) {
		printf("Usage:  %s server_ip_address [local_id]\n", argv[0]);
		return -1;
	}
	if (argc > 2) {
		local_id = atoi(argv[2]);
		if (local_id < 0) {
			local_id = 0;
		} else if (local_id > 255) {
			local_id = 0;
		}
	}
	printf("local_id: %d\n", local_id);

	const char* ServerAddress = argv[1];
	const uint16_t DataRxPort = 3101;
	const uint16_t ImageRxPort = 3201;

	UdpSender udpSender(ServerAddress, ImageRxPort);

#endif //USE_NETWORK_DISPLAY

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

	if (serialNumber.length() > 1) {
		printf("Trying to open RealSense serial#: %s\n", serialNumber.c_str());
		cfg.enable_device(serialNumber);
	}
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

	auto rgb_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
	const rs2_intrinsics rgb_intrinsics = rgb_stream.get_intrinsics();
	printf("intrinsics: %d %d  %f %f %f %f\n", 
		rgb_intrinsics.width, rgb_intrinsics.height,
		rgb_intrinsics.ppx, rgb_intrinsics.ppy, rgb_intrinsics.fx, rgb_intrinsics.fy);


	// ---------------------------------------------------------
	//  END:  Start RealSense
	// ---------------------------------------------------------



	// ---------------------------------------------------------
	//  BEGIN:  Start NN
	// ---------------------------------------------------------
	std::string modelPath = std::string(nn_weightfile);
	std::string configPath = std::string(nn_cfgfile);

	CDNeuralNet cdNet(modelPath, configPath);

	// Run the neural network in a different thread:
	std::thread worker(worker_thread, cdNet);

	// ---------------------------------------------------------
	//  END:  Start Darknet
	// ---------------------------------------------------------

#ifdef USE_LOCAL_DISPLAY
	cv::namedWindow("RealSense", cv::WINDOW_NORMAL);
#endif // USE_LOCAL_DISPLAY

	while (true) {

		rs2::frameset frames = pipe.wait_for_frames();
		auto processed = align.process(frames);
		rs2::video_frame vid_frame = processed.get_color_frame();
		rs2::depth_frame dep_frame = processed.get_depth_frame();

		//double frame_ts = vid_frame.get_timestamp();

		auto raw_rgb_data = vid_frame.get_data();
		// Copy the Realsense frames into OpenCV-land:
		cv::Mat imD(cv::Size(dep_frame.get_width(), dep_frame.get_height()), CV_16UC1, (void*)dep_frame.get_data(), cv::Mat::AUTO_STEP);
		imRGB = cv::Mat(cv::Size(vid_frame.get_width(), vid_frame.get_height()), CV_8UC3, (void*)raw_rgb_data, cv::Mat::AUTO_STEP);


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


		int nboxes_a;
		struct _bbox bboxes_a[25];

		// Copy the bounding boxes provided from the neural-network
		nboxes_a = cdNet.get_output_boxes(bboxes_a, 25);


		float half_w = _RGB_WIDTH / 2.0;
		float half_h = _RGB_HEIGHT / 2.0;


#ifdef USE_NETWORK_DISPLAY
		// Send all the bounding boxes and distances to the display server:
		unsigned char metaDataBuffer[1500];  // MTU of ethernet

		struct bbox_packet *bbox_packets;
		bbox_packets = (struct bbox_packet*) &(metaDataBuffer[4]);

		int pkt_number = 0;

#endif // USE_NETWORK_DISPLAY

		for (int i=0; i<nboxes_a; ++i) {

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

#ifdef USE_NETWORK_DISPLAY
			bbox_packets[pkt_number].min_distance = depth_min * depth_scale;
			bbox_packets[pkt_number].x_min = x_min;
			bbox_packets[pkt_number].x_max = x_max;
			bbox_packets[pkt_number].y_min = y_min;
			bbox_packets[pkt_number].y_max = y_max;
			pkt_number++;

#endif // USE_NETWORK_DISPLAY

#ifdef USE_LOCAL_DISPLAY
			if (depth_min < all_depth_min) {
				all_depth_min = depth_min;
			}

			cv::Rect r = cv::Rect(x_min, y_min, width, height);

			if (depth_min < RED_DISTANCE) {
				cv::rectangle(imRGB, r, cv::Scalar(0,0,200), 11);
			} else if (depth_min < YELLOW_DISTANCE) {
				cv::rectangle(imRGB, r, cv::Scalar(0,230,230), 9);
			}
#endif // USE_LOCAL_DISPLAY
		}


#ifdef USE_NETWORK_DISPLAY
		// waste 2 bytes to maintain 32-bit boundaries:
		metaDataBuffer[0] = 0;
		metaDataBuffer[1] = 0;
		metaDataBuffer[2] = local_id;
		metaDataBuffer[3] = (unsigned char) pkt_number;

		int dataSize = 4 + sizeof(struct bbox_packet) * pkt_number;

		udpSender._sendto(ServerAddress, DataRxPort, metaDataBuffer, dataSize);
		//udpSender._sendto("127.0.0.1", DataRxPort, metaDataBuffer, dataSize);

#ifdef USE_SCANLINE
		// Get a scan of one row
		const unsigned char *ptr;
		unsigned char scanOut[ 848 * 2 ];  // will be larger than ethernet MTU  :-( ...
		ptr = imD.ptr(240); // 240 is the middle row (the horizon, in theory)
		for (int col=0; col < 848*2; ++col) {
			scanOut[col] = ptr[col];
		}
		udpSender._sendto(ServerAddress, 3125, scanOut, 848*2);
		//udpSender._sendto("127.0.0.1", 3125, scanOut, 848*2);
#endif // USE_SCANLINE

		udpSender.sendImage(imRGB);
#endif // USE_NETWORK_DISPLAY


#ifdef USE_LOCAL_DISPLAY
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
#endif // USE_LOCAL_DISPLAY

		//gettimeofday(&tv, NULL);
		//printf("ts: %ld.%06ld\n", tv.tv_sec, tv.tv_usec);
	}

	return 0;
}


