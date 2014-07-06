/**
 * @file track_node.cpp
 * @brief The program entry of the endeffector_tracking system
 * @author Pengfei Wu
 * @version 1.0
 * @date 2013-03-20
 */

//#include <ros/ros.h>

// for opencv
#include "opencv2/opencv.hpp"
// for parallel computing
#include <omp.h>

// user defined
#include "endeffector_tracking/track_ros.h"

// file io
#include <fstream>

// for timing and boost the performance of the system
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

#define VIDEO 0
#define ROS   1

// TODO: change the face appear and disappear angles
// TODO: maintain the final codes and give substantial comments
// TODO: kill a the TODOs then remove this one

using namespace std;

// global variable maintaned
// TODO: some of them should be moved into the main function

/**
 * @brief The status of the current tracker
 */
tracking_status status = INITIAL;



//image_transport::Subscriber subCam;
//image_transport::Publisher imgPub;
// several config file obtained from the launch file
std::string camera_topic;
std::string config_file;
std::string model_name;
std::string init_file;
/* End of ROS param */

/* current state */
/**
 * @brief  current frame
 */
cv::Mat curImg;

/**
 * @brief  the processed ver. of current frame to display the tracking rst
 */
cv::Mat processedImg;

// pose here is either generated manually or measured by the connected hough lines

/**
 * @brief  	the current pose
 */
vpHomogeneousMatrix cMo;
/**
 * @brief  the pose vector corresponding to the pose matrix
 * 			tx, ty, tz, thetaux, thetauy, thetauz
 */
vpPoseVector poseVector;

/**
 * @brief  current window obtained by KernelBasedTracking
 */
cv::Rect TrackerWindow;

/* End of current state */

/**
 * @brief  rows and cols
 */
int rows, cols;

/**
 * @brief  the camera currently used
 */
vpCameraParameters cam;

// different type of trackers
//mbtEdgeTracker meTracker;
//kltFbTracker fbTracker;
superResolutionTracker srTracker;
// TODO: init the super-resolution tracker here, and make sure that both the following sections can use the tracker.

// for display
vpDisplayX d;

/**
 * @brief The program entry point 
 *
 * @param argc
 * @param argv
 *
 * @return 
 */
int main(int argc, char **argv)
{
	//instantiate the endeffector_tracking class
	//the ros initial steps are done in the constructor, so only instantiate the CEndeffectorTracking class is sufficient
	std::string name = "endeffector_tracking";

	omp_set_nested(1);
#pragma omp parallel 
	{
#pragma omp sections nowait
		{
#pragma omp section
			{
				// for real-time frame processing
				// using the ros
				//

				//ros::init(argc, argv, name);

				///* ROS params */
				//ros::NodeHandle n;


				////get param from the launch file or command line

				////param([param_name], [member_variable], [default_value])

				////FIXME:the param namespace is weird, how to figure out the namespace of an param?
				//n.param<std::string>("endeffector_tracking/camera_topic", camera_topic, "camera/image_raw");
				////the config_file containing the info of the cam
				//n.param<std::string>("endeffector_tracking/config_file", config_file, "config.xml");
				//n.param<std::string>("endeffector_tracking/model_name", model_name, "config.cao");
				//n.param<std::string>("endeffector_tracking/init_file", init_file, "config.init");
				//		
				//image_transport::ImageTransport it(n);
				//cv::VideoCapture cap(0); // open the default video file
				cv::VideoCapture cap("/home/pengfei/Desktop/6.avi"); // open the default video file
				cv::Mat tmp;
				for (int i = 0; i < 174; i++)  //174 for 6.avi
					cap >> tmp;



				int MODE = VIDEO;
				std::ofstream f1("pose_tracked.log");
				switch (MODE)
				{
					case VIDEO:
						config_file = "config.xml";
						model_name = "config.cao";
						init_file = "config.init";
						// capture frames

						if(cap.isOpened()) // check if we succeeded
							for (;;)
							//for (int j = 0; j < 1025; j++)
							{
								// retrieve image from the camera
								cv::Mat curImg;
								cap >> curImg; // get a new frame from camera
								if(curImg.empty())
								{
									std::cout<<"The tracking procedure finished successfully!"<<std::endl;
									break;
								}
								//tracking based on different status
								if (status == LOST || status == INITIAL)
								{
									// init the model from file
									// only requires when initializing
									if (status == INITIAL)
									{
										// get some basic info about the video
										cv::Mat grayImg;
										cv::cvtColor(curImg, grayImg, CV_BGR2GRAY);
										rows = grayImg.rows;
										cols = grayImg.cols;

										//initializeTracker(srcImg);
										srTracker.initialization(curImg, config_file, model_name, init_file);
										srTracker.vpMbTracker::getPose(cMo);
									}
									//finish the initialization step and start to track
									status = TRACKING;
								}	
								else if (status == TRACKING)
								{
									clock_t t; //timing
									t = clock();
									srTracker.retrieveImage(curImg);
									srTracker.track();
									srTracker.vpMbTracker::getPose(cMo);
									t = clock() - t;
									//std::cout<<"single tracking iteration consumes "<<((float)t)/CLOCKS_PER_SEC<<" seconds."<<std::endl;
								}

								// save the pose into the file
								poseVector.buildFrom(cMo);

								f1<<poseVector[0]<<" "<<poseVector[1]<<" "<<poseVector[2]<<" "<<poseVector[3]<<" "<<poseVector[4]<<" "<<poseVector[5]<<" "<<std::endl;
							}
						break;
						//case ROS:
						//	// subCam is member variable subscribe to the msg published by the camera
						//	// trackCallback do the tracking procedure
						//	// the callback function is a member function of a class, so we should point out which class it belongs to explicitly
						//	subCam = it.subscribe(camera_topic, 1, trackCallback);

						//	// publish message
						//	imgPub = it.advertise("result", 1);
						//	ros::spin();
						//	break;
					default:
						break;
				}
				f1.close();
			}

#pragma omp section
			{
				// for super-resolution dataset processing
				while (true)
					srTracker.updateDataBase();
			}
		}
	}
		return 0;
}

/**
 * @brief  	The tracking procedure is done here
 * 			Note that different trackers can be employed here
 * 			This function is only the glue function of ROS and all the other 
 * 			trackers
 * 			Only some very basic info. obtained from the ROS topic is processed here
 * 			INITIAL: the init points from .init file are obtained
 * 					 if use the houghLineBasedTracker, the model based on the init points are generated and maintained in the houghLineBasedTracker (not in this interface class).
 * 			LOST: this is the reinitialization step, including the initial pose of the target, corresponding tracking window, etc.
 * 			TRACKING: the tracking function provided by other tracking classes, and the tracked object is published as a ROS topic.
 *
 *
 * @param srcImg 	the sensor_msgs of ROS topic, 
 * 					the message will be transfered to OpenCV image and then stored.
 */
//void 
//trackCallback(const sensor_msgs::ImageConstPtr& srcImg)
//{
//	// transform the image msg into cv::Mat and store in the member variable
//	msg2mat(srcImg, curImg);
//
//	//tracking based on different status
//	if (status == LOST || status == INITIAL)
//	{
//		// init the model from file
//		// only requires when initializing
//		if (status == INITIAL)
//		{
//			// get some basic info about the video
//			cv::Mat grayImg;
//			cv::cvtColor(curImg, grayImg, CV_BGR2GRAY);
//			rows = grayImg.rows;
//			cols = grayImg.cols;
//			
//			//initializeTracker(srcImg);
//			srTracker.initialization(curImg, config_file, model_name, init_file);
//			//fbTracker.getInitPoints(init_file);
//			//fbTracker.init(curImg);
//
//			//meTracker.retrieveImage(srcImg);
//			//meTracker.initialize(config_file, model_name, init_file);
//		}
//
//		//fbTracker.retrieveImage(curImg);
//		//fbTracker.initialize(cam, cMo, poseVector, rows, cols);
//
//		//finish the initialization step and start to track
//		status = TRACKING;
//	}	
//	else if (status == TRACKING)
//	{
//		srTracker.retrieveImage(curImg);
//		srTracker.track();
//	}
//}

/**
 * @brief convert from the ros image message to a CvImage
 *
 * @param srcImg
 * @param curImg_
 */
//void 
//msg2mat(const sensor_msgs::ImageConstPtr& srcImg, cv::Mat& curImg_)	
//{
//	cv_bridge::CvImagePtr cv_ptr;
//	try
//	{
//		cv_ptr = cv_bridge::toCvCopy(srcImg, sensor_msgs::image_encodings::BGR8);
//	}
//	catch (cv_bridge::Exception& e)
//	{
//		ROS_ERROR("%s",e.what());
//		return;
//	}	
//
//	curImg_ = cv_ptr->image; 
//
//}

/**
 * @brief 			use the vpMbEdgeTracker to initialize the tracker
 *
 * @param srcImg 	the sensor_msgs of ROS topic, it will be transfered to vpImage in the function
 */
//void 
//initializeTracker(const sensor_msgs::ImageConstPtr& srcImg)
//{
//	// the vpImg is a mono channel image
//	vpImage<uchar> vpImg;
//	// the operator:=() will allocate the memory automatically
//	vpImg = visp_bridge::toVispImage(*srcImg);
//
//	// We only use the tracker in visp to initialize the program
//	vpMbEdgeTracker initializer;
//	// NOTE:the params can also be retrieved from the sensor_msgs::CameraInfo using the visp_bridge package
//	initializer.loadConfigFile(config_file);		
//	initializer.getCameraParameters(cam);
//	initializer.loadModel(model_name);
//	initializer.setCovarianceComputation(true);
//
//	//initial the display
//	//these steps are not shown in the manual document
//	d.init(vpImg);
//
//	initializer.initClick(vpImg, init_file);
//	// obtain the initial pose of the model
//	cMo = initializer.getPose();
//	poseVector.buildFrom(cMo);
//
//	//clean up
//	vpXmlParser::cleanup();
//}

/**
 * @brief publish the processed Img or the tracking rst
 * 			TODO: modifications required.
 *
 * @param srcImg
 */
//void
//pubRst(const sensor_msgs::ImageConstPtr& srcImg)
//{
//	// Step 5: publish the tracking result
//
//	cv_bridge::CvImage out_msg;
//	out_msg.header = srcImg->header;
//	out_msg.encoding = sensor_msgs::image_encodings::BGR8;
//	out_msg.image = processedImg;
//
//	imgPub.publish(out_msg.toImageMsg());
//}
