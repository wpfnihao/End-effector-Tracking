/**
 * @file track_ros.h
 * @brief This class wraps all the tracker and ROS, so all the trackers can be implemented without notice that they are working with ROS.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-23
 */

//#include <ros/ros.h>
//#include <cv_bridge/cv_bridge.h> 
//#include <sensor_msgs/image_encodings.h> 
//#include <image_transport/image_transport.h>

//OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// for Kalman filter
#include <opencv2/video/tracking.hpp>

//visp headers
//the vpMbTracker class, used to get the initial pose
#include <visp/vpMbEdgeTracker.h>
//the vpPose classs, used to calc pose from features
//the vpHomogeneousMatrix.h is also included in this head file
#include <visp/vpPose.h>
#include <visp/vpImage.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpPoseFeatures.h>
#include "visp/vpDisplay.h"
#include "visp/vpDisplayX.h"
#include "visp/vpXmlParser.h"
// for conversion
#include "visp/vpMeterPixelConversion.h"

// for convert between sensor_msgs and visp image
//#include <conversions/image.h>

// std headers
#include <vector>
#include <iostream>
#include <cmath>

// user defined
//#include "endeffector_tracking/mbtEdgeTracker.h"
//#include "endeffector_tracking/kltFbTracker.h"
//#include "endeffector_tracking/superResolutionTracker.h"
#include "endeffector_tracking/keyFrameBasedTracker.h"
//#include "endeffector_tracking/mbtKltTracker.h"
//#include "endeffector_tracking/mbtEdgeKltTracker.h"

// enum, typedef, struct, class ...
	
	/**
	 * @brief The status of the current tracker
	 */
	enum tracking_status
	{
		TRACKING,
		LOST,
		INITIAL
	};

// member functions
			
	//member function 
	//the tracking callback function, which actually does the tracking procedure
	//void trackCallback(const sensor_msgs::ImageConstPtr& srcImg);	

	//convert from the ros image message to a CvImage
	//void msg2mat(const sensor_msgs::ImageConstPtr& srcImg, cv::Mat& curImg_);	

	/**
	 * @brief  	when initialize the tracker or the tracker lost the target
	 			call this function to (re)initialize it
	 *
	 * @param srcImg
	 */
	//void initializeTracker(const sensor_msgs::ImageConstPtr& srcImg);

	// tracking is done here
	void track(void);

	// publish the tracking result
	//void pubRst(const sensor_msgs::ImageConstPtr& srcImg);
