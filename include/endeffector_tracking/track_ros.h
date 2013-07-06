/**
 * @file track_ros.h
 * @brief This class wraps all the tracker and ROS, so all the trackers can be implemented without notice that they are working with ROS.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-23
 */

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

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

// for convert between sensor_msgs and visp image
#include <conversions/image.h>

// std headers
#include <vector>
#include <iostream>
#include <cmath>

// user defined
#include "endeffector_tracking/KernelBasedTracking.h"
#include "endeffector_tracking/CamShiftTracking.h"
#include "endeffector_tracking/medianFlowTracking.h"
#include "endeffector_tracking/houghLineBasedTracker.h"
#include "endeffector_tracking/mbtEdgeTracker.h"
#include "endeffector_tracking/kltFbTracker.h"
#include "endeffector_tracking/activeModelTracker.h"
//#include "endeffector_tracking/mbtKltTracker.h"
//#include "endeffector_tracking/mbtEdgeKltTracker.h"

class CEndeffectorTracking
{
// enum, typedef, struct, class ...
public:
	
	/**
	 * @brief The status of the current tracker
	 */
	enum tracking_status
	{
		TRACKING,
		LOST,
		INITIAL
	};

//member variable
private:
	/**
	 * @brief The status of the current tracker
	 */
	tracking_status status;

	// different type of trackers
	KernelBasedTracking kbTracker;
	CamShiftTracking csTracker;
	medianFlowTracking mfTracker;
	houghLineBasedTracker hlTracker;
	mbtEdgeTracker meTracker;
	kltFbTracker fbTracker;
	//mbtKltTracker kltTracker;
	//mbtEdgeKltTracker mekltTracker;

	/* ROS params */
	ros::NodeHandle n;
	image_transport::Subscriber subCam;
	image_transport::Publisher imgPub;

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





	/* Global state */
	/**
	 * @brief  rows and cols
	 */
	int rows, cols;

	/**
	 * @brief  the camera currently used
	 */
	vpCameraParameters cam;

// member functions
public:
	// constructor
	// ros is inited in this function
	CEndeffectorTracking(int argc, char **argv);
			
	//member function 
	//the tracking callback function, which actually does the tracking procedure
	void trackCallback(const sensor_msgs::ImageConstPtr& srcImg);	

	//convert from the ros image message to a CvImage
	void msg2mat(const sensor_msgs::ImageConstPtr& srcImg, cv::Mat& curImg_);	

	/**
	 * @brief  	when initialize the tracker or the tracker lost the target
	 			call this function to (re)initialize it
	 *
	 * @param srcImg
	 */
	void initializeTracker(const sensor_msgs::ImageConstPtr& srcImg);

	// tracking is done here
	void track(void);

	// publish the tracking result
	void pubRst(const sensor_msgs::ImageConstPtr& srcImg);
};
