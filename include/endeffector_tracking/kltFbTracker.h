/**
 * @file kltFbTracker.h
 * @brief the head file of the kltFbTracker.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-31
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
//the vpPose classes, used to calc pose from features
//the vpHomogeneousMatrix.h is also included in this head file
#include <visp/vpPose.h>
#include <visp/vpImage.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpPoseFeatures.h>

//#include <conversions/image.h>

// std headers
#include <vector>
#include <iostream>
#include <cmath>

// user defined 
#include "endeffector_tracking/cadModel.h"

class kltFbTracker: public cadModel
{
// enum, typedef, struct, class ...
public:
	
//member variable
private:

	/* current state */

	// pose here is either generated manually or measured by the connected hough lines

	/**
	 * @brief  	the current pose
	 */
	vpHomogeneousMatrix cMo;
	vpHomogeneousMatrix p_cMo;

	/**
	 * @brief  the pose vector corresponding to the pose matrix
	 * 			tx, ty, tz, thetaux, thetauy, thetauz
	 */
	vpPoseVector poseVector;

	/**
	 * @brief  current window for hough
	 *  p1 let up
	 *  p2 right bottom
	 *  window: x1, y1, x2, y2
	 */
	int window[4];

	/**
	 * @brief current cvImg maintained by the class
	 * 			TODO: the maintaining of the img
	 */
	cv::Mat curImg, preImg;
	std::vector<cv::Mat> pPyramid, cPyramid;
	cv::Mat processedImg;

	std::vector<vpPoint> stableFeatures3d;
	std::vector<cv::Point2f> stableFeatures2d;
	std::vector<float> fbError;
	std::vector<cv::Point2f> stableInitFeatures2d;

	int winSize;
	int maxLevel;

	/* End of current state */
	
	vpCameraParameters cam;

	/* Global state */

	/**
	 * @brief  rows and cols
	 * 			TODO: the maintaining of the params
	 *
	 */
	int rows, cols;
	
	/**
	 * @brief if tracking lost, this flag will be set true
	 * 			TODO: publish it to the outside 
	 */
	bool isLost;

	/* End of global state */



// member functions
public:
	/**
	 * @brief  constructor
	 */
	kltFbTracker();
			
	/**
	 * @brief  project the model based on the obtained cMo and init the Kalman filter.
	 *
	 * @param cam_ the camera param
	 * @param cMo_ the initial pose matrix
	 * @param pose the initial pose vector
	 * @param rows_ rows of image
	 * @param cols_ cols of image
	 */
	void initialize(const vpCameraParameters& cam_, const vpHomogeneousMatrix& cMo_, const vpPoseVector& pose, int rows_, int cols_);

	/**
	 * @brief  publish the tracked rst to the upper level
	 *
	 * @param img image with the tracker markers
	 * @param box tracked box
	 *
	 * @return whether the tracker is lost target
	 */
	bool pubRst(cv::Mat& img, cv::Rect& box);

	/**
	 * @brief 		retrieve Image from the up level class
	 *
	 * @param img 	the input image
	 */
	void retrieveImage(const cv::Mat& img);

	/**
	 * @brief  tracking is done here
	 */
	void track(void);

	/**
	 * @brief use the proposed error estimate method to refine the tracked pose
	 */
	void refinePose(void);

	void init(cv::Mat& img);


	/**
	 * @brief  measure pose from the connected features
	 *
	 * @param stableFeatures3d :the vpPoint 3d features
	 * @param stableFeatures2d :the cv::Point2f 2d features
	 */
	void measurePose(std::vector<vpPoint>& stableFeatures3d, std::vector<cv::Point2f>& stableFeatures2d, vpHomogeneousMatrix& poseMatrix);

	inline void getPose(vpHomogeneousMatrix& cMo_)
	{
		this->p_cMo = cMo_;
	}
	inline void pubPose(vpHomogeneousMatrix& cMo_)
	{
		cMo_ = this->cMo;
	}
// utility functions
protected:


	/**
	 * @brief  distance between two poseVector
	 *
	 * @param p1
	 * @param p2
	 *
	 * @return 
	 */
	inline double poseDist(const vpPoseVector& p1, const vpPoseVector& p2);

	/**
	 * @brief check whether the pose estimated is flipped
	 *
	 * @param p1
	 * @param p2
	 *
	 * @return 
	 */
	bool isFlipped(const vpPoseVector& p1, const vpPoseVector& p2);

	/**
	 * @brief generate features for tracking from the visible faces
	 *
	 * @param features :vector containing the features for tracking
	 * @param faceID   :the id of the visible face
	 */
	void genFeaturesForTrackOnFace(std::vector<vpPoint>& features, int faceID, int numOfPtsPerFace);

	/**
	 * @brief using the forward backward klt tracker to track the obtained features
	 *
	 * @param initFeatures2d
	 * @param bFeatures
	 * @param cFeatures
	 * @param fStatus
	 * @param bStatus
	 */
	void fbTracking(
			std::vector<cv::Point2f>& 	initFeatures2d, 
			std::vector<cv::Point2f>& 	bFeatures, 
			std::vector<cv::Point2f>& 	cFeatures, 
			std::vector<unsigned char>& fStatus, 
			std::vector<unsigned char>& bStatus
			);

	/**
	 * @brief find the stably tracked features from the forward backward tracker
	 *
	 * @param initFeatures
	 * @param bFeatures
	 * @param cFeatures
	 * @param fStatus
	 * @param bStatus
	 * @param th
	 */
	void findStableFeatures(
			std::vector<vpPoint>& 		initFeatures,
			std::vector<cv::Point2f>& 	initFeatures2d,
			std::vector<cv::Point2f>& 	bFeatures,
			std::vector<cv::Point2f>& 	cFeatures,
			std::vector<unsigned char>& fStatus, 
			std::vector<unsigned char>& bStatus,
			float 						th
			);

	/**
	 * @brief draw the tracked features and object for visual inspection
	 */
	void plotRst(void);
};
