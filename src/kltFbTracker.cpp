/**
 * @file kltFbTracker.cpp
 * @brief this file implements the KLT based forward backward tracker
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-31
 */

#include "endeffector_tracking/kltFbTracker.h"

#include "visp/vpDisplay.h"
#include "visp/vpDisplayX.h"

// for conversion
#include "visp/vpMeterPixelConversion.h"

//#define DEBUG

#ifdef DEBUG
#include "opencv2/highgui/highgui.hpp"
#endif

using namespace std;


kltFbTracker::kltFbTracker()
:isLost(false)
,winSize(21)
,maxLevel(3)
{
}

void 
kltFbTracker::initialize(const vpCameraParameters& cam_, const vpHomogeneousMatrix& cMo_, const vpPoseVector& pose, int rows_, int cols_)
{

	this->rows = rows_;
	this->cols = cols_;
	this->cMo = cMo_;
	this->poseVector = pose;
	this->cam = cam_;
	// project the model 
	// both the lines and the corners are projected
	projectModel(cMo, cam);
}

void 
kltFbTracker::track(void)
{
	cMo = p_cMo;

	isLost = false;
	// Step 1: find visible faces and generate points on the visible faces for klt tracking
		
	// features generated from visible faces for tracking
	std::vector<vpPoint> initFeatures;
	std::vector<cv::Point2f> initFeatures2d;

	std::vector<cv::Point2f> bFeatures;
	std::vector<cv::Point2f> cFeatures;
	std::vector<unsigned char> fStatus;
	std::vector<unsigned char> bStatus;


	// number of points used to track per face = (numOfPtsPerFace +1) * (numOfPtsPerFace + 1);
	int numOfPtsPerFace = 10;
	bool isVisibleFace[6];
	for (int i = 0; i < 6; i++)
	{
		isVisibleFace[i] = pyg[i].isVisible(cMo);
		if (isVisibleFace[i])
		{
			genFeaturesForTrackOnFace(initFeatures, i, numOfPtsPerFace);
		}
	}

	// Step 2: project the points
	for (size_t i = 0; i < initFeatures.size(); i++)
	{
		initFeatures[i].changeFrame(cMo);
		initFeatures[i].project();
		// use the camera intrinsic parameter to get the coordinate in image plane in pixel
		double u, v;
		vpMeterPixelConversion::convertPoint(cam, initFeatures[i].get_x(), initFeatures[i].get_y(), u, v);
		initFeatures2d.push_back(cv::Point2f(u, v)); 
	}


	// Step 3: forward backward tracking
	fbTracking(initFeatures2d, bFeatures, cFeatures, fStatus, bStatus);
		

	// Step 4: find the correspondence between 2d and 3d points
	float th = 0.5;
	findStableFeatures(initFeatures, initFeatures2d, bFeatures, cFeatures, fStatus, bStatus, th);	
	
	// Step 5: measure the new pose
	measurePose(stableFeatures3d, stableFeatures2d, cMo);

	// Step 6: do some checks (optional)
	
}

void
kltFbTracker::measurePose(std::vector<vpPoint>& stableFeatures3d, std::vector<cv::Point2f>& stableFeatures2d, vpHomogeneousMatrix& poseMatrix)
{
	vpPoseFeatures pose;

	if (stableFeatures3d.size() < 6)
	{
		std::cout<<"too few features!"<<stableFeatures3d.size()<<std::endl;
		isLost = true;
		return;
	}

	// Point Features
	for (size_t i = 0; i < stableFeatures3d.size(); i++)
	{
		// image frame to image plane
		double x, y;
		vpPixelMeterConversion::convertPoint(cam, stableFeatures2d[i].x, stableFeatures2d[i].y, x, y);
		stableFeatures3d[i].set_x(x);
		stableFeatures3d[i].set_y(y);
		pose.addFeaturePoint(stableFeatures3d[i]);
	}

	// get the pose from features
	pose.setLambda(0.6);
	// the initial value of the pose is the previous pose

	try
	{
		//std::cout<<"start computePose"<<std::endl;
		// TODO: poor feature may still force the program to quit without throwing any Exception
		pose.computePose(poseMatrix);
		poseVector.buildFrom(poseMatrix);
		//std::cout<<"complete computePose"<<std::endl;

		//std::cout<<"pose vector = "<<std::endl;
		//std::cout<<poseVector.t()<<std::endl;
		
	}
	catch(...) // catch all kinds of Exceptions
	{
		std::cout<<"Exception raised in measurePose"<<std::endl;

		// TODO: add the tracking lost process codes here
	}
}

inline double 
kltFbTracker::ptsDist(double x1, double y1, double x2, double y2)
{
	return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

inline double 
kltFbTracker::poseDist(const vpPoseVector& p1, const vpPoseVector& p2)
{
	return std::sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) 
			+ (p1[1] - p2[1]) * (p1[1] - p2[1]) 
			+ (p1[2] - p2[2]) * (p1[2] - p2[2])
			+ (p1[3] - p2[3]) * (p1[3] - p2[3])
			+ (p1[4] - p2[4]) * (p1[4] - p2[4])
			+ (p1[5] - p2[5]) * (p1[5] - p2[5])
			);
}

bool
kltFbTracker::isFlipped(const vpPoseVector& p1, const vpPoseVector& p2)
{
	using namespace std;
	bool a = abs(p1[0]) - abs(p2[0]) < 0.5;
	bool b = abs(p1[1]) - abs(p2[1]) < 0.5;
	bool c = abs(p1[2]) - abs(p2[2]) < 0.5;
	bool d = abs(p1[3]) - abs(p2[3]) < 0.5;
	bool e = abs(p1[4]) - abs(p2[4]) < 0.5;
	bool f = abs(p1[5]) - abs(p2[5]) < 0.5;
	bool g = abs(p1[3] - p2[3]) > 0.1;
	bool h = abs(p1[4] - p2[4]) > 0.1;
	bool i = abs(p1[5] - p2[5]) > 0.1;
	return (a & b & c & d & e & f & g & h & i);
}

void
kltFbTracker::retrieveImage(const cv::Mat& img)
{
	preImg = curImg.clone();
	cv::buildOpticalFlowPyramid(preImg, pPyramid, cv::Size(winSize, winSize), maxLevel);
	cv::cvtColor(img, curImg, CV_BGR2GRAY);
	cv::buildOpticalFlowPyramid(curImg, cPyramid, cv::Size(winSize, winSize), maxLevel);

	processedImg = img.clone();
}

bool
kltFbTracker::pubRst(cv::Mat& img, cv::Rect& box)
{
	plotRst();
	img = processedImg.clone();

	box.x 		= window[1];
	box.y 		= window[0];
	box.height 	= window[2] - window[0];
	box.width 	= window[3] - window[1];

	return this->isLost;
}

void 
kltFbTracker::genFeaturesForTrackOnFace(std::vector<vpPoint>& features, int faceID, int numOfPtsPerFace)
{
	const vpPoint& 	vp1 = pyg[faceID].p[0],
			 		vp2 = pyg[faceID].p[1],
			 		vp3 = pyg[faceID].p[2],
			 		vp4 = pyg[faceID].p[3];

	float step = 1.0 / numOfPtsPerFace;

	// avoid the edge
	for (int i = 1; i < numOfPtsPerFace; i++)
		for (int j = 1; j < numOfPtsPerFace; j++)
		{
			vpPoint fp;
			float alpha = i * step;
			float beta  = j * step;
			fp.set_oX((alpha + beta - 1) * vp2.get_oX() + (1 - alpha) * vp1.get_oX() + (1 - beta) * vp3.get_oX());
			fp.set_oY((alpha + beta - 1) * vp2.get_oY() + (1 - alpha) * vp1.get_oY() + (1 - beta) * vp3.get_oY());
			fp.set_oZ((alpha + beta - 1) * vp2.get_oZ() + (1 - alpha) * vp1.get_oZ() + (1 - beta) * vp3.get_oZ());

			features.push_back(fp);
		}
}

void 
kltFbTracker::fbTracking(
		std::vector<cv::Point2f>& 	initFeatures2d, 
		std::vector<cv::Point2f>& 	bFeatures, 
		std::vector<cv::Point2f>& 	cFeatures, 
		std::vector<unsigned char>& fStatus, 
		std::vector<unsigned char>& bStatus
		)
{

	// forward tracking
	std::vector<float> fErr;
	cv::calcOpticalFlowPyrLK(pPyramid, cPyramid, initFeatures2d, cFeatures, fStatus, fErr, cv::Size(winSize, winSize), maxLevel);

	// backward tracking
	std::vector<float> bErr;
	cv::calcOpticalFlowPyrLK(cPyramid, pPyramid, cFeatures, bFeatures, bStatus, bErr, cv::Size(winSize, winSize), maxLevel);
}

void 
kltFbTracker::findStableFeatures(
		std::vector<vpPoint>& 		initFeatures, 
		std::vector<cv::Point2f>& 	initFeatures2d, 
		std::vector<cv::Point2f>& 	bFeatures,
		std::vector<cv::Point2f>& 	cFeatures,
		std::vector<unsigned char>& fStatus, 
		std::vector<unsigned char>& bStatus,
		float 						th
		)
{
	fbError.clear();
	stableInitFeatures2d.clear();
	stableFeatures3d.clear();
	stableFeatures2d.clear();

	for (size_t i = 0; i < initFeatures.size(); i++)
	{
		if (fStatus[i] && bStatus[i])
		{
			float dx = initFeatures2d[i].x - bFeatures[i].x;
			float dy = initFeatures2d[i].y - bFeatures[i].y;
			float dist = std::sqrt(dx * dx + dy * dy);
			if (dist < th)
			{
				stableFeatures2d.push_back(cFeatures[i]);
				stableFeatures3d.push_back(initFeatures[i]);
				fbError.push_back(dist);
				stableInitFeatures2d.push_back(initFeatures2d[i]);
			}
		}
	}
}

void 
kltFbTracker::plotRst(void)
{
	projectModel(cMo, cam);

	// tracked corners
	for (size_t i = 0; i < prjCorners.size(); i++)
		cv::circle(processedImg, prjCorners[i], 3, cv::Scalar(255, 0, 0));

	// lines of the cube
	for (int i = 0; i < 12; i++)
	{
		int p1, p2;
		line2Pts(i, p1, p2);
		cv::line(processedImg, prjCorners[p1], prjCorners[p2], cv::Scalar(255, 0, 0), 2);
	}

	// tracked features
	for (size_t i = 0; i < stableFeatures2d.size(); i++)
		cv::circle(processedImg, stableFeatures2d[i], 3, cv::Scalar(0, 0, 255));
}

void
kltFbTracker::init(cv::Mat& img)
{
	// init the model
	initModel();

	cv::cvtColor(img, curImg, CV_BGR2GRAY);
	cv::buildOpticalFlowPyramid(curImg, cPyramid, cv::Size(winSize, winSize), maxLevel);
}

void
kltFbTracker::line2Pts(const int lineID, int& p1, int& p2)
{
	if (lineID < 4)
	{
		// four lines of the upper face
		p1 = lineID;
		p2 = (lineID + 1) % 4;
	}
	else if (lineID >= 4 && lineID < 8)
	{
		// four lines of the bottom face
		p1 = lineID;
		if (lineID == 7)
			p2 = 4;
		else
			p2 = lineID + 1;
	}
	else // 8 - 11
	{
		// four vertical lines
		p1 = lineID - 8;
		p2 = lineID - 4;
	}
}

void
kltFbTracker::refinePose(void)
{
	// step 1: project the stableFeatures3d
	std::vector<cv::Point2f> projFeatures2d;

	for (size_t i = 0; i < stableFeatures3d.size(); i++)
	{
		stableFeatures3d[i].changeFrame(cMo);
		stableFeatures3d[i].project();
		// use the camera intrinsic parameter to get the coordinate in image plane in pixel
		double u, v;
		vpMeterPixelConversion::convertPoint(cam, stableFeatures3d[i].get_x(), stableFeatures3d[i].get_y(), u, v);
		projFeatures2d.push_back(cv::Point2f(u, v)); 
	}

	// step 2: measure the error of stableFeatures2d
	// e = stableFeatures2d - projFeatures2d
	std::vector<cv::Point2f> featureError;	
	for (size_t i = 0; i < stableFeatures2d.size(); i++)
	{
		featureError.push_back(cv::Point2f(stableFeatures2d[i].x - projFeatures2d[i].x, stableFeatures2d[i].y - projFeatures2d[i].y));
	}

	// step 3: measure the weight of tracking error
	// first normalize the error: map to [0,1]
	// TODO: add a parameter to control the interval [low, high]
	float up = 0.5;
	std::vector<float>::iterator maxe;
	std::vector<float>::iterator mine;
	maxe = std::max_element(fbError.begin(), fbError.end());
	mine = std::min_element(fbError.begin(), fbError.end());
	float mi = *mine;
	float ma = *maxe - *mine;
	for (size_t i = 0; i < fbError.size(); i++)
		fbError[i] = (fbError[i] - mi) / ma * up;

	// step 4: extract the matching error and update the stableFeatures2d
	// TODO: extract the tracking error and refine the tracking procedure
	for (size_t i = 0; i < fbError.size(); i++)
	{
		float x = (1 - fbError[i]) * featureError[i].x;
		float y = (1 - fbError[i]) * featureError[i].y;
		stableInitFeatures2d[i].x = stableInitFeatures2d[i].x - x;
		stableInitFeatures2d[i].y = stableInitFeatures2d[i].y - y;
	}	

	// step 5: refine the previous pose 
	measurePose(stableFeatures3d, stableInitFeatures2d, p_cMo);
}
