/**
 * @file nonparametricTextureLearningTracker.cpp
 * @brief The implementation of the non-parametric kernel density estimation based texture features learning tracker.
 * Try my best to make the tracker work and work well.
 * The method used here will be submitted to ICRA2014, no alternative.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-14
 */

#include "endeffector_tracking/nonparametricTextureLearningTracker.h"

void
nonparametricTextureLearningTracker::init(vpHomogeneousMatrix cMo_, vpCameraParameters cam_, std::string init_file)
{
	this->cMo = cMo_;
	this->cam = cam_;
	//model
	initModel();
	//pose
	//texture
	texTracker.getInitPoints(init_file);
	texTracker.init(curImg, cMo, cam);
}

void 
nonparametricTextureLearningTracker::track(void)
{
	// edge map for edge detection
	sobelGradient(curImg, gradient);

	// texture based track 
	texTracker.track(curImg, gradient);	
	texTracker.pubPose(cMo);

	// edge based tracker
	//
	
	// merge
	//
	
	// update
	texTracker.updatePatches(cMo);
}

bool 
nonparametricTextureLearningTracker::pubRst(cv::Mat& img, cv::Rect& box)
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

	img = processedImg.clone();

	return false;
}

void 
nonparametricTextureLearningTracker::sobelGradient(const cv::Mat& curImg, cv::Mat& gradient)
{
	cv::Mat sobel_x, sobel_y;
	cv::Sobel(curImg, sobel_x, CV_32F, 1, 0, 3);
	cv::convertScaleAbs(sobel_x, sobel_x);
	cv::Sobel(curImg, sobel_y, CV_32F, 0, 1, 3);
	cv::convertScaleAbs(sobel_y, sobel_y);
	cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, gradient);
	// debug only
	//cv::imshow("sobel", gradient);
	//cv::waitKey(30);
}
