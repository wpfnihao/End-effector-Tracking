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
textureTracker::init()
{
	//model
	//pose
	//texture
}

void 
textureTracker::retrieveImage(const cv::Mat& img)
{}

void 
textureTracker::track(void)
{
	// edge map for edge detection
	// texture based track 
	// edge based tracker
	// merge
}

bool 
textureTracker::pubRst(cv::Mat& img, cv::Rect& box)
{}
