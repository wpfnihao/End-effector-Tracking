/* This class is written by wpf @ Singapore
 * 2013/5/9
 */
#include "endeffector_tracking/medianFlowTracking.h"

medianFlowTracking::medianFlowTracking()
:maxLevel(3)
,winSize(21)
{

}

void
medianFlowTracking::track(void)
{
	// maintain the features
	pFeatures = cFeatures;
	cFeatures.clear();
	pbox = cbox;
	
	// generate points for tracking
	std::vector<cv::Point2f> initFeatures;
	genInitFeatures(pbox, winSize, initFeatures);


	// forward tracking
	std::vector<unsigned char> fStatus;
	std::vector<float> fErr;
	cv::calcOpticalFlowPyrLK(pPyramid, cPyramid, initFeatures, cFeatures, fStatus, fErr, cv::Size(winSize, winSize), maxLevel);

	// backward tracking
	std::vector<unsigned char> bStatus;
	std::vector<float> bErr;
	std::vector<cv::Point2f> bFeatures;
	cv::calcOpticalFlowPyrLK(cPyramid, pPyramid, cFeatures, bFeatures, bStatus, bErr, cv::Size(winSize, winSize), maxLevel);

	// find out the features can be tracked forward and backward stably
	std::vector<cv::Point2f> stableFeatures;
	std::vector<cv::Point2f> motion;
	float th = 2;
	findStableFeatures(initFeatures, bFeatures, cFeatures, fStatus, bStatus, stableFeatures, motion, th);

	cFeatures = stableFeatures;
	std::cout<<"length"<<stableFeatures.size()<<std::endl;

	// update the tracking box
	findMotion(motion, dx, dy);
	cbox.x += int(dx);
	cbox.y += int(dy);
	std::cout<<"dx"<<dx<<std::endl;
	std::cout<<"dy"<<dy<<std::endl;
	// TODO:change of box size should be tackled with the help of pose estimator
}

void
medianFlowTracking::genInitFeatures(cv::Rect box, int size, std::vector<cv::Point2f>& features)
{
	int start = (size - 1) / 2;

	for (int i = start; i < box.height; i += size)
		for (int j = start; j < box.width; j += size)
			features.push_back(cv::Point2f(j + box.x, i + box.y));
}

void 
medianFlowTracking::findStableFeatures(
		const std::vector<cv::Point2f>& 	features1, 
		const std::vector<cv::Point2f>& 	features2, 
		std::vector<cv::Point2f>& 			cFeatures_, 
		const std::vector<unsigned char>& 	status1,
		const std::vector<unsigned char>& 	status2,
		std::vector<cv::Point2f>& 			outputFeatures,
		std::vector<cv::Point2f>& 			motion_,
		float 								th
		)
{
	// find the stable features
	for (size_t i = 0; i < features1.size(); i++)
	{
		if (status1[i] && status2[i])
		{
			float dx = features2[i].x - features1[i].x;
			float dy = features2[i].y - features1[i].y;
			float dist = std::sqrt(dx * dx + dy * dy);
			if (dist < th)
			{
				outputFeatures.push_back(cFeatures_[i]);
				dx = cFeatures_[i].x - features1[i].x;
				dy = cFeatures_[i].y - features1[i].y;
				motion_.push_back(cv::Point2f(dx, dy));
			}
		}
	}


	// check the features
	float ratio = outputFeatures.size() / features1.size();
	if (ratio < 0.1)
	{}
	else
	{}
}

void 
medianFlowTracking::findMotion(const std::vector<cv::Point2f>& motion_, float& dx_, float& dy_)
{
	dx_ = 0;
	dy_ = 0;
	for (size_t i = 0; i < motion_.size(); i++)
	{
		dx_ += motion_[i].x;
		dy_ += motion_[i].y;
	}

	dx_ /= motion_.size();
	dy_ /= motion_.size();
}

void
medianFlowTracking::init(const int *win_,const cv::Mat& img)
{
	cv::cvtColor(img, cImg, CV_BGR2GRAY);
	cv::buildOpticalFlowPyramid(cImg, cPyramid, cv::Size(winSize, winSize), maxLevel);

	cbox.x 		= win_[1];
	cbox.y 		= win_[0];
	cbox.width 	= win_[3] - win_[1];
	cbox.height = win_[2] - win_[0];
}

void
medianFlowTracking::retrieveImage(const cv::Mat& img)
{
	pImg = cImg.clone();
	//pPyramid = cPyramid;
	cv::buildOpticalFlowPyramid(pImg, pPyramid, cv::Size(winSize, winSize), maxLevel);
	cv::cvtColor(img, cImg, CV_BGR2GRAY);
	cv::buildOpticalFlowPyramid(cImg, cPyramid, cv::Size(winSize, winSize), maxLevel);
}

cv::Rect 
medianFlowTracking::getWindow(void)
{
	return this->cbox;
}
