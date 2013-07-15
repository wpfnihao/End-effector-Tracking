/**
 * @file textureTracker.cpp
 * @brief Core of the nonparametricTextureLearningTracker
 * 			The implementation of the texture based tracker
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-15
 */

#include "endeffector_tracking/textureTracker.h"

void 
textureTracker::initCoor(void)
{
	for (int i = 0; i < 6; i++)
	{
		initCoorOnFace(patchCoor[i], pyg[i], numOfPtsPerFace);
	}
}

void 
textureTracker::retrievePatch(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_, std::map<int, std::vector<cv::Mat> >& curPatch_)
{
	curPatch_.clear();
	this->cMo = cMo_;
	for (int i = 0; i < 6; i++)
	{
		if (pyg[i].isVisible(cMo_))
		{
			cv::Mat patch(numOfPtsPerFace, numOfPtsPerFace, CV_8UC1);
			for (int j = 0; j < numOfPtsPerFace; j++)
				for (int k = 0; k < numOfPtsPerFace; k++)
				{
					vpPoint P = patchCoor[i][j * numOfPtsPerFace + k];
					P.changeFrame(cMo_);
					P.project();
					double u, v;
					vpMeterPixelConversion::convertPoint(cam, P.get_x(), P.get_y(), u, v);
					patch.at<unsigned char>(j, k) = img.at<unsigned char>(v, u);
				}
			curPatch_[i].push_back(patch);
		}
	}
}

void
textureTracker::track(void)
{
	// optimized the pose based on the match of patches
}

void 
textureTracker::initCoorOnFace(std::vector<vpPoint>& features, vpMbtPolygon& pyg, int numOfPtsPerFace)
{
	const vpPoint& 	vp1 = pyg.p[0],
			 		vp2 = pyg.p[1],
			 		vp3 = pyg.p[2],
			 		vp4 = pyg.p[3];

	float step = 1.0 / (numOfPtsPerFace - 1);

	// avoid the edge
	for (int i = 0; i < numOfPtsPerFace; i++)
		for (int j = 0; j < numOfPtsPerFace; j++)
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
textureTracker::init(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_)
{
	// init the patch database
	retrievePatch(img, cMo_, cam_, patches);
}
