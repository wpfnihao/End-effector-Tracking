/**
 * @file activeModelTracker.cpp
 * @brief The active model based tracker. The shape prior is obtained from the 
 * 			3d CAD model.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-05
 */

// user defined
#include "endeffector_tracking/activeModelTracker.h"

void
activeModelTracker::track(void)
{
}

void 
activeModelTracker::deformSilhouette(vpHomogeneousMatrix& cMo_)
{
	// step 1: based on the pose and cad model, forward project the silhouette of the model onto the image plane.
	// step 1.1: find visible lines
	findVisibleLines(cMo_);	
	
	// step 1.2: generate the control points, based on the distance between corners
	controlPoints.clear();
	projectModel(cMo_, cam);

	for (int i = 0; i < 12; i++)
	{
		if (isVisible[i])
		{
			int p1, p2;
			line2Pts(i, p1, p2);
			vpPoint c1, c2;
			c1.set_x(prjCorners[p1].x);
			c1.set_y(prjCorners[p1].y);
			c1.setWorldCoordinates(corners[p1].x,corners[p1].y,corners[p1].z);
			c2.set_x(prjCorners[p2].x);
			c2.set_y(prjCorners[p2].y);
			c2.setWorldCoordinates(corners[p2].x,corners[p2].y,corners[p2].z);

			// save the corners first
			controlPoints[i].push_back(c1);
			controlPoints[i].push_back(c2);

			//then generate the control points
			genControlPoints(controlPoints[i]);
		}
	}
	
	// step 2: actively find the high gradient region
	//
	// step 3: based on the tentatively moved control points, estimate the pose
}

void 
activeModelTracker::initialize(const vpHomogeneousMatrix& cam_, const vpHomogeneousMatrix& cMo_, int rows_, int cols_)
{
	this->rows = rows_;
	this->cols = cols_;
	this->cMo = cMo_;
	this->cam = cam_;

	initModel();
}

inline void 
activeModelTracker::retrieveImage(const cv::Mat& img)
{
	// no copy
	// retrieved image has been preprocessed and should be gray scale
	curImg = img;
}

void 
activeModelTracker::pubRst(cv::Mat& img)
{
	plotRst();
	img = processedImg.clone();
}

void 
activeModelTracker::genControlPoints(std::vector<vpPoint>& controlPoints_)
{
	// the minimum distance between control points
	int minDist = 15;
	int dist = ptsDist(
			controlPoints_[0].get_x(), 
			controlPoints_[0].get_y(),
			controlPoints_[1].get_x(), 
			controlPoints_[1].get_y());
	int numOfPtsPerLine = (dist % minDist == 0) ? (dist / minDist -1) : (dist / minDist);

	// 2d
	double dist_x = (controlPoints_[1].get_x() - controlPoints_[0].get_x()) / (numOfPtsPerLine + 1);
	double dist_y = (controlPoints_[1].get_y() - controlPoints_[0].get_y()) / (numOfPtsPerLine + 1);

	// 3d
	double dist_X = (controlPoints_[1].get_oX() - controlPoints_[0].get_oX()) / (numOfPtsPerLine + 1);
	double dist_Y = (controlPoints_[1].get_oY() - controlPoints_[0].get_oY()) / (numOfPtsPerLine + 1);
	double dist_Z = (controlPoints_[1].get_oZ() - controlPoints_[0].get_oZ()) / (numOfPtsPerLine + 1);

	vpPoint vp;
	for (int i = 0; i < numOfPtsPerLine; i++)
	{
		vp.set_x(controlPoints_[0].get_x() + (i + 1) * dist_x);
		vp.set_y(controlPoints_[0].get_y() + (i + 1) * dist_y);

		vp.setWorldCoordinates(
				controlPoints_[0].get_oX() + (i + 1) * dist_X,
				controlPoints_[0].get_oY() + (i + 1) * dist_Y,
				controlPoints_[0].get_oZ() + (i + 1) * dist_Z);

		controlPoints_.push_back(vp);
	}
}

double 
activeModelTracker::lineSlope( cv::Point curPnt, cv::Point prePnt, cv::Point nxtPnt, bool& isHorizontal )
{
	isHorizontal = false;
	if (prePnt.y - nxtPnt.y == 0)
	{
		return 0;
	} 
	else if (prePnt.x - nxtPnt.x == 0)
	{
		isHorizontal = true;
		return 1;
	}
	else
	{
		if ((prePnt.x - nxtPnt.x) * (prePnt.y - nxtPnt.y) > 0)
		{
			return (double)abs(prePnt.y - nxtPnt.y) / abs(prePnt.x - nxtPnt.x);
		}
		else
		{
			return -(double)abs(prePnt.y - nxtPnt.y) / abs(prePnt.x - nxtPnt.x);
		}
	}
}
