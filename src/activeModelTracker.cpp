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

#include "opencv2/highgui/highgui.hpp"

void
activeModelTracker::track(void)
{
	sobelGradient(curImg, gradient);
	for (int i = 0; i < 10; i++)
	{
		deformSilhouette(cMo);
	}
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
	// step 2.1: for each line, get its slope
	// step 2.2: for each control point on the line, find the position with the highest gradient magnitude
	//
	for (int i = 0; i < 12; i++)
	{
		if (isVisible[i])
		{
			cv::Point p1(controlPoints[i][0].get_x(), controlPoints[i][0].get_y());
			cv::Point p2(controlPoints[i][1].get_x(), controlPoints[i][1].get_y());
			bool isHorizontal;
			double step = lineSlope(p1, p2, isHorizontal);

			int detectionRange = 5;
			deformLine(step, isHorizontal, controlPoints[i], gradient, rows, cols, detectionRange);
		}
	}

	// step 3: based on the tentatively moved control points, estimate the pose
	//
	vpPoseFeatures pose;
	for (int i = 0; i < 12; i++)
	{
		if (isVisible[i])
		{
			for (size_t j = 0; j < controlPoints[i].size(); j++)
			{
				// debug only
				// tracked features
				// TODO: this will plot all the deformation step of the tracker
				cv::circle(processedImg, cv::Point(controlPoints[i][j].get_x(), controlPoints[i][j].get_y()), 3, cv::Scalar(0, 0, 255));

				double x, y;
				vpPixelMeterConversion::convertPoint(cam, controlPoints[i][j].get_x(), controlPoints[i][j].get_y(), x, y);
				controlPoints[i][j].set_x(x);
				controlPoints[i][j].set_y(y);
				pose.addFeaturePoint(controlPoints[i][j]);
			}
		}
	}	
	pose.setLambda(0.6);
	try
	{
		pose.computePose(cMo_);
	}
	catch(...) // catch all kinds of Exceptions
	{
		std::cout<<"Exception raised in computing pose"<<std::endl;
	}
}

void 
activeModelTracker::initialize(const vpCameraParameters& cam_, const vpHomogeneousMatrix& cMo_, int rows_, int cols_)
{
	this->rows = rows_;
	this->cols = cols_;
	this->cMo = cMo_;
	this->cam = cam_;

	initModel();
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
activeModelTracker::lineSlope( cv::Point prePnt, cv::Point nxtPnt, bool& isHorizontal )
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
		return ((double)(prePnt.y - nxtPnt.y)) / (prePnt.x - nxtPnt.x);
		/*
		if ((prePnt.x - nxtPnt.x) * (prePnt.y - nxtPnt.y) > 0)
		{
			return (double)abs(prePnt.y - nxtPnt.y) / abs(prePnt.x - nxtPnt.x);
		}
		else
		{
			return -(double)abs(prePnt.y - nxtPnt.y) / abs(prePnt.x - nxtPnt.x);
		}
		*/
	}
}

void 
activeModelTracker::deformLine(double step, bool isHorizontal, std::vector<vpPoint>& controlPoints_, const cv::Mat& gradient, int rows, int cols, int detectionRange)
{
	// for each point on the line
	for (size_t i = 0; i < controlPoints_.size(); i++)
	{
		int x = controlPoints_[i].get_x();
		int y = controlPoints_[i].get_y();
		int cx = x, 
			cy = y;
		int tmpx = x, 
			tmpy = y;
		unsigned char maxGradient = gradient.at<unsigned char>(cy, cx);
		for(int j = -detectionRange; j <= detectionRange; j++)
		{
			// the normal of the line
			// two bugs are fixed here
			if (isHorizontal)
			{
				cx = x + j;
				cy = y;
			}
			else if (abs(step) < 1)
			{
				cy = y + j;
				cx = x - step * j;
			}
			else
			{
				cx = x + j;
				// a bug fixed here
				cy = y - 1 / step * j;
			}

			// check whether the pixel is in the image 
			checkIndex(cx, cy, rows, cols);

			if (gradient.at<unsigned char>(cy, cx) > maxGradient)
			{
				maxGradient = gradient.at<unsigned char>(cy, cx);
				tmpx = cx;
				tmpy = cy;
			}
		}
		controlPoints_[i].set_x(tmpx);
		controlPoints_[i].set_y(tmpy);
	}
}

void 
activeModelTracker::sobelGradient(const cv::Mat& curImg, cv::Mat& gradient)
{
	cv::Mat sobel_x, sobel_y;
	cv::Mat blurImg;
	cv::GaussianBlur(curImg, blurImg, cv::Size(5, 5), 3, 3);
	cv::Sobel(blurImg, sobel_x, CV_32F, 1, 0, 3);
	cv::convertScaleAbs(sobel_x, sobel_x);
	cv::Sobel(blurImg, sobel_y, CV_32F, 0, 1, 3);
	cv::convertScaleAbs(sobel_y, sobel_y);
	cv::addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, gradient);
	// debug only
	cv::imshow("sobel", gradient);
	cv::waitKey(30);
}

void 
activeModelTracker::plotRst(void)
{}
