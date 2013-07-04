/**
 * @file houghLineBasedTracker.cpp
 * @brief The basic houghLineBasedTracker. This is the first workable version using the Hough line detection + Kalman filter + VVS pose estimator. It can only work in very limited environments.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-05-23
 */

#include "endeffector_tracking/houghLineBasedTracker.h"

#include "visp/vpDisplay.h"
#include "visp/vpDisplayX.h"

// for conversion
#include "visp/vpMeterPixelConversion.h"

//#define DEBUG

#ifdef DEBUG
#include "opencv2/highgui/highgui.hpp"
#endif

using namespace std;


/**
 * @brief 	constructor
 *
 * @param argc
 * @param argv
 */
houghLineBasedTracker::houghLineBasedTracker()
:KF(12, 6)
,isLost(false)
{
	//param initialization for those params can't be initialized in the constructor (not the assignment in the body like the following)
	for (int i = 0; i < 12; i++)
		isVisible[i] = false;	
}

/**
 * @brief hough line detection
 *
 * @param win
 */
void 
houghLineBasedTracker::houghLineDetection(const int* win)
{
	// Step 1: generate the image patch
	cv::Mat cannyDst, graySrc;	
	cv::cvtColor(curImg, graySrc, CV_BGR2GRAY);

	// create the image patch
	int winEnlarge = 20;
	// l:low h:high r:row c:column
	int lr = max(win[0]-winEnlarge, 0);
	int hr = min(win[2]+winEnlarge, rows);
	int lc = max(win[1]-winEnlarge, 0);
	int hc = min(win[3]+winEnlarge, cols);

	// cv::Range(inclusive, exclusive)
	cv::Mat imgPatch = graySrc(cv::Range(lr, hr), cv::Range(lc, hc));

	// Step 2: hough Line detection only within the patch 
	// TODO:may need to tune the param here
	// TODO:canny can also be done only in the window
	cv::Canny(imgPatch, cannyDst, 30, 50, 3);

	// tmp lines
	std::vector<cv::Vec4i> tmpLines;
	cv::HoughLinesP(cannyDst, tmpLines, 1, CV_PI/180, 50, 30, 10);

	// Step 3: transform the lines got into the global coordinate system
	//
	// we should first clear the lines before we push_back newly obtained ones
	// won't reallocate memory, so it's very efficiency
	lines.clear();

	// then save the new ones
	std::vector<cv::Vec4i>::const_iterator itr;
	for (itr = tmpLines.begin(); itr != tmpLines.end(); ++itr)
	{
		float x1 = (*itr)[0] + win[1] - winEnlarge;
		float x2 = (*itr)[1] + win[0] - winEnlarge;
		float x3 = (*itr)[2] + win[1] - winEnlarge;
		float x4 = (*itr)[3] + win[0] - winEnlarge;

		lines.push_back(cv::Vec4f(x1, x2, x3, x4));
	}

#ifdef DEBUG 
	cv::Mat cDst;
	cv::cvtColor(cannyDst, cDst, CV_GRAY2BGR);
	processedImg = cDst; 
#endif
}

/**
 * @brief 
 *
 * @param srcImg
 */
void 
houghLineBasedTracker::initialize(const vpCameraParameters& cam_, const vpHomogeneousMatrix& cMo_, const vpPoseVector& pose, int rows_, int cols_)
{

	this->rows = rows_;
	this->cols = cols_;
	this->cMo = cMo_;
	this->poseVector = pose;
	this->cam = cam_;
	// project the model 
	// both the lines and the corners are projected
	projectModel(cMo);

	// initialize the Kalman filter to track and predict the pose (cMo) of the pen (endeffector)
	initKalman();
}

/**
 * @brief The main tracking function
 */
void 
houghLineBasedTracker::track(void)
{
	isLost = false;
#ifndef DEBUG
	//test only
	//draw the rst
	processedImg = curImg.clone();
#endif

	// Step 1: predict the target pose using Kalman filter
	cv::Mat predict = KF.predict();
	for (int i = 0; i < 6; i++)
	{
		// the default matrix type in Kalman filter is CV_32F 
		p_Pose[i] = predict.at<float>(i, 0);
	}
	// convert from pose vector to pose matrix (homogeneous matrix)
	p_cMo.buildFrom(p_Pose);
	//p_cMo.buildFrom(poseVector);

	// then project the model
	projectModel(p_cMo);
	//projectModel(cMo);

	// use the projected model to generate a window, then hough transform can be done only in the window, which will lead to both efficiency and accuracy
	genTrackingWindow();

	// Step 2: use hough transform to detect lines in the predicted window
	houghLineDetection(window);

	// Step 3: use the lines detected to track the cube
	//
	// 3.1 find which line is visible under the >>>>predicted pose<<<<
	// 		using the vpMbtPolygon class
	findVisibleLines(p_cMo);	
	//findVisibleLines(cMo);	
	
	// 3.2 link the detected line with the visible line
	makeConnection();
	
	// 3.3 compute the pose
	measurePose();
	
	// Step 4: correct the prediction made by Kalman Filter
	cv::Mat measurement(6, 1, CV_32FC1);
	for (int i = 0; i < 6; i++)
		measurement.at<float>(i, 0) = poseVector[i];

	KF.correct(measurement);
	
}

 /**
  * @brief load the cad model and do some initialization
  *
  * @param initP
  */
void 
houghLineBasedTracker::initModel(std::vector<cv::Point3f>& initP)
{
	// Step 1: init the corners
	//
	// the corners are arrange by two faces anti-clockwise
	// the first three ones have the same order with the init points
	// I've checked the points here, seems correct
	corners.push_back(initP[0]);
	corners.push_back(initP[1]);
	corners.push_back(initP[2]);

	// P4
	cv::Point3f P4(initP[0].x, initP[0].y, initP[2].z);
	corners.push_back(P4);

	// only y-axis is different from the previous points
	// P5
	cv::Point3f P5(initP[0].x, initP[3].y, initP[0].z);
	corners.push_back(P5);

	// P6
	corners.push_back(initP[3]);

	// P7
	cv::Point3f P7(initP[2].x, initP[3].y, initP[2].z);
	corners.push_back(P7);

	// P8
	cv::Point3f P8(initP[0].x, initP[3].y, initP[2].z);
	corners.push_back(P8);

	// Step 2: init the lines of the cube from the points
	for (int i = 0; i < 12; i++)
	{
		int p1, p2;
		line2Pts(i, p1, p2);

		houghLineBasedTracker::modelLine L;

		L.push_back(corners[p1]);
		L.push_back(corners[p2]);

		model.push_back(L);
	}

	// Step 3: init the polygons from the corners
	//
	// six faces
	// FIXME:the corners are added according to right hand coordinate system
	// with normal of the face points from inside to the outside
	// check whether it is the case in visp
	//
	//
	//
	// VERY IMPORTANT:
	// new operator used in vpMbtPolygon class and operator=() is NOT redefined in the class, which means the vpMbtPolygon instants should be instantiated in the constructor of this class, if not, the data will NOT be properly copied during push_back and the whole program will crash.
	
	// each face is a rectangle
	//
	// face 1
	// corner order: 0, 1, 2, 3 
	pyg[0].setNbPoint(4);
	for (int i = 0; i < 4; i++)
	{
		vpPoint p;
		p.setWorldCoordinates(corners[i].x,corners[i].y,corners[i].z);
		pyg[0].addPoint(i, p);
	}

	// face 2
	// corner order 4, 7, 6, 5
	pyg[1].setNbPoint(4);
	for (int i = 0; i < 4; i++)
	{
		vpPoint p;
		int idx = 4 + (4 - i) % 4;
		p.setWorldCoordinates(corners[idx].x,corners[idx].y,corners[idx].z);
		pyg[1].addPoint(i, p);
	}

	// face 3 - 6
	// corner order 
	// 3: 0, 4, 5, 1
	// 4: 1, 5, 6, 2
	// 5: 2, 6, 7, 3
	// 6: 3, 7, 4, 0
	for (int i = 0; i < 4; i++)
	{
		pyg[i + 2].setNbPoint(4);
		for (int j = 0; j < 4; j++)
		{
			int idx;

			// get the corner index
			if (j == 0)
				idx =  i;
			else if (j == 1)
				idx = 4 + i;
			else if (j == 2)
				idx = 4 + (i + 1) % 4;
			else if (j == 3)
				idx = (i + 1) % 4;

			vpPoint p;
			p.setWorldCoordinates(corners[idx].x, corners[idx].y, corners[idx].z);
			pyg[i + 2].addPoint(j, p);
		}
	}
} // end of initModel

/**
 * @brief use the camera pose to project the model
 *
 * @param cMo_
 */
void 
houghLineBasedTracker::projectModel(const vpHomogeneousMatrix& cMo_)
{
	// do some clearing
	prjModel.clear();
	prjCorners.clear();
	
	// Step 1: project the lines
	std::vector<houghLineBasedTracker::modelLine>::const_iterator line; 

#ifdef DEBUG
	// evaluate the projected model
	cv::Mat box = cv::Mat::zeros(this->rows, this->cols, CV_8UC3);
#endif

	for (line = model.begin(); line != model.end(); ++line)
	{
		// the projected line for further processing
		cv::Vec4f L;
		
		// start to convert
		//
		// convert cv::Point to vpPoint
		vpPoint P[2];

		P[0].setWorldCoordinates((*line)[0].x, (*line)[0].y, (*line)[0].z);
		P[1].setWorldCoordinates((*line)[1].x, (*line)[1].y, (*line)[1].z);
		
		// project the point
		P[0].changeFrame(cMo_);
		P[1].changeFrame(cMo_);
		P[0].project();	
		P[1].project();	

		// use the camera intrinsic parameter to get the coordinate in image plane in pixel
		double u, v;
		vpMeterPixelConversion::convertPoint(cam, P[0].get_x(), P[0].get_y(), u, v);
		L[0] = u; 
		L[1] = v; 

		vpMeterPixelConversion::convertPoint(cam, P[1].get_x(), P[1].get_y(), u, v);
		L[2] = u; 
		L[3] = v; 

#ifdef DEBUG
	// evaluate the projected model
	//cv::line(box, cv::Point(L[0], L[1]), cv::Point(L[2], L[3]), cv::Scalar(0, 0, 255), 1, CV_AA);
#endif
		// save the lines
		prjModel.push_back(L);
	}

#ifdef DEBUG
	// evaluate the projected model
	//std::cout<<"start to plot"<<std::endl;
	//cv::imshow("projectModel", box);
	//cv::waitKey();
	//cv::destroyWindow("projectModel");
	//std::cout<<"finish plot"<<std::endl;
#endif

	// Step 2: project the corners
	std::vector<cv::Point3f>::const_iterator corner;
	for (corner = corners.begin(); corner != corners.end(); ++corner)
	{
		// convert cv::Point to vpPoint
		vpPoint P;
		P.setWorldCoordinates((*corner).x, (*corner).y, (*corner).z); 
		// project the point
		P.changeFrame(cMo_);
		P.project();
		// use the camera intrinsic parameter to get the coordinate in image plane in pixel
		double u, v;
		vpMeterPixelConversion::convertPoint(cam, P.get_x(), P.get_y(), u, v);
		prjCorners.push_back(cv::Point2f(u, v)); 
	}
}

void 
houghLineBasedTracker::initKalman(void)
{
	// initialize the state transition matrix using PV model
	// PV model: 	state = state + deltaState;
	// 				deltaState = deltaState;
	//
	// state transition Matrix
	cv::Mat st(12, 12, CV_32FC1);
	cv::setIdentity(st);
	for (int i = 0; i < 6; i++)
	{
		st.at<float>(i, i + 6) = 1;
	}
	KF.transitionMatrix = st.clone();

	// initialize other matrices
	// TODO:should I specify the params here?
	cv::setIdentity(KF.measurementMatrix);
	cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(8e-2));
	cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(5e-1));
	cv::setIdentity(KF.errorCovPost, cv::Scalar::all(1));

	// use the initial pose as the statePost
	cv::Mat sp(12, 1, CV_32FC1);
	for (int i = 0; i < 6; i++)
	{
		sp.at<float>(i, 0) = poseVector[i];
	}
	// the initial velocity is set to zero
	for (int i = 6; i < 12; i++)
	{
		sp.at<float>(i, 0) = 0;
	}
	KF.statePost = sp.clone();
}

void 
houghLineBasedTracker::genTrackingWindow(void)
{
	int left 	= this->cols - 1, 
		right 	= 0, 
		up 		= this->rows - 1, 
		bottom 	= 0;
	// can't use the STL max_element here
	std::vector<cv::Point2f>::const_iterator corner;
	for (corner = prjCorners.begin(); corner != prjCorners.end(); ++corner)
	{
		if ((*corner).y > bottom)
			bottom = (*corner).y; 
		if ((*corner).y < up)
			up = (*corner).y; 
		if ((*corner).x < left)
			left = (*corner).x; 
		if ((*corner).x > right)
			right = (*corner).x; 
	}
	this->window[0] = up;
	this->window[1] = left;
	this->window[2] = bottom;
	this->window[3] = right;

	std::cout<<up<<"..."<<left<<"..."<<bottom<<"..."<<right<<std::endl;
}

void 
houghLineBasedTracker::findVisibleLines(const vpHomogeneousMatrix& cMo_)
{
	// find visible polygons
	//
	// six faces (polygons)
	bool isVisibleFace[6];
	for (int i = 0; i < 6; i++)
	{
		isVisibleFace[i] = pyg[i].isVisible(cMo_);
	}

	for (int i = 0; i < 12; i++)
	{
		int f1, f2;
		line2face(i, f1, f2);
		if (isVisibleFace[f1] || isVisibleFace[f2])
			this->isVisible[i] = true;
		else
			this->isVisible[i] = false;

		// tmp 
		//this->isVisible[i] = true;
	}

}

void
houghLineBasedTracker::transLineType(const cv::Vec4f& line_, vpLine& sl)
{
	// only unwrap the namespace in this function
	// for so many math function in STL used here
	using namespace std;

	// use two points to get the line equation
	// ax + by + c = 0
	double a, b, c;
	
	getPrjLineEqu(line_, a, b, c);

	// see "visp tutorial geometric objects" pp. 29
	double rho, theta;
	rho = - c / sqrt(a * a + b * b); 
	if (a == 0)
		theta = 2 * atan(1);
	else
		theta = atan(b / a);

	sl.setRho(rho);
	sl.setTheta(theta);
}

void
houghLineBasedTracker::makeConnection(void)
{
	// clear the vector first
	detectedModel.clear();
	detectedCorners.clear();

	// transform the line type of the detected hough lines
	std::vector<vpLine> slmodel;
	std::vector<cv::Vec4f>::const_iterator itr;
	for (itr = lines.begin(); itr != lines.end(); ++itr)
	{
		vpLine vl;
		transLineType(*itr, vl);
		slmodel.push_back(vl);
	}


#ifdef DEBUG
	// check the line connection
	cv::Mat connection = this->curImg.clone(); 
	cv::imshow("connection", connection);
	//for (size_t i = 0; i < lines.size(); i++)	
	//cv::line(connection, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 255, 0), 1, CV_AA);
#endif

	double distTH = 20; 	 // TODO: tune this param
	for (size_t i = 0; i < prjModel.size(); i++)
	{
		if (isVisible[i])
		{
			// find n lines with close distance and orientation
			cv::Vec4f l;
			l = prjModel[i];
			vpLine sl; 
			transLineType(l, sl);

			// find the corresponding line detected
			//
			// sl : predicted straightLine
			// *cit: hough detected straightLine

			std::vector<double> dist;
			for (size_t j = 0; j < slmodel.size(); j++)
			{
				// compareStraightLine(straightLine Model, detected straightLine, model line ID, detected line Segment)
				dist.push_back(compareStraightLine(sl, slmodel[j], l, lines[j]));
			}
			
			if (dist.size() == 0)
			{
				// TODO:do something;
			}

			std::vector<double>::iterator itr = std::min_element(dist.begin(), dist.end());
			int idx = itr - dist.begin();

#ifdef DEBUG
			// validate the connection and the isVisible steps
			std::cout<<"idx = "<<idx<<std::endl;
			std::cout<<"dist = "<<dist[idx]<<std::endl;
#endif
			// only add feature when the distance is in a reasonable scale
			if (*itr < distTH)
			{
				// line Segment
				vpPoint p1, p2;

				// image frame to image plane
				double x1, y1, x2, y2;
				vpPixelMeterConversion::convertPoint(cam, lines[idx][0], lines[idx][1], x1, y1);
				p1.set_x(x1);
				p1.set_y(y1);
				vpPixelMeterConversion::convertPoint(cam, lines[idx][2], lines[idx][3], x2, y2);
				p2.set_x(x2);
				p2.set_y(y2);

				// the corresponding world frame point has two possibles
				// p1 -> model[i][0]
				// or p1 -> model[i][1]
				bool isOnetoOne = checkPtsConnection(x1, y1, x2, y2, model[i][0], model[i][1]);
				if (isOnetoOne)
				{	
					p1.setWorldCoordinates(model[i][0].x, model[i][0].y, model[i][0].z);
					p2.setWorldCoordinates(model[i][1].x, model[i][1].y, model[i][1].z);
				}
				else
				{
					p1.setWorldCoordinates(model[i][1].x, model[i][1].y, model[i][1].z);
					p2.setWorldCoordinates(model[i][0].x, model[i][0].y, model[i][0].z);
				}

				// straightLine
				vpLine tmpLine;
				// FIXME:the transLineType here DO affect the final result
				cv::Vec4f l;
				l[0] = x1;
				l[1] = y1;
				l[2] = x2;
				l[3] = y2;
				transLineType(l, tmpLine);

				// object frame
				double a1, b1, c1, d1,
					   a2, b2, c2, d2;

				getLineEquFromID(i, a1, b1, c1, d1, a2, b2, c2, d2);
				tmpLine.setWorldCoordinates(a1, b1, c1, d1, a2, b2, c2, d2);

#ifdef DEBUG
				std::cout<<"rho = "<<tmpLine.getRho()<<std::endl;
				std::cout<<"theta = "<<tmpLine.getTheta()<<std::endl;
				tmpLine.project(cMo);
				std::cout<<"rho = "<<tmpLine.getRho()<<std::endl;
				std::cout<<"theta = "<<tmpLine.getTheta()<<std::endl;
#endif

				// save the connection
				houghLineBasedTracker::detectedLine dl;
				dl.lineID = i; 			// line ID
				dl.detectID = idx; 		// detected line ID
				dl.vl = tmpLine; 		// straightLine vpLine with image plane pts and world frame pts
				dl.p1 = p1; 			// line Segment p1 vpPoint
				dl.p2 = p2; 			// line Segment p2 vpPoint
				detectedModel.push_back(dl);

#ifdef DEBUG
				// check the line connection
				// RED for the detectedLine with HoughLinesP
				cv::line(connection, cv::Point(lines[idx][0], lines[idx][1]), cv::Point(lines[idx][2], lines[idx][3]), cv::Scalar(0, 0, 255), 1, CV_AA);
				// BLUE for the connected line in the model
				cv::line(connection, cv::Point(prjModel[i][0], prjModel[i][1]), cv::Point(prjModel[i][2], prjModel[i][3]), cv::Scalar(255, 0, 0), 1, CV_AA);
#endif

#ifndef DEBUG
				// check the line connection
				// RED for the detectedLine with HoughLinesP
				cv::line(this->processedImg, cv::Point(lines[idx][0], lines[idx][1]), cv::Point(lines[idx][2], lines[idx][3]), cv::Scalar(0, 0, 255), 1, CV_AA);
				// BLUE for the connected line in the model
				cv::line(this->processedImg, cv::Point(prjModel[i][0], prjModel[i][1]), cv::Point(prjModel[i][2], prjModel[i][3]), cv::Scalar(255, 0, 0), 1, CV_AA);
#endif
			}
			dist[idx] = INT_MAX;
		}
	}	



	// connect the corners
	bool cornerID[8];
	for (int i = 0; i < 8; i++)
		cornerID[i] = false;

	for (size_t i = 0; i < detectedModel.size(); i++)
	{
		int p1, p2;
		line2Pts(detectedModel[i].lineID, p1, p2);

		if (!cornerID[p1])
		{
			for (size_t j = 0; j < detectedModel.size(); j++)
			{
				if (detectedModel[i].lineID != detectedModel[j].lineID)
				{
					int pp1, pp2;
					line2Pts(detectedModel[j].lineID, pp1, pp2);
					if (pp1 == p1 || pp2 == p1)
					{
						cornerID[p1] = true;
						double u, v, x, y;
						vpPoint vp;
						// i, j detectedModel id
						getPtFromStraightLine(i, j, u, v);
						vpPixelMeterConversion::convertPoint(cam, u, v, x, y);
						vp.set_x(x);
						vp.set_y(y);
						vp.setWorldCoordinates(corners[p1].x, corners[p1].y, corners[p1].z);
						detectedCorners.push_back(vp);	
#ifdef DEBUG
						cv::circle(connection, cv::Point(u, v), 3, cv::Scalar(0, 0, 255));
						cv::circle(connection, cv::Point(prjCorners[p1].x,prjCorners[p1].y), 3, cv::Scalar(255, 0, 0));
#endif
#ifndef DEBUG
						cv::circle(this->processedImg, cv::Point(u, v), 3, cv::Scalar(0, 0, 255));
						cv::circle(this->processedImg, cv::Point(prjCorners[p1].x,prjCorners[p1].y), 3, cv::Scalar(255, 0, 0));
#endif
						// more points is better
						//break;
					}
				}
			}
		}

		if (!cornerID[p2])
		{
			for (size_t j = 0; j < detectedModel.size(); j++)
			{
				if (detectedModel[i].lineID != detectedModel[j].lineID)
				{
					int pp1, pp2;
					line2Pts(detectedModel[j].lineID, pp1, pp2);
					if (pp1 == p2 || pp2 == p2)
					{
						cornerID[p2] = true;
						double u, v, x, y;
						vpPoint vp;
						getPtFromStraightLine(i, j, u, v);
						vpPixelMeterConversion::convertPoint(cam, u, v, x, y);
						vp.set_x(x);
						vp.set_y(y);
						vp.setWorldCoordinates(corners[p2].x, corners[p2].y, corners[p2].z);
						detectedCorners.push_back(vp);	
#ifdef DEBUG
						cv::circle(connection, cv::Point(u, v), 3, cv::Scalar(0, 0, 255));
						cv::circle(connection, cv::Point(prjCorners[p2].x,prjCorners[p2].y), 3, cv::Scalar(255, 0, 0));
#endif
#ifndef DEBUG
						cv::circle(this->processedImg, cv::Point(u, v), 3, cv::Scalar(0, 0, 255));
						cv::circle(this->processedImg, cv::Point(prjCorners[p2].x,prjCorners[p2].y), 3, cv::Scalar(255, 0, 0));
#endif
					}
				}
			}
		}
	}

#ifdef DEBUG
	cv::imshow("connection", connection);
	cv::waitKey();
	cv::destroyWindow("connection");
#endif
}

double
houghLineBasedTracker::compareStraightLine(const vpLine& l1, const vpLine& l2, const cv::Vec4f& ml1, const cv::Vec4f& dl2)
{
	using namespace std;

	double a, b, c;
	getPrjLineEqu(ml1, a, b, c);

	double x = dl2[0];
	double y = dl2[1];
	// distance from a point(x, y) to the line a * x + b * y + c = 0
	double dist1 = abs(a * x + b * y + c) / (sqrt(a * a + b * b));	

	x = dl2[2];
	y = dl2[3];
	// distance from a point(x, y) to the line a * x + b * y + c = 0
	double dist2 = abs(a * x + b * y + c) / (sqrt(a * a + b * b));	
	double dist = (dist1 + dist2) / 2;

	// distance between points (line segment)
	std::vector<double> pd;
	pd.push_back(ptsDist(ml1[0], ml1[1], dl2[0], dl2[1]));	
	pd.push_back(ptsDist(ml1[0], ml1[1], dl2[2], dl2[3]));	
	pd.push_back(ptsDist(ml1[2], ml1[3], dl2[0], dl2[1]));	
	pd.push_back(ptsDist(ml1[2], ml1[2], dl2[2], dl2[3]));	
	std::vector<double>::const_iterator itr = std::min_element(pd.begin(), pd.end());
	if (*itr > 20)
		return INT_MAX;

	//double dRho = abs(l1.getRho() - l2.getRho());
	double dTheta = abs(l1.getTheta() - l2.getTheta());
	// three degree
	if (dTheta > 0.05)
		return INT_MAX;
	else
		return dist;
}

void 
houghLineBasedTracker::getPanelEquFromThreePt(const std::vector<cv::Point3f>& pt, double& a, double& b, double& c, double& d)
{
	cv::Point3f p1,p2,p3;

    p1 = pt[0];
    p2 = pt[1];
    p3 = pt[2];
    
	// solve the determination
    a = ( (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) );
    b = ( (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z) );
    c = ( (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x) );
    d = ( 0-(a*p1.x+b*p1.y+c*p1.z) );
}

void
houghLineBasedTracker::line2face(const int lineID, int& f1, int& f2)
{
	//0: 0 2
	//1: 0 3
	//2: 0 4
	//3: 0 5
	//
	//4: 1 2
	//5: 1 3
	//6: 1 4
	//7: 1 5
	//
	//8: 2 5
	//9: 2 3
	//10:3 4
	//11:4 5
	
	if (lineID > 11 or lineID < 0)
	{
		std::cout<<"wrong lineID!"<<std::endl;
		return;
	}
	
	if (lineID < 4)
	{
		f1 = 0;
		f2 = lineID + 2;
	}
	else if (lineID >= 4 && lineID < 8)
	{
		f1 = 1;
		f2 = lineID - 2;
	}
	else if (lineID == 8)
	{
		f1 = 2;
		f2 = 5;
	}
	else // 9 - 11
	{
		f1 = lineID - 7;
		f2 = lineID - 6;
	}
}

void
houghLineBasedTracker::line2Pts(const int lineID, int& p1, int& p2)
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
houghLineBasedTracker::getLineEquFromID(int ID, double& a1,double& b1,double& c1,double& d1,double& a2,double& b2,double& c2,double& d2)
{
	// find two faces
	int f1, f2;
	line2face(ID, f1, f2);

	// get three points for each face
	std::vector<cv::Point3f> pts1, pts2;

	// oX oY oZ are the coordinates in object frame
	vpPoint vp = pyg[f1].p[0];
	pts1.push_back(cv::Point3f(vp.get_oX(), vp.get_oY(), vp.get_oZ()));
	vp = pyg[f1].p[1];
	pts1.push_back(cv::Point3f(vp.get_oX(), vp.get_oY(), vp.get_oZ()));
	vp = pyg[f1].p[2];
	pts1.push_back(cv::Point3f(vp.get_oX(), vp.get_oY(), vp.get_oZ()));

	vp = pyg[f2].p[0];
	pts2.push_back(cv::Point3f(vp.get_oX(), vp.get_oY(), vp.get_oZ()));
	vp = pyg[f2].p[1];
	pts2.push_back(cv::Point3f(vp.get_oX(), vp.get_oY(), vp.get_oZ()));
	vp = pyg[f2].p[2];
	pts2.push_back(cv::Point3f(vp.get_oX(), vp.get_oY(), vp.get_oZ()));

	// get panel equation from the three points
	getPanelEquFromThreePt(pts1, a1, b1, c1, d1);
	getPanelEquFromThreePt(pts2, a2, b2, c2, d2);
}

void
houghLineBasedTracker::measurePose(void)
{
	vpPoseFeatures pose;

	if (detectedModel.size() < 2)
	{
		//TODO: do something
		std::cout<<"too few features!"<<detectedModel.size()<<std::endl;
		isLost = true;
		return;
	}

	// Line Features and Segment Features
	std::vector<houghLineBasedTracker::detectedLine>::const_iterator litr;
	for (litr = detectedModel.begin(); litr != detectedModel.end(); ++litr)
	{
		//pose.addFeatureLine((*litr).vl);
		vpPoint p1, p2;
		p1 = (*litr).p1;
		p2 = (*litr).p2;
		//pose.addFeatureSegment(p1, p2);
	}

	// Point Features
	std::vector<vpPoint>::const_iterator itr;
	for (itr = detectedCorners.begin(); itr != detectedCorners.end(); ++itr)
	{
		pose.addFeaturePoint(*itr);
		//pose.addFeaturePoint3D(*itr);
	}
	
	// get the pose for features
	pose.setLambda(0.6);
	// the initial value of the pose is the previous pose

	try
	{
		std::cout<<"start computePose"<<std::endl;
		// TODO: poor feature may still force the program to quit without throwing any Exception
		pose.computePose(cMo);
		poseVector.buildFrom(cMo);
		std::cout<<"complete computePose"<<std::endl;

		std::cout<<"pose vector = "<<std::endl;
		std::cout<<poseVector.t()<<std::endl;
		
		if (isFlipped(poseVector, p_Pose))
		{
			// flip back
			poseVector[3] = -poseVector[3];
			poseVector[4] = -poseVector[4];
			poseVector[5] = -poseVector[5];
			cMo.buildFrom(poseVector);
			std::cout<<"warning:flip detected!"<<std::endl;
			// TODO: add the tracking lost process codes here
		}
		else if (poseDist(poseVector, p_Pose) > 3)
		{
			//this->cMo = this->p_cMo;
			//this->poseVector = this->p_Pose;
			std::cout<<"great difference between prediction and measurement!"<<std::endl;
			// TODO: add the tracking lost process codes here
		}
		
	}
	catch(...) // catch all kinds of Exceptions
	{
		this->cMo = this->p_cMo;
		this->poseVector = this->p_Pose;
		std::cout<<"Exception raised in measurePose"<<std::endl;

		// TODO: add the tracking lost process codes here
	}
}

bool
houghLineBasedTracker::checkPtsConnection(double x1, double y1, double x2, double y2, const cv::Point3f& w1, const cv::Point3f& w2)
{
	bool isOnetoOne;

	vpPoint p1, p2;
	p1.setWorldCoordinates(w1.x, w1.y, w1.z);
	p2.setWorldCoordinates(w2.x, w2.y, w2.z);
	// project the point
	p1.changeFrame(this->p_cMo);
	p2.changeFrame(this->p_cMo);
	p1.project();	
	p2.project();	

	double d11, d12, d21, d22;
	d11 = (x1 - p1.get_x()) * (x1 - p1.get_x()) + (y1 - p1.get_y()) * (y1 - p1.get_y());
	d12 = (x1 - p2.get_x()) * (x1 - p2.get_x()) + (y1 - p2.get_y()) * (y1 - p2.get_y());
	d21 = (x2 - p1.get_x()) * (x2 - p1.get_x()) + (y2 - p1.get_y()) * (y2 - p1.get_y());
	d22 = (x2 - p2.get_x()) * (x2 - p2.get_x()) + (y2 - p2.get_y()) * (y2 - p2.get_y());


	if (d11 + d22 < d12 + d21)
		isOnetoOne = true;
	else
		isOnetoOne = false;

	return isOnetoOne;
}

void 
houghLineBasedTracker::getPtFromStraightLine(int l1, int l2, double& x, double& y)
{
	int id1 = detectedModel[l1].detectID;
	int id2 = detectedModel[l2].detectID;

	double a1, b1, c1,
		   a2, b2, c2;
	getPrjLineEqu(lines[id1], a1, b1, c1);
	getPrjLineEqu(lines[id2], a2, b2, c2);

	x = - (c1 * b2 - c2 * b1) / (a1 * b2 - a2 * b1);
	y = - (c1 * a2 - c2 * a1) / (b1 * a2 - b2 * a1);
}

void 
houghLineBasedTracker::getPrjLineEqu(const cv::Vec4f& line_, double& a, double& b, double& c)
{
	double x1, y1, x2, y2;
	x1 = line_[0];
	y1 = line_[1];
	x2 = line_[2];
	y2 = line_[3];
	a = y2 - y1;
	b = x1 - x2;
	c = y1 * (x2 - x1) - x1 * (y2 - y1);
}

inline double 
houghLineBasedTracker::ptsDist(double x1, double y1, double x2, double y2)
{
	return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

inline double 
houghLineBasedTracker::poseDist(const vpPoseVector& p1, const vpPoseVector& p2)
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
houghLineBasedTracker::isFlipped(const vpPoseVector& p1, const vpPoseVector& p2)
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
houghLineBasedTracker::retrieveImage(const cv::Mat& img)
{
	this->curImg = img.clone();
}

bool
houghLineBasedTracker::pubRst(cv::Mat& img, cv::Rect& box)
{
	img = processedImg.clone();

	box.x 		= window[1];
	box.y 		= window[0];
	box.height 	= window[2] - window[0];
	box.width 	= window[3] - window[1];

	return this->isLost;
}
