/**
 * @file cadModel.cpp
 * @brief This is the implementation of the cadModel class maintaining the cad model.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-06
 */

#include "endeffector_tracking/cadModel.h"

/**
 * @brief  	Init the model from the *.init file, only used for the cube tracking
 * 			initClick in the ViSP software is referred for the IO operations used here
 *  		fstream is used here
 */
void 
cadModel::getInitPoints(const std::string& init_file)
{
	std::fstream finit;	
	finit.open(init_file.c_str(), std::ios::in);
	if (finit.fail())
	{
		std::cout << "cannot read " << init_file << std::endl;
		throw vpException(vpException::ioError, "cannot read init file for model initialization!");
	}

	//file parser
	//number of points
	//X Y Z
	//X Y Z
	 
	double X,Y,Z ;

	unsigned int n ;
	finit >> n ;
	std::cout << "number of points " << n << std::endl ;
	for (unsigned int i=0 ; i < n ; i++)
	{
		finit >> X ;
		finit >> Y ;
		finit >> Z ;
		// NOTE: X,Y,Z are small float variables in meter, do NOT use int to save them
		cv::Point3f curP(X, Y, Z);
		// initialize the initP, which is the member variable of the class
		initP.push_back(curP); // (X,Y,Z)
	}

	finit.close();
}

void 
cadModel::initModel(void)
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


	// Step 2: init the polygons from the corners
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

void 
cadModel::projectModel(const vpHomogeneousMatrix& cMo_, const vpCameraParameters& cam)
{
	// do some clearing
	prjCorners.clear();
	
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
cadModel::findVisibleLines(const vpHomogeneousMatrix& cMo_)
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
	}
}

void
cadModel::line2face(const int lineID, int& f1, int& f2)
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
cadModel::line2Pts(const int lineID, int& p1, int& p2)
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

