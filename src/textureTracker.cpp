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

// TODO: update procedure is not considered now
// should be considered in the future
void 
textureTracker::retrievePatch(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_, int scale, bool isDatabase)
{
	if (!isDatabase)
		curPatch.clear();

	//this->cMo = cMo_;
	for (int i = 0; i < 6; i++)
	{
		if (pyg[i].isVisible(cMo_))
		{
			// process
			std::vector<unsigned char> patch;
			for (size_t j = 0; j < patchCoor[i].size(); j++)
			{
				vpPoint P = patchCoor[i][j];
				P.changeFrame(cMo_);
				P.project();
				double u, v;
				vpMeterPixelConversion::convertPoint(cam_, P.get_x(), P.get_y(), u, v);
				patch.push_back(img.at<unsigned char>(v / scale, u / scale));
			}

			// update
			if (isDatabase)
			{
				// TODO: some STL related performance improvement might be possible here
				patches[i].push_back(patch);
				while (patches[i].size() > (size_t)numOfPatch)
					patches[i].erase(patches[i].begin());
				// DEBUG
				//std::cout<<"patch[0] = "<<(int)patch[0]<<std::endl;
				//std::cout<<"patches[i][0][0] = "<<(int)patches[i][0][0]<<std::endl;
			}
			else
			{
				curPatch[i] = patch;
				// DEBUG
				//std::cout<<"patch[0] = "<<(int)patch[0]<<std::endl;
				//std::cout<<"curPatch[i][0] = "<<(int)curPatch[i][0]<<std::endl;
			}
		}
	}
}

void
textureTracker::track(const cv::Mat& img, const cv::Mat& grad)
{
	this->curImg = img;
	this->gradient = grad;
	// TODO: convert image might cause some performance issues
	vpImageConvert::convert(curImg, vpI);

	findVisibleLines(cMo);
	nbFeatures = 0;
	for (int i = 0; i < 12; i++)
	{
		if (isVisible[i])
		{
			if (lines[i].meline == NULL)
				lines[i].initMovingEdge(vpI, cMo);
			lines[i].trackMovingEdge(vpI, cMo); // pose param cMo is not used in this function
			lines[i].initInteractionMatrixError();
			nbFeatures += lines[i].nbFeature;
		}
	}

	col = 0;
	for (int i = 0; i < 6; i++)
	{
		if (pyg[i].isVisible(cMo) && patches[i].size() != 0)
		{
			col += patchCoor[i].size();
		}
	}

	int len;
	switch (method)
	{
		case textureTracker::TYPE_EDGE:
			len = nbFeatures;
			break;
		case textureTracker::TYPE_TEXTURE:
			len = col;
			break;
		case textureTracker::TYPE_HYBRID:
			len = nbFeatures + col;
			break;
		default:
			break;
	}

	W = cv::Mat::zeros(len, len, CV_32FC1);
	robust = vpRobust(len) ;
	w = vpColVector(len);
	res = vpColVector(len);

	// optimized the pose based on the match of patches
	int itr = 0;
	curdiff = 255;
	// TODO: tune the diff threshold
	// additional criterion based on the similarity between two itrs
	for (int i = 0; i < scales; i++)
	{
		int scale = pow(2, scales - 1 - i);
		cv::Mat scaleImg;
		cv::resize(curImg, scaleImg, cv::Size(curImg.cols / scale, curImg.rows / scale));
		//bool status = true;
		stopFlag = false;
		while (itr < 10) // && !stopFlag)
		{
			// track
			retrievePatch(curImg, cMo, cam, scale);
			//if (itr == 0)
			//	measureFit(false);
			optimizePose(scaleImg, scale, itr);
			//status = measureFit(false);

			// criterion update
			itr++;
		}
	}

	// update the moving edge status
	findVisibleLines(cMo);
	for (int i = 0; i < 12; i++)
	{
		if (isVisible[i])
		{
			lines[i].updateMovingEdge(vpI, cMo); 
			if (lines[i].nbFeature == 0)
				lines[i].Reinit = true;

			lines[i].isvisible = true;
			if (lines[i].meline == NULL)
				lines[i].initMovingEdge(vpI, cMo);
			if (lines[i].Reinit)
				lines[i].reinitMovingEdge(vpI, cMo);
		}
	}

}

void 
textureTracker::initCoorOnFace(std::vector<vpPoint>& features, vpMbtPolygon& pyg, int numOfPtsPerFace)
{
	const vpPoint& 	vp1 = pyg.p[0],
			 		vp2 = pyg.p[1],
			 		vp3 = pyg.p[2];

	float step = 1.0 / (numOfPtsPerFace - 1);

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
	method = textureTracker::TYPE_EDGE;

	this->cam = cam_;
	this->cMo = cMo_;
	this->curImg = img;
	// TODO: convert image might cause some performance issues
	vpImageConvert::convert(curImg, vpI);

	numOfPatch = 10;
	// TODO: for fast computing, it should be 30
	numOfPtsPerFace = 10;
	curdiff = 255;
	gStep = 0.6;
	minStep = 0.1;
	maxStep = 0.6;

	scales = 1;

	initModel();
	// for the moving edge tracker
	initLines();
	initCoor();

	// init the patch database
	retrievePatch(img, cMo, cam, 1, true);
}

void 
textureTracker::optimizePose(const cv::Mat& img, int scale, int itr)
{
	// TODO: some code optimization can be done here to improve the performance

	cv::Mat Jacobian(col, 6, CV_32FC1);
	cv::Mat curFeature(col, 1, CV_32FC1);
	cv::Mat tarFeature(col, 1, CV_32FC1);

	int count = 0;
	for (int i = 0; i < 6; i++)
	{
		if (pyg[i].isVisible(cMo) && patches[i].size() != 0)
		{
			for (size_t j = 0; j < patchCoor[i].size(); j++)
			{
				vpPoint& p = patchCoor[i][j]; 
				p.changeFrame(cMo);
				p.project();
				double x, y;
				vpMeterPixelConversion::convertPoint(cam, p.get_x(), p.get_y(), x, y);
				//int grad = gradient.at<unsigned char>(y, x);

				cv::Mat J = jacobianImage(p, scale);

				cv::Mat G = gradientImage(img, x / scale, y / scale);
				// FIXME: why directly use patchCoor[i][j] will cause an compile error
				// FIXME: what's the type of GJ? The same with G , J or anything else?
				cv::Mat GJ = G * J;
				stackJacobian(Jacobian, GJ, count);
				curFeature.at<float>(count, 0) = curPatch[i][j];
				// TODO: the match between target and current is not correct
				tarFeature.at<float>(count, 0) = meanShiftMax(curPatch[i][j], i, j);

				count++;
			}
		}
	}


	cv::Mat JacobianMe, eMe;
	MovingEdgeBasedTracker(JacobianMe, eMe);

	cv::Mat e = curFeature - tarFeature;

	// normalize
	//float maxE 	 = *std::max_element(e.begin<float>(), e.end<float>());
	//float maxEMe = *std::max_element(eMe.begin<float>(), eMe.end<float>());
	//float rate = maxE / maxEMe;
	//eMe = rate * eMe;
	//JacobianMe = rate * JacobianMe;

	switch(method)
	{
		case textureTracker::TYPE_TEXTURE:
			break;
		case textureTracker::TYPE_HYBRID:
			stackMatrix(Jacobian, JacobianMe, Jacobian);
			stackMatrix(e, eMe, e);
			break;
		case textureTracker::TYPE_EDGE:
			e = eMe;
			Jacobian = JacobianMe;
			break;
		default:
			break;
	}

	/* robust */
	float curRes;
	if (itr == 0)
	{
		for (int i = 0; i < e.rows; i++)
			W.at<float>(i, i) = 1;
		residual = 1e8;
	}
	else
	{
		robust.setThreshold(0.0) ;

		switch(method)
		{
			case textureTracker::TYPE_TEXTURE:
				for (int i = 0; i < e.rows; i++)
					res[i] = e.at<float>(i, 0) * e.at<float>(i, 0);
				break;
			case textureTracker::TYPE_HYBRID:
				for (int i = 0; i < col; i++)
					res[i] = e.at<float>(i, 0) * e.at<float>(i, 0);
				for (int i = col; i < e.rows; i++)
				{
					res[(i - col) * 2 + col] = e.at<float>(i, 0) * e.at<float>(i, 0) 
						+ e.at<float>(i + 1, 0) * e.at<float>(i + 1, 0);
					res[(i - col) * 2 + col + 1] = res[(i - col) * 2 + col];
				}
				break;
			case textureTracker::TYPE_EDGE:
				for (int i = 0; i < e.rows / 2; i++)
				{
					res[i * 2] = e.at<float>(i, 0) * e.at<float>(i, 0) 
						+ e.at<float>(i + 1, 0) * e.at<float>(i + 1, 0);
					res[i * 2 + 1] = res[i * 2];
				}
				break;
			default:
				break;
		}

		for (size_t i = 0; i < res.getRows(); i++)
			curRes += res[i];
		curRes /= res.getRows();
		//DEBUG
		std::cout<<"curRes = "<<curRes<<std::endl;
		if (/*abs(curRes - residual) < 1e-8 ||*/ curRes < 2e-4)
		{
			stopFlag = true;
			return;
		}
		else
			residual = curRes;

		robust.setIteration(0);
		robust.MEstimator(vpRobust::TUKEY, res, w);
		for (int i = 0; i < e.rows; i++)
			W.at<float>(i, i) = w[i];
	}
	Jacobian = W * Jacobian;
	/* end of robust */

	cv::Mat L, v;
	L = Jacobian.inv(cv::DECOMP_SVD);
	//L = (Jacobian.t() * Jacobian).inv() * Jacobian.t();
	v = - gStep * L * W * e;


	vpColVector vpV(6);
	for (int i = 0; i < 6; i++)
		vpV[i] = v.at<float>(0, i);
	p_cMo = cMo;
	cMo = vpExponentialMap::direct(vpV).inverse() * cMo;



	/* backup 
//	vpPoseVector pv;
//	pv.buildFrom(cMo);
//
//	for (int i = 0; i < 6; i++)
//	{
//		std::cout<<" orig["<<i<<"] = "<<pv[i]<<std::endl; 
//		// TODO: is the param here ok?
//		pv[i] = pv[i] - v.at<float>(i, 0);
//		// DEBUG
//		std::cout<<" diff["<<i<<"] = "<<v.at<float>(i, 0)<<std::endl;
//		std::cout<<" chng["<<i<<"] = "<<pv[i]<<std::endl; 
//	}
//
//	p_cMo = cMo;
//	cMo.buildFrom(pv);
   end of backup */
}

double
textureTracker::meanShift(int intensity, int faceID, int index)
{
	double up = 0;
	double down = 0;
	for (size_t i = 0; i < patches[faceID].size(); i++)
	{
		double xi = patches[faceID][i][index];
		//TODO: delta can be tuned
		double delta = 10;
		double density = kernelDensity(intensity, xi, delta);
		up += xi * density;
		down += density;
	}
	if (down > 1e-100)
		return (up / down);
	else
		return intensity;
}

double 
textureTracker::meanShiftMax(int intensity, int faceID, int index)
{
	double cur, pre;
	cur = intensity;
	pre = INT_MAX;
	int count = 0;
	while (abs(cur - pre) > 1 && count < 10)
	{
		pre = cur;
		cur = meanShift(cur, faceID, index);
		count++;
	}

	return cur;
}

cv::Mat
textureTracker::meanShift2(int intensity, int grad, int faceID, int index)
{
	cv::Mat m(1, 2, CV_32FC1);

	double up = 0;
	double down = 0;
	for (size_t i = 0; i < patches[faceID].size(); i++)
	{
		double xi = patches[faceID][i][index];
		//TODO: delta can be tuned
		double delta = 10;
		double density = kernelDensity(intensity, xi, delta);
		up += xi * density;
		down += density;
	}
	if (down > 1e-100)
		m.at<float>(0, 0) = up / down - intensity;
	else
		m.at<float>(0, 0) = -intensity;

	// TODO: how to find the mean shift of the gradient
	m.at<float>(0, 1) = (255 - grad) / 25;

	return m;
}

inline cv::Mat
textureTracker::gradientImage(const cv::Mat& img, int x, int y)
{
	cv::Mat g(1, 2, CV_32FC1);
	g.at<float>(0, 0) = (float)img.at<unsigned char>(y, x + 1) - (float)img.at<unsigned char>(y, x);
	g.at<float>(0, 1) = (float)img.at<unsigned char>(y + 1, x) - (float)img.at<unsigned char>(y, x);

	return g;
}

// TODO: the matrix type of the gradient image
inline cv::Mat
textureTracker::gradientImage2(const cv::Mat& img, const cv::Mat& gradient, int x, int y)
{
	cv::Mat g(2, 2, CV_32FC1);
	g.at<float>(0, 0) = (float)img.at<unsigned char>(y, x + 1) - (float)img.at<unsigned char>(y, x);
	g.at<float>(0, 1) = (float)img.at<unsigned char>(y + 1, x) - (float)img.at<unsigned char>(y, x);
	g.at<float>(1, 0) = (float)gradient.at<unsigned char>(y, x + 1) - (float)gradient.at<unsigned char>(y, x);
	g.at<float>(1, 1) = (float)gradient.at<unsigned char>(y + 1, x) - (float)gradient.at<unsigned char>(y, x);

	return g;
}

inline cv::Mat
textureTracker::jacobianImage(vpPoint p, int scale)
{
	float fx = cam.get_px() / scale;
	float fy = cam.get_py() / scale;
	cv::Mat J(2, 6, CV_32FC1);

	float Z = p.get_Z() / p.get_W();
	float x = p.get_x();
	float y = p.get_y();

	J.at<float>(0, 0) = -fx * 1 / Z;
	J.at<float>(0, 1) = 0;
	J.at<float>(0, 2) = fx * x / Z;
	J.at<float>(0, 3) = fx * x * y;
	J.at<float>(0, 4) = -fx * (1 + x * x);
	J.at<float>(0, 5) = fx * y;

	J.at<float>(1, 0) = 0;
	J.at<float>(1, 1) = -fy * 1 / Z;
	J.at<float>(1, 2) = fy * y / Z;
	J.at<float>(1, 3) = fy * (1 + y * y);
	J.at<float>(1, 4) = -fy * x * y;
	J.at<float>(1, 5) = -fy * x;

	return J;
}

cv::Mat
textureTracker::meanMat(std::vector<cv::Mat>& m)
{
	cv::Mat mMat = cv::Mat::zeros(m[0].rows, m[0].cols, CV_32FC1);
	for (size_t i = 0; i < m.size(); i++)
	{
		mMat = mMat + m[i];
		//cv::add(mMat, m[i], mMat);
	}

	mMat = (1.0 / m.size()) * mMat;
	//scaleMat(mMat, 1 / m.size());
	return mMat;
}

inline cv::Mat
textureTracker::scaleMat(cv::Mat m, double scale)
{
	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.cols; j++)
			m.at<float>(i, j) *= scale;

	return m;
}

inline double
textureTracker::kernelDensity(double x, double xi, double delta)
{
	return exp(-(x - xi) * (x - xi) / delta / delta);
}


// TODO: if tmpdiff > th, reset the pose, change the step width
// this may need to update several other functions
// also need to maintain the curdiff param
bool 
textureTracker::measureFit(bool isUpdate)
{
	// DEBUG only
	return false;

	float tmpdiff = 0;
	int count = 0;
	float th = 1;
	retrievePatch(curImg, cMo, cam, 1);	
	
	for (int i = 0; i < 6; i++)
	{
		// some visible face under current pose might don't have corresponding database
		if (pyg[i].isVisible(cMo) && patches[i].size() != 0)
		{
			for (size_t j = 0; j < curPatch[i].size(); j++)
			{
				tmpdiff += pixelDiff(curPatch[i][j], i, j);
				count++;	
			}			
		}
	}
	tmpdiff /= (float)count;
	

	if (isUpdate)
	{
		if (tmpdiff < curdiff)
			curdiff = tmpdiff;

		//TODO: DEBUG
		return false;
		return tmpdiff < th;
	}
	else
	{
		if (abs(tmpdiff - curdiff) < 0.05 || tmpdiff < 1)
		{
			curdiff = tmpdiff;
			return false;
		}
		else
		{
			curdiff = tmpdiff;
			return true;
		}
	}
}

void 
textureTracker::updatePatches(vpHomogeneousMatrix& cMo_)
{
	// based on the merged info update the pose
	getPose(cMo_);

	if (measureFit(true))
	{
		// update here
		retrievePatch(curImg, cMo, cam, 1, true);
	}
}

inline float
textureTracker::pixelDiff(int intensity, int faceID, int index)
{
	float minVal = 255;
	for (size_t i = 0; i < patches[faceID].size(); i++)
	{
		float cur = abs(intensity - patches[faceID][i][index]);
		if (cur < minVal)
			minVal = cur;
	}
	return minVal;
}

inline bool
textureTracker::isEdge(int index)
{
	int i = index / numOfPtsPerFace;
	int j = index % numOfPtsPerFace;
	if (i == 0 || i == numOfPtsPerFace - 1)
		return true;
	else
	{
		if (j == 0 || j == numOfPtsPerFace - 1)
			return true;
		else
			return false;
	}
}

inline void
textureTracker::stackJacobian(cv::Mat& Jacobian, cv::Mat& GJ, int count)
{
	for (int i = 0; i < 6; i++)
		Jacobian.at<float>(count, i) = GJ.at<float>(0, i);
}

void 
textureTracker::initLines(void)
{
	for (int i = 0; i < 12; i++)
	{
		int p1, p2;
		line2Pts(i, p1, p2);
		vpPoint vp1, vp2;
		vp1.setWorldCoordinates(corners[p1].x, corners[p1].y, corners[p1].z);
		vp2.setWorldCoordinates(corners[p2].x, corners[p2].y, corners[p2].z);

		mes[i].setMaskSize(5);
		mes[i].setMaskNumber(180);
		mes[i].setRange(8);
		mes[i].setThreshold(2000);
		mes[i].setMu1(0.5);
		mes[i].setMu2(0.5);
		mes[i].setSampleStep(4);
		mes[i].setNbTotalSample(250);

		lines[i].buildFrom(vp1, vp2);
		lines[i].setCameraParameters(cam);
		lines[i].setMovingEdge(&mes[i]);
		lines[i].initMovingEdge(vpI, cMo);

		lines[i].updateMovingEdge(vpI, cMo); 
		if (lines[i].nbFeature == 0 && lines[i].isVisible())
			lines[i].Reinit = true;
	}
}

void 
textureTracker::MovingEdgeBasedTracker(cv::Mat& JacobianMe, cv::Mat& eMe)
{
	// find the visible lines
	findVisibleLines(cMo);

	// track the line
	// size of all the feature points on the visible lines
	for (int i = 0; i < 12; i++)
	{
		if (isVisible[i])
		{
			lines[i].computeInteractionMatrixError(cMo); 
		}
	}

	// create the Jacobian and the error
	JacobianMe = cv::Mat(nbFeatures, 6, CV_32FC1);
	eMe = cv::Mat(nbFeatures, 1, CV_32FC1);

	// copy the data
	// TODO: some performance issues here
	int count = 0;
	for (int i = 0; i < 12; i++)
	{
		if (isVisible[i])
		{
			for (size_t j = 0; j < lines[i].L.getRows(); j++)
			{
				for (int k = 0; k < 6; k++)
					JacobianMe.at<float>(count, k) = lines[i].L[j][k];
				eMe.at<float>(count, 0) = lines[i].error[j];

				count++;
			}
		}
	}
}

void 
textureTracker::stackMatrix(cv::Mat& src1, cv::Mat& src2, cv::Mat& dst)
{
	int row1 = src1.rows;
	int row2 = src2.rows;

	int col1 = src1.cols;
	int col2 = src2.cols;

	if (col1 != col2)
	{
		throw("the dimension of the two matrices does not match");
		return;
	}

	cv::Mat rst(row1 + row2, col1, CV_32FC1);
	for (int i = 0; i < row1; i++)
		for (int j = 0; j < col1; j++)
			rst.at<float>(i, j) = src1.at<float>(i, j);

	for (int i = 0; i < row2; i++)
		for (int j = 0; j < col2; j++)
			rst.at<float>(i + row1, j) = src2.at<float>(i, j);

	dst = rst;
}
