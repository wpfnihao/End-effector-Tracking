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
textureTracker::retrievePatch(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_, bool isDatabase)
{
	if (!isDatabase)
	{
		curPatch.clear();
		for (int i = 0; i < 6; i++)
			curHist[i] = cv::Mat::zeros(1, numOfHist, CV_32FC1);
	}

	//this->cMo = cMo_;
	for (int i = 0; i < 6; i++)
	{
		if (pyg[i].isVisible(cMo_))
		{
			// process
			cv::Mat hist = cv::Mat::zeros(1, numOfHist, CV_32FC1);
			std::vector<unsigned char> patch;
			for (size_t j = 0; j < patchCoor[i].size(); j++)
			{
				vpPoint P = patchCoor[i][j];
				P.changeFrame(cMo_);
				P.project();
				double u, v;
				vpMeterPixelConversion::convertPoint(cam_, P.get_x(), P.get_y(), u, v);
				unsigned char intensity = img.at<unsigned char>(v, u);
				patch.push_back(intensity);
				addToHist(hist, intensity);
			}

			// update
			if (isDatabase)
			{
				// TODO: some STL related performance improvement might be possible here
				patches[i].push_back(patch);
				while (patches[i].size() > (size_t)numOfPatch)
					patches[i].erase(patches[i].begin());

				hists[i].push_back(hist);
				while (hists[i].size() > (size_t)numOfPatch)
					hists[i].erase(hists[i].begin());
				// DEBUG
				//std::cout<<"patch[0] = "<<(int)patch[0]<<std::endl;
				//std::cout<<"patches[i][0][0] = "<<(int)patches[i][0][0]<<std::endl;
			}
			else
			{
				curPatch[i] = patch;
				curHist[i]  = hist;
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
	// optimized the pose based on the match of patches
	int itr = 0;
	curdiff = 255;
	// TODO: tune the diff threshold
	// additional criterion based on the similarity between two itrs
	while (itr < 10 && curdiff > 3)
	{
		// track
		retrievePatch(curImg, cMo, cam);
		//if (itr == 0)
		//	measureFit(false);
		optimizePose(curImg);
		measureFit(false);

		// criterion update
		itr++;
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
	this->cam = cam_;
	this->cMo = cMo_;

	numOfHist = 32;
	numOfPatch = 10;
	// TODO: for fast computing, it should be 30
	numOfPtsPerFace = 10;
	curdiff = 255;
	gStep = 0.2;
	minStep = 0.1;
	maxStep = 0.6;

	initModel();
	initCoor();

	// init the patch database
	retrievePatch(img, cMo, cam, true);
}

void 
textureTracker::optimizePose(const cv::Mat& img)
{
	// TODO: some code optimization can be done here to improve the performance

	// TODO: first don't consider the gradient maximum, add this later
	int col = 0;
	for (int i = 0; i < 6; i++)
	{
		if (pyg[i].isVisible(cMo) && patches[i].size() != 0)
		{
			col += patchCoor[i].size();
		}
	}
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
				int grad = gradient.at<unsigned char>(y, x);

				cv::Mat J = jacobianImage(p);

				cv::Mat G = gradientImage(img, x, y);
				// FIXME: why directly use patchCoor[i][j] will cause an compile error
				// FIXME: what's the type of GJ? The same with G , J or anything else?
				cv::Mat GJ = G * J;
				stackJacobian(Jacobian, GJ, count);
				curFeature.at<float>(count, 0) = curPatch[i][j];
				tarFeature.at<float>(count, 0) = meanShiftMax(curPatch[i][j], i, j);

				count++;
			}
		}
	}

	cv::Mat L = (Jacobian.t() * Jacobian).inv() * Jacobian.t();
	cv::Mat e = tarFeature - curFeature;

	cv::Mat v = - gStep * L * e;
	vpColVector vpV(6);
	vpV[0] = v.at<float>(0, 0);
	vpV[1] = v.at<float>(0, 1);
	vpV[2] = v.at<float>(0, 2);
	vpV[3] = v.at<float>(0, 3);
	vpV[4] = v.at<float>(0, 4);
	vpV[5] = v.at<float>(0, 5);
	p_cMo = cMo;
	cMo = vpExponentialMap::direct(vpV).inverse()*cMo;



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
textureTracker::jacobianImage(vpPoint p)
{
	float fx = cam.get_px();
	float fy = cam.get_py();
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
	float tmpdiff = 0;
	int count = 0;
	float th = 3;
	retrievePatch(curImg, cMo, cam);	
	
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
		if (tmpdiff < curdiff)
		{
			curdiff = tmpdiff;
			gStep = gStep * 2.0 > maxStep ? maxStep : gStep * 2.0;
			return true;
		}
		else
		{
			cMo = p_cMo;
			gStep = gStep / 2.0 < minStep ? minStep : gStep / 2.0;
			return false;
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
		retrievePatch(curImg, cMo, cam, true);
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

inline void
textureTracker::addToHist(cv::Mat& hist, int intensity)
{
	// TODO: requires more attention
	int binWidth = 256 / numOfHist;
	int locate = intensity / numOfHist;
	int shift = binWidth / 2;
	int cl = locate * binWidth - shift;
	int cc = cl + binWidth;
	int cr = cc + binWidth;
	if (locate == numOfHist - 1)
	{
		hist[locate - 1] += 
		hist[locate] 	 += 
	}
	else
	{
		hist[locate + 1] += 
		hist[locate] 	 += 
	}
}
