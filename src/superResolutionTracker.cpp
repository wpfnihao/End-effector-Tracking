/**
 * @file superResolutionTracker.cpp
 * @brief This is the implementation of the super resolution tracker.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-08-08
 */

#include "endeffector_tracking/superResolutionTracker.h"

#include <assert.h>
#include <visp/vpImageIo.h>
#include <sstream>

superResolutionTracker::superResolutionTracker()
:numOfPatchScale(10)
,buffSize(20)
,numOfPatches(10)
,numOfPatchesUsed(2)
,superScale(1)
,winSize(7)
,maxLevel(1)
,maxDownScale(16)
,faceAngle(70)
,frameCount(0)
,minFrameCount(10)
,res(0)
,frame_num(0)
{
	omp_init_lock(&buffLock);
	omp_init_lock(&dataLock);
}

void
superResolutionTracker::track(void)
{
	int len = vpMbEdgeTracker::faces.getPolygon().size();
	float rate = getRate(maxDownScale, numOfPatchScale);
	vpImageConvert::convert(curImg, I);

	// maintain the visibility test
	int i = 0;
	for (std::vector<vpMbtPolygon *>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin(); itr != vpMbEdgeTracker::faces.getPolygon().end(); ++itr)
	{
		isVisible[i] = (*itr)->isVisible(cMo, vpMath::rad(faceAngle));
		++i;
	}

	// a copy of the whole dataset only including the tracking related patches
	dataPatches.clear();

	// upScale the src, pre, and database patches for tracking
	// src
	cv::Mat upScaleImg;
	cv::resize(curImg, upScaleImg, cv::Size(curImg.cols * rate, curImg.rows * rate));
	// dataset
	for (int i = 0; i < len; i++)
		if (isVisible[i])
		{
			int patchScale = findPatchScale(i);
			// find the corresponding patches, and copy them out of the dataset
			// then project them onto the current pose with the virtual camera
			findCopyProcessPatch(i, patchScale, cMo, dataPatches[i]);
		}

	// track
	// super resolution tracking only used in the match between the database and the current frame. The match between the pre and cur frame are based on the org scale
	//vpPoseVector pose;
	//pose.buildFrom(cMo);
	//std::cout<<pose<<std::endl;
	// FIXME: this function runs rather slow
	optimizePose(upScaleImg, prePatch, dataPatches);
	//vpMbEdgeTracker::track(I);

	// maintain the visibility test
	i = 0;
	for (std::vector<vpMbtPolygon *>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin(); itr != vpMbEdgeTracker::faces.getPolygon().end(); ++itr)
	{
		isVisible[i] = (*itr)->isVisible(cMo, vpMath::rad(faceAngle));
		++i;
	}

	// save the patch for next frame tracking
#pragma omp parallel for num_threads(4) schedule(dynamic, 1)
	for (int i = 0; i < len; i++)
	{
		prePatch[i].clear();
		if (isVisible[i])
		{
			patch p;
			obtainPatch(i, p);
			prePatch[i].push_back(p);
		}
	}

	// measure whether the tracked frame is the key frame
	// if true
	// push the frame into the buff
	if (isKeyFrame())
	{
		std::cout<<"NOTICE: a key frame detected!"<<std::endl;
		frameCount = 0;
//#pragma omp parallel for num_threads(4)
// buffLock in the pushDataIntoBuff() function, i think the parallel here is useless 
		for (int i = 0; i < len; i++)
			if (isVisible[i])
			{
				// note: one key frame will produce several patches
				// buff lock is encapsulated in the function
				patch p;
				deepCopyPrePatch(prePatch[i].front(), p);
				pushDataIntoBuff(p);
			}
	}

	// display
	vpDisplay::display(I);
	// for display
	display(I, cMo, cam, vpColor::red, 2, true);
	vpDisplay::displayFrame(I, cMo, cam, 0.025, vpColor::none, 3);
	vpDisplay::flush(I);
	vpTime::wait(1);   // TODO: what about a smaller value here?

	// writes a color RGBa image to a JPEG file on the disk.
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	std::stringstream img_num;
	img_num<<frame_num;
	std::string ext = ".jpeg";
	std::string fold = "./frames/";
	std::string img_name = fold + img_num.str() + ext;

	cv::Mat img2write = colorImg;
	for (std::vector<vpMbtPolygon *>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin(); itr != vpMbEdgeTracker::faces.getPolygon().end(); ++itr)
	{
		int npt = (*itr)->getNbPoint();
		std::vector<cv::Point> pt(npt);
		for (int i = 0; i < npt; i++)
		{
			vpPoint& vp = (*itr)->p[i];
			vp.changeFrame(cMo);
			vp.project();

			double u, v;
			vpMeterPixelConversion::convertPoint(cam, vp.get_x(), vp.get_y(), u, v);
			pt[i].x = u;
			pt[i].y = v;
		}

		for (int i = 0; i < 4; i++)
			cv::line(img2write, pt[i], pt[(i+1)%4], cv::Scalar(0, 0, 255), 2);
	}
	cv::imwrite(img_name, img2write);
	frame_num++;
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
}

void
superResolutionTracker::updateDataBase(void)
{
	bool isNewData = false;

	patch keyPatch;
	// buff lock is encapsulated in the function
	isNewData = getDataFromBuff(keyPatch);	

	// only when new key frame arrived, the update procedure will be called
	// else, the thread sleeps for a while and then check whether a new key frame has arrived.
	if (isNewData)
	{
		processKeyPatch(keyPatch);
		// integrate the patch into the database
		updataDataset(keyPatch);
		// refresh the patches already in the database, since new key frame arrived
		refreshDataset();

	}
	else
	{
		// sleep for a while
		// TODO: find a better way
		for (int i = 0; i<1e7; i++)
			;
	}
}

void 
superResolutionTracker::initDataset(void)
{
}

void 
superResolutionTracker::pushDataIntoBuff(patch& patchData)
{
	omp_set_lock(&buffLock);
	if (buff.size() < (size_t)buffSize)
		buff.push_back(patchData);
	else
		std::cout<<"WARNING: the buff of the image patches is full, and some data have been dropped!"<<std::endl;
	omp_unset_lock(&buffLock);
}

bool 
superResolutionTracker::getDataFromBuff(patch& patchData)
{
	bool isNewData;
	omp_set_lock(&buffLock);
	if (!buff.empty())
	{
		patchData = buff.front();
		buff.pop_front();
		isNewData = true;
	}
	else
		isNewData = false;
	omp_unset_lock(&buffLock);

	return isNewData;
}

void 
superResolutionTracker::updataDataset(patch& patchData)
{
	omp_set_lock(&dataLock);
	dataset[patchData.faceID].push_back(patchData);
	if (dataset[patchData.faceID].size() > (size_t)numOfPatches)
		dataset[patchData.faceID].pop_front();
	omp_unset_lock(&dataLock);
}

void 
superResolutionTracker::processKeyPatch(patch& patchData)
{
	// generate the scale patch which is in the scale closest to the orgPatch
	genOrgScalePatch(patchData);
	//
	// loop: until all the patches in all the scales generated
	for (int i = 0; i < numOfPatchScale; i++)
		genRestScalePatch(patchData, i);
}

int 
superResolutionTracker::findPatchScale(int faceID)
{
	patch p;

	cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);

	std::vector<vpMbtPolygon*>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin();
	for (int i = 0; i < faceID; i++)
		++itr;

	int npt = (*itr)->getNbPoint();
	std::vector<cv::Point> pt(npt);
	for (int i = 0; i < npt; i++)
	{
		vpPoint& vp = (*itr)->p[i];
		vp.changeFrame(cMo);
		vp.project();

		double u, v;
		vpMeterPixelConversion::convertPoint(cam, vp.get_x(), vp.get_y(), u, v);
		pt[i].x = u;
		pt[i].y = v;
	}

	for (int i = 0; i < npt; i++)
		cv::line(mask, pt[i], pt[(i+1) % npt], 255);
	cv::floodFill(mask, meanPoint(pt), 255);

	// mask
	p.mask = mask;
	p.faceID = faceID;
	findPatchScale(p);

	return p.patchScale;
}

void 
superResolutionTracker::findPatchScale(patch& patchData)
{	
	cv::Scalar np = cv::sum(patchData.mask);	
	int numOfPixels = np[0];
	numOfPixels /= 255;
	float curWidth  = 0,
		  curHeight = 0,
		  nxtWidth  = 0,
		  nxtHeight = 0;
	int patchScale = -1;
	bool scaleFound = false;
	scaleInfo& si = faceScaleInfo[patchData.faceID];
	for (int i = 0; i < numOfPatchScale; i++)
	{
		if (i == 0)
		{
			nxtWidth = si.faceSizes[i].width;
			nxtHeight = si.faceSizes[i].height;
			if (numOfPixels < nxtWidth * nxtHeight)
			{
				patchScale = i;	
				scaleFound = true;
			}
		}
		else if (i == numOfPatchScale-1)
		{
			patchScale = i;	
			scaleFound = true;
		}
		else
		{
			curWidth = nxtWidth;
			curHeight = nxtHeight;
			nxtWidth = si.faceSizes[i+1].width;
			nxtHeight = si.faceSizes[i+1].height;
			if (numOfPixels > curWidth * curHeight && numOfPixels < nxtWidth * nxtHeight)
			{
				patchScale = numOfPixels / (curWidth * curHeight) > (nxtWidth * nxtHeight) / numOfPixels ? i+1:i;	
				scaleFound = true;
			}
		}

		if (scaleFound)
			break;
	}

	patchData.patchScale = patchScale;
}

/* old version, new version see below
//void
//superResolutionTracker::genOrgScalePatch(patch& patchData)
//{
//	// set the confidence
//	patchData.confidence[patchData.patchScale] = true;
//	patchData.highestConfidenceScale = patchData.patchScale;
//	// gen the patch
//	scaleInfo& si = faceScaleInfo[patchData.faceID];
//	vpMatrix K = cam.get_K();
//	vpMatrix invK = si.Ks[patchData.patchScale].inverseByLU();
//	vpMatrix invP = si.cMo.inverseByLU();
//	vpMatrix P = patchData.pose;
//	vpMatrix PinvP = P * invP;
//	int shiftX = patchData.patchRect.x;
//	int shiftY = patchData.patchRect.y;
//	
//	int height = si.faceSizes[patchData.patchScale].height;
//	int width = si.faceSizes[patchData.patchScale].width;
//	float depth = si.depth;
//	//float* pd = (float*) (patchData.depth.data);
//	cv::Mat sp(si.faceSizes[patchData.patchScale], CV_8UC1);
//	uchar* ps = (uchar*) (sp.data);
//	for (int i = 0; i < height; i++)
//		for (int j = 0; j < width; j++)
//		{
//			int xc, yc;
//			float Z;
//			projPoint(invK, PinvP, K, depth, j, i, xc, yc, Z);
//			xc -= shiftX;
//			yc -= shiftY;
//			xc = std::min(xc, patchData.orgPatch.cols-1);
//			yc = std::min(yc, patchData.orgPatch.rows-1);
//			xc = std::max(xc, 0);
//			yc = std::max(yc, 0);
//			*ps++ = patchData.orgPatch.at<uchar>(yc, xc);
//		}
//	patchData.scaledPatch[patchData.patchScale] = sp;
//	// DEBUG only
//	cv::imshow("patch", sp);
//	cv::imshow("orgPatch", patchData.orgPatch);
//	cv::waitKey(100);
//	//END OF DEBUG
//}
*******************************************/

void
superResolutionTracker::genOrgScalePatch(patch& patchData)
{
	// set the confidence
	patchData.confidence[patchData.patchScale] = true;
	patchData.highestConfidenceScale = patchData.patchScale;
	// gen the patch
	scaleInfo& si = faceScaleInfo[patchData.faceID];
	vpMatrix K = cam.get_K();
	vpMatrix invK = si.Ks[patchData.patchScale].inverseByLU();
	vpMatrix invP = si.cMo.inverseByLU();
	vpMatrix P = patchData.pose;
	vpMatrix PinvP = P * invP;
	int shiftX = patchData.patchRect.x;
	int shiftY = patchData.patchRect.y;
	
	// get four corners
	int height = si.faceSizes[patchData.patchScale].height;
	int width = si.faceSizes[patchData.patchScale].width;
	float depth = si.depth;
	cv::Mat sp(si.faceSizes[patchData.patchScale], CV_8UC1);
	// define the corresponding points
	std::vector<cv::Point2f> orgPoints(4);
	std::vector<cv::Point2f> transPoints(4);
	// only calculate the four corners
	for (int i = 0; i < 4; i++)
	{
		// xc, yc the coordinate in the orgPatch
		// ox, oy the coordinate in the scale patch
		int xc, yc, ox, oy;
		float Z;
		switch (i)
		{
			case 0:
				ox = 0;
				oy = 0;
				break;
			case 1:
				ox = width;
				oy = 0;
				break;
			case 2:
				ox = 0;
				oy = height;
				break;
			case 3:
				ox = width;
				oy = height;
				break;
			default:
				break;
		}	
		projPoint(invK, PinvP, K, depth, ox, oy, xc, yc, Z);
		xc -= shiftX;
		yc -= shiftY;
		xc = std::min(xc, patchData.orgPatch.cols-1);
		yc = std::min(yc, patchData.orgPatch.rows-1);
		xc = std::max(xc, 0);
		yc = std::max(yc, 0);
		orgPoints[i]   = cv::Point2f(xc, yc);
		transPoints[i] = cv::Point2f(ox, oy);
	}

	// find the perspective transform
	cv::Mat warp_matrix = cv::getPerspectiveTransform(orgPoints, transPoints);
	// perspective transform the patch
	cv::warpPerspective(patchData.orgPatch, sp, warp_matrix, cv::Size(width, height));


	patchData.scaledPatch[patchData.patchScale] = sp;
	// DEBUG only
	//cv::imshow("patch", sp);
	//cv::imshow("orgPatch", patchData.orgPatch);
	//cv::waitKey(100);
	//END OF DEBUG
}

	void
superResolutionTracker::genRestScalePatch(patch& patchData, int scaleID)
{
	if (scaleID == patchData.patchScale)
		return;

	if (scaleID < patchData.patchScale)
	{
		interpolate(patchData, scaleID);
		patchData.confidence[scaleID] = true;
	}
	else
	{
		int patchID;
		if (findConfidence(scaleID, patchID, patchData))
		{
			superResolution(scaleID, patchID, patchData);
			patchData.confidence[scaleID] = true;
		}
		else
		{
			interpolate(patchData, scaleID);
			patchData.confidence[scaleID] = false;
		}
	}
}

	inline void
superResolutionTracker::interpolate(patch& patchData, int scaleID)
{
	cv::resize(
			patchData.scaledPatch[patchData.patchScale], 
			patchData.scaledPatch[scaleID], 
			faceScaleInfo[patchData.faceID].faceSizes[scaleID]
			);
}

	bool
superResolutionTracker::findConfidence(int scaleID, int& patchID, patch& patchData)
{
	std::vector<int> IDs;
	int count = 0;
	for (std::list<patch>::iterator itr = dataset[patchData.faceID].begin(); itr != dataset[patchData.faceID].end(); ++itr)
	{
		// only the original patchScale will be treated as a confidence superResolution
		if (itr->patchScale >= scaleID)
			IDs.push_back(count);
		count++;
	}

	if (IDs.empty())
		return false;
	else
	{
		cv::RNG rng;
		int idx = rng.uniform(0, (int)IDs.size());
		patchID = IDs[idx];
		return true;
	}
}

	void
superResolutionTracker::superResolution(int scaleID, int patchID, patch& patchData)
{
	// DEBUG only
	std::cout<<"start superResolution.............................................."<<std::endl;
	// END OF DEBUG
	assert(scaleID > patchData.patchScale);

	// scaleID - 1 here won't be negative
	//const cv::Mat& lowPatch = patchData.scaledPatch[scaleID - 1];
	const cv::Mat& lowPatch = patchData.scaledPatch[patchData.patchScale];
	std::list<patch>::iterator itr = dataset[patchData.faceID].begin();
	for (int i = 0; i < patchID; i++)
		++itr;
	const cv::Mat& highPatch = itr->scaledPatch[scaleID];

	cv::Size lowSize(lowPatch.cols, lowPatch.rows);
	cv::Size highSize(highPatch.cols, highPatch.rows);
	cv::Mat count = cv::Mat::zeros(lowSize, CV_32FC1) + 1e-10;
	cv::Mat tarLowPatch = cv::Mat::zeros(lowSize, CV_32FC1);
	cv::Mat light = cv::Mat::ones(lowSize, CV_32FC1);
	patchData.scaledPatch[scaleID] = cv::Mat(highSize, CV_8UC1);
	cv::Mat xCoor(highSize, CV_32SC1);
	cv::Mat yCoor(highSize, CV_32SC1);

	

	//// define the corresponding points
	//std::vector<cv::Point2f> orgPoints(4);
	//std::vector<cv::Point2f> transPoints(4);
	//// save the corresponding points
	//orgPoints[0]   = cv::Point2f(0, 				 0);
	//transPoints[0] = cv::Point2f(0, 				 0);
	//orgPoints[1]   = cv::Point2f(highSize.width - 1, 0);
	//transPoints[1] = cv::Point2f(lowSize.width  - 1, 0);
	//orgPoints[2]   = cv::Point2f(0, 				 highSize.height - 1);
	//transPoints[2] = cv::Point2f(0, 				 lowSize.height  - 1);
	//orgPoints[3]   = cv::Point2f(highSize.width - 1, highSize.height - 1);
	//transPoints[3] = cv::Point2f(lowSize.width  - 1, lowSize.height  - 1);
	//// find the perspective transform
	//cv::Mat warp_matrix = cv::getPerspectiveTransform(orgPoints, transPoints);
	//// perspective transform the patch
	//cv::warpPerspective(highPatch, tarLowPatch, warp_matrix, cv::Size(lowSize.width, lowSize.height));




	// process
	// hight to low projection
	scaleInfo& si = faceScaleInfo[patchData.faceID];
	//vpMatrix K = si.Ks[scaleID-1];
	vpMatrix K = si.Ks[patchData.patchScale];
	vpMatrix invK = si.Ks[scaleID].inverseByLU();
	vpMatrix invP = si.cMo.inverseByLU();
	vpMatrix P = si.cMo;
	vpMatrix PinvP = P * invP;
	float depth = si.depth;
	const uchar* hp = (const uchar*) (highPatch.data);
	int* xp = (int*) (xCoor.data);
	int* yp = (int*) (yCoor.data);
	for (int i = 0; i < highPatch.rows; i++)
		for (int j = 0; j < highPatch.cols; j++)		
		{
			// project
			int xc, yc;
			float Z;
			projPoint(invK, PinvP, K, depth, j, i, xc, yc, Z);
			xc = std::min(xc, lowSize.width-1);
			yc = std::min(yc, lowSize.height-1);
			xc = std::max(xc, 0);
			yc = std::max(yc, 0);

			// update
			*xp++ = xc;
			*yp++ = yc;
			count.at<float>(yc, xc) += 1;
			tarLowPatch.at<float>(yc, xc) += *hp++;
		}

	cv::divide(tarLowPatch, count, tarLowPatch);

	findLight(tarLowPatch, lowPatch, light);

	// DEBUG only
	//std::cout<<"start fill patch"<<std::endl;
	// END OF DEBUG

	// generate the super resolution patch
	hp = (const uchar*) (highPatch.data);
	uchar* chp = (uchar*) (patchData.scaledPatch[scaleID].data);
	xp = (int*) (xCoor.data);
	yp = (int*) (yCoor.data);
	omp_set_lock(&dataLock);
	for (int i = 0; i < highPatch.rows; i++)
		for (int j = 0; j < highPatch.cols; j++)		
			*chp++ = (uchar) std::min((int)((float)(*hp++) * light.at<float>(*yp++, *xp++)), 255);
	omp_unset_lock(&dataLock);

	// DEBUG only
	std::cout<<"end superResolution.............................................."<<std::endl;
	// END OF DEBUG
	
	
	

	// save processed data
	//patchData.scaledPatch[scaleID] = curHighPatch;	

	// DEBUG only
	//std::cout<<"after fill patch"<<std::endl;
	// END OF DEBUG

	// DEBUG only
	//std::cout<<"start checking the database..."<<std::endl;
	//// go through all the patches in the database
	//for (size_t i = 0; i < faces.getPolygon().size(); i++)
	//	for (std::list<patch>::iterator itr = dataset[i].begin(); itr != dataset[i].end(); ++itr)
	//		for (int j = 0; j < numOfPatchScale; j++)
	//		{
	//			std::cout<<"face "<<i<<", scale"<<j<<std::endl;
	//			cv::imshow("database", itr->scaledPatch[j]);
	//			cv::waitKey(100);
	//		}
	//std::cout<<"finish checking the database..."<<std::endl;
	// END OF DEBUG

	//DEBUG only
	//cv::imshow("tarLowPatch", tarLowPatch / 255);
	//cv::imshow("light", (light - 0.5) / 4.5);
	//cv::imshow("hightPatch", highPatch);
	//cv::imshow("lowPatch", lowPatch);
	//cv::imshow("curHighPatch", patchData.scaledPatch[scaleID]);
	//cv::waitKey(100);
	//std::cout<<"dummy sentence, for debug only."<<std::endl;
	// END OF DEBUG
}

	void
superResolutionTracker::refreshDataset(void)
{
	int len = vpMbEdgeTracker::faces.getPolygon().size();

	// reset the isChanged flags
	for (int i = 0; i < numOfPatchScale; i++)	
		for (int j = 0; j < len; j++)
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
				itrPatch->isChanged[i] = false;

	// number of scales, start from the second lowest scale
	// there is no need to update the lowest scale, so we start from scale 1 here
	for (int i = 1; i < numOfPatchScale; i++)	
		for (int j = 0; j < len; j++) // number of faces
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
				if (!itrPatch->confidence[i]) // only those without confidence should be updated
				{
					int patchID;
					if (findConfidence(i, patchID, *itrPatch))
					{
						superResolution(i, patchID, *itrPatch);
						itrPatch->confidence[i] = true;
						itrPatch->isChanged[i] = true;
					}
					else
						if (itrPatch->isChanged[i-1])
						{
							omp_set_lock(&dataLock);
							interpolate(*itrPatch, i);
							itrPatch->isChanged[i] = true;
							omp_unset_lock(&dataLock);
						}
				}

	// maintain the highestConfidenceScale
	omp_set_lock(&dataLock);
	for (int i = 0; i < numOfPatchScale; i++)	
		for (size_t j = 0; j < len; j++)
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
				if (itrPatch->confidence[i])
					itrPatch->highestConfidenceScale = i;
	omp_unset_lock(&dataLock);

}

	void
superResolutionTracker::findLight(const cv::Mat& tarLowPatch,const cv::Mat& lowPatch, cv::Mat& light)
{
	// use graph cuts here
	int num_pixels = light.rows * light.cols;
	int width = light.cols;
	int height = light.rows;
	int num_labels = 50;

	// setup the weight of labels
	float low = 0.5,
		  high = 5;
	std::vector<float> lightWeight(num_labels);
	for (int i = 0; i < num_labels; i++)
		lightWeight[i] = low + (float)i / num_labels * (high - low);

	// FIXME: tune this value
	int smoothWeight = 20;
	int ws = 2;
	try
	{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// first set up data costs individually
		float *tar = (float*) (tarLowPatch.data);
		//uchar *cur = (uchar*) (lowPatch.data);
		for ( int i = 0; i < num_pixels; i++ )
			for (int l = 0; l < num_labels; l++ )
			{
				// TODO: complete this function
				int cost = findMinCost(lightWeight[l] * tar[i], i, lowPatch, height , width, ws);
				//int cost = fabs(lightWeight[l] * tar[i] - (float)cur[i]);
				gc->setDataCost(i,l, cost);
			}

		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ )
			{
				int cost = smoothWeight * fabs(lightWeight[l1] - lightWeight[l2]);
				gc->setSmoothCost(l1,l2,cost); 
			}

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->swap(2);
		//gc->expansion(1);
		//gc->swap(1);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		float *result = (float*) (light.data);
		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = lightWeight[gc->whatLabel(i)];

		delete gc;
	}
	catch (GCException e)
	{
		e.Report();
	}
}

// FIXME: if necessary, only use the most close scale
	void
superResolutionTracker::findCopyProcessPatch(
		int faceID,
		int patchScale,
		vpHomogeneousMatrix pose,
		std::list<patch>& patchList)
{
	// check whether the dataset is empty, is so, do nothing
	omp_set_lock(&dataLock);
	bool isEmpty = dataset[faceID].empty();
	omp_unset_lock(&dataLock);
	if (isEmpty)
		return;	

	omp_set_lock(&dataLock);
	// 1. Choose which patch to copy
	// 1.1. Randomly choose among the database, or (activated)
	// 1.2. Choose the patch has the closest pose (future work)
	//
	// 2. Then, find the highest confidence patch scale in the chosen patch
	// 3. Copy then out
	std::vector<int> id; // the id of the patches whose highestConfidenceScale is larger than the curPatch scale
	int count = 0;
	int highestScale = 0;
	int highestID = 0;
	for (std::list<patch>::iterator itr = dataset[faceID].begin(); itr != dataset[faceID].end(); ++itr)
	{
		//if (itr->highestConfidenceScale >= std::min(patchScale + superScale, numOfPatchScale-1))
		id.push_back(count);
		//if (itr->highestConfidenceScale > highestScale)
		//{
		//	highestScale = itr->highestConfidenceScale;
		//	highestID = count;
		//}
		++count;
	}

	if (id.empty())
	{
		//the deepCopyPatch here only copy the information used by the tracker
		std::list<patch>::iterator itr = dataset[faceID].begin();
		for (int i = 0; i < highestID; i++)
			++itr;
		for (int i = 0; i < numOfPatchesUsed; i++)
			deepCopyPatch(patchList, *itr);
	}
	else
	{
		std::vector<int> idx(numOfPatchesUsed);
		vpPoseVector vp;
		vp.buildFrom(pose);
		findClosestPatch(vp, id, idx, numOfPatchesUsed, faceID);
		for (int i = 0; i < numOfPatchesUsed; i++)
		{
			std::list<patch>::iterator itr = dataset[faceID].begin();
			for (int j = 0; j < idx[i]; j++)
				++itr;
			deepCopyPatch(patchList, *itr);
		}
	}
	omp_unset_lock(&dataLock);

	// 4. Project them on the image based on the virtual internal parameters and the pre-pose
	projPatch(pose, faceID, patchList);
}

/* old version back up
void
superResolutionTracker::projPatch(vpHomogeneousMatrix pose, int id, std::list<patch>& dataPatches)
{
	float rate = getRate(maxDownScale, numOfPatchScale);
	int r = rate * rows,
		c = rate * cols;
	for (std::list<patch>::iterator itr = dataPatches.begin(); itr != dataPatches.end(); ++itr)
	{
		itr->pose = pose;

		int hs = itr->highestConfidenceScale;

		cv::Mat projImg = cv::Mat::zeros(r, c, CV_8UC1);		
		cv::Mat depth   = cv::Mat::zeros(r, c, CV_32FC1);		
		cv::Mat mask    = cv::Mat::zeros(r, c, CV_8UC1);		

		vpMatrix invK = faceScaleInfo[id].Ks[hs].inverseByLU();
		vpMatrix invP = faceScaleInfo[id].cMo.inverseByLU();
		vpHomogeneousMatrix P = itr->pose;
		vpMatrix PP= itr->pose;
		vpMatrix PinvP = PP * invP;
		vpMatrix K = virtualCam;
		float dp = faceScaleInfo[id].depth;

		// find the four corners in the org rectangle patch
		cv::Size pSize = faceScaleInfo[id].faceSizes[hs];
		std::vector<cv::Point> corners;
		for (int i = 0; i < 4; i++)
		{
			int x, y;
			switch (i)
			{
				case 0:
					x = 0;
					y = 0;
					break;
				case 1:
					x = 0;
					y = pSize.height;
					break;
				case 2:
					x = pSize.width;
					y = pSize.height;
					break;
				case 3:
					x = pSize.width;
					y = 0;
					break;
				default:
					break;
			}

			// project the corners to the target quadrangle patch
			int xc, yc;
			float Z;
			projPoint(invK, PinvP, K, dp, x, y, xc, yc, Z);
			corners.push_back(cv::Point(xc, yc));
		}
		// fill the mask
		for (size_t i = 0; i < 4; i++)
			cv::line(mask, corners[i], corners[(i+1) % 4], 255);
		cv::floodFill(mask, meanPoint(corners), 255);

		// restore the four corners in 3d space
		vpPoint corners3d[4];
		for (int i = 0; i < 4; i++)
		{
			backProj(invK, invP, dp, corners[i].x, corners[i].y, corners3d[i]);
			corners3d[i].changeFrame(P);
		}

		invK = invVirtualCam;
		invP = (itr->pose).inverseByLU();
		P = faceScaleInfo[id].cMo;
		PP = faceScaleInfo[id].cMo;
		PinvP = PP * invP;
		K = faceScaleInfo[id].Ks[hs];
		uchar* pm = (uchar*) (mask.data);
		float* pd = (float*) (depth.data);
		uchar* pi = (uchar*) (projImg.data);
		const cv::Mat& oPatch = itr->scaledPatch[hs];
		vpMatrix A(3, 3), 
				 b(3, 1);
		float coefficient[6];    
		//	a11a22
		//  a01a12
		//  a02a21
		//  a12a21
		//  a01a22		
		//  a02a11
		cv::Point lu, rb; //left up, right bottom
		findMinMax(corners, lu, rb);
		prepCalcDepth(corners3d, A, b, coefficient);
		int maxy = rb.y + 1;
#pragma omp parallel for num_threads(4)
		for (int i = lu.y; i < maxy; i++)
			for (int j = lu.x; j < rb.x + 1; j++)
			{
				int offset = c * i + j;
				if (*(pm + offset) == 255)
				{
					*(pd + offset) = calcDepth(A, b, coefficient, invP, invK, cv::Point(j, i));
					int xc, yc;
					float Z;
					projPoint(invK, PinvP, K, *(pd + offset), j, i, xc, yc, Z);
					xc = std::min(xc, oPatch.cols-1);
					yc = std::min(yc, oPatch.rows-1);
					xc = std::max(xc, 0);
					yc = std::max(yc, 0);
					// still have the possibility to get segmentation fault here
					*(pi + offset) = oPatch.at<uchar>(yc, xc);
				}
			}

		// copy to the patches
		itr->orgPatch = projImg;
		itr->depth = depth;
		itr->mask = mask;
		//DEBUG only
		//cv::imshow("scalepatch", itr->scaledPatch[hs]);
		//cv::imshow("projImg", itr->orgPatch);
		//cv::imshow("mask", itr->mask);
		//cv::imshow("depth", itr->depth);
		//cv::waitKey();
		// END OF DEBUG
	}
}
end of old version back up */

void
superResolutionTracker::projPatch(vpHomogeneousMatrix pose, int id, std::list<patch>& dataPatches)
{
	float rate = getRate(maxDownScale, numOfPatchScale);
	int r = rate * rows,
		c = rate * cols;
	for (std::list<patch>::iterator itr = dataPatches.begin(); itr != dataPatches.end(); ++itr)
	{
		itr->pose = pose;

		int hs = itr->highestConfidenceScale;

		cv::Mat projImg = cv::Mat::zeros(r, c, CV_8UC1);		
		cv::Mat depth   = cv::Mat::zeros(r, c, CV_32FC1);		
		cv::Mat mask    = cv::Mat::zeros(r, c, CV_8UC1);		

		vpMatrix invK = faceScaleInfo[id].Ks[hs].inverseByLU();
		vpMatrix invP = faceScaleInfo[id].cMo.inverseByLU();
		vpHomogeneousMatrix P = itr->pose;
		vpMatrix PP= itr->pose;
		vpMatrix PinvP = PP * invP;
		vpMatrix K = virtualCam;
		float dp = faceScaleInfo[id].depth;

		// define the corresponding points
		std::vector<cv::Point2f> orgPoints(4);
		std::vector<cv::Point2f> transPoints(4);
		// find the four corners in the org rectangle patch
		cv::Size pSize = faceScaleInfo[id].faceSizes[hs];
		std::vector<cv::Point> corners;
		for (int i = 0; i < 4; i++)
		{
			int x, y;
			switch (i)
			{
				case 0:
					x = 0;
					y = 0;
					break;
				case 1:
					x = 0;
					y = pSize.height - 1;
					break;
				case 2:
					x = pSize.width  - 1;
					y = pSize.height - 1;
					break;
				case 3:
					x = pSize.width  - 1;
					y = 0;
					break;
				default:
					break;
			}

			// project the corners to the target quadrangle patch
			int xc, yc;
			float Z;
			projPoint(invK, PinvP, K, dp, x, y, xc, yc, Z);
			corners.push_back(cv::Point(xc, yc));

			// save the corresponding points
			orgPoints[i]   = cv::Point2f(x, y);
			transPoints[i] = cv::Point2f(xc, yc);
		}
		// fill the mask
		for (size_t i = 0; i < 4; i++)
			cv::line(mask, corners[i], corners[(i+1) % 4], 255);
		cv::floodFill(mask, meanPoint(corners), 255);

		// find the perspective transform
		cv::Mat warp_matrix = cv::getPerspectiveTransform(orgPoints, transPoints);
		// perspective transform the patch
		const cv::Mat& oPatch = itr->scaledPatch[hs];
		cv::warpPerspective(oPatch, projImg, warp_matrix, cv::Size(c, r));



		// restore the four corners in 3d space
		vpPoint corners3d[4];
		for (int i = 0; i < 4; i++)
		{
			backProj(invK, invP, dp, corners[i].x, corners[i].y, corners3d[i]);
			corners3d[i].changeFrame(P);
		}

		invK = invVirtualCam;
		invP = (itr->pose).inverseByLU();
		P = faceScaleInfo[id].cMo;
		PP = faceScaleInfo[id].cMo;
		PinvP = PP * invP;
		K = faceScaleInfo[id].Ks[hs];
		uchar* pm = (uchar*) (mask.data);
		float* pd = (float*) (depth.data);
		uchar* pi = (uchar*) (projImg.data);
		vpMatrix A(3, 3), 
				 b(3, 1);
		float coefficient[6];    
		//	a11a22
		//  a01a12
		//  a02a21
		//  a12a21
		//  a01a22		
		//  a02a11
		cv::Point lu, rb; //left up, right bottom
		findMinMax(corners, lu, rb);
		prepCalcDepth(corners3d, A, b, coefficient);
		int maxy = rb.y + 1;
#pragma omp parallel for num_threads(4)
		for (int i = lu.y; i < maxy; i++)
			for (int j = lu.x; j < rb.x + 1; j++)
			{
				int offset = c * i + j;
				if (*(pm + offset) == 255)
					*(pd + offset) = calcDepth(A, b, coefficient, invP, invK, cv::Point(j, i));
			}

		// copy to the patches
		itr->orgPatch = projImg;
		itr->depth = depth;
		itr->mask = mask;
		//DEBUG only
		//cv::imshow("scalepatch", itr->scaledPatch[hs]);
		//cv::imshow("projImg", itr->orgPatch);
		//cv::imshow("mask", itr->mask);
		//cv::imshow("depth", itr->depth);
		//cv::waitKey();
		// END OF DEBUG
	}
}

void
superResolutionTracker::deepCopyPatch(std::list<patch>& dataPatches, patch& src)
{
	patch p;
	p.scaledPatch[src.highestConfidenceScale] = src.scaledPatch[src.highestConfidenceScale].clone();
	p.highestConfidenceScale = src.highestConfidenceScale;
	p.patchScale = src.patchScale;
	p.faceID = src.faceID;
	p.pose = src.pose;

	dataPatches.push_back(p);
}

// FIXME: note that the codes here assume that the faces are rectangles
void
superResolutionTracker::obtainPatch(int faceID, patch& p)
{
	// note: the patchRect should be handled here
	// project the face onto the mask image, draw the boarder lines
	// save the depth of the four corners
	// fill the projected face

	cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::Mat depth = cv::Mat::zeros(rows, cols, CV_32FC1);

	std::vector<vpMbtPolygon*>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin();
	for (int i = 0; i < faceID; i++)
		++itr;
	int npt = (*itr)->getNbPoint();
	std::vector<cv::Point> pt(npt);
	for (int i = 0; i < npt; i++)
	{
		vpPoint& vp = (*itr)->p[i];
		vp.changeFrame(cMo);
		vp.project();

		double u, v;
		vpMeterPixelConversion::convertPoint(cam, vp.get_x(), vp.get_y(), u, v);
		u = std::min((int)u, cols-1);
		v = std::min((int)v, rows-1);
		u = std::max((int)u, 0);
		v = std::max((int)v, 0);
		pt[i].x = u;
		pt[i].y = v;
	}

	for (int i = 0; i < npt; i++)
		cv::line(mask, pt[i], pt[(i+1) % npt], 255);
	cv::floodFill(mask, meanPoint(pt), 255);

	// handle the mask problem here
	cv::Point lt, rb;
	lt.x = cols;
	lt.y = rows;
	rb.x = 0;
	rb.y = 0;
	for (int i = 0; i < npt; i++)
	{
		if (pt[i].x < lt.x)
			lt.x = pt[i].x;
		if (pt[i].y < lt.y)
			lt.y = pt[i].y;
		if (pt[i].x > rb.x)
			rb.x = pt[i].x;
		if (pt[i].y > rb.y)
			rb.y = pt[i].y;
	}

	// fill the depth map using the interpolation method
	vpMatrix invP = cMo.inverseByLU();
	vpMatrix invK = cam.get_K().inverseByLU();
	vpMatrix A(3, 3),
			 b(3, 1);
	float coefficient[6];
	prepCalcDepth((*itr)->p, A, b, coefficient); 
	uchar* pm = (uchar*) (mask.data);
	float* pd = (float*) (depth.data);
	// obtainPatch is paralleled in the upper level functions which call it
	for (int i = 0; i < mask.rows; i++)
		for (int j = 0; j < mask.cols; j++)
		{
			int offset = i * mask.cols + j;
			if (*(pm + offset) == 255)
				*(pd + offset) = calcDepth(A, b, coefficient, invP, invK, cv::Point(j, i));
		}


	// save the patch using the filled Project face
	//
	// patchRect
	p.patchRect.x 		= lt.x;
	p.patchRect.y 		= lt.y;
	p.patchRect.width 	= rb.x - lt.x + 1;
	p.patchRect.height 	= rb.y - lt.y + 1;
	// faceID
	p.faceID = faceID;
	// pose
	p.pose = cMo;
	// orgPatch
	p.orgPatch = cv::Mat(curImg, cv::Range(lt.y, rb.y + 1), cv::Range(lt.x, rb.x + 1)).clone();
	// mask
	p.mask = cv::Mat(mask, cv::Range(lt.y, rb.y + 1), cv::Range(lt.x, rb.x + 1));
	// depth
	p.depth = cv::Mat(depth, cv::Range(lt.y, rb.y + 1), cv::Range(lt.x, rb.x + 1));
	// scale
	findPatchScale(p);
}

vpMatrix 
superResolutionTracker::getVirtualCam(void)
{
	float rate = getRate(maxDownScale, numOfPatchScale);
	vpMatrix K = cam.get_K();
	K[0][0] *= rate;
	K[0][2] *= rate;
	K[1][1] *= rate;
	K[1][2] *= rate;
	return K;
}

// old version backup, new version can be found in the next function
//
//inline void
//superResolutionTracker::backProj(
//		const vpMatrix& invK, 
//		const vpMatrix& invP, 
//		float 			depth, 
//		float 			ix, 
//		float 			iy, 
//		vpPoint&        p)
//{
//	// 2d coordinate in org frame
//	vpMatrix x(3, 1);
//	x[0][0] = ix;
//	x[1][0] = iy;
//	x[2][0] = 1;
//
//	// 3d coordinate without homo
//	x = depth * invK * x;
//	// 3d coordinate homo
//	vpMatrix X(4, 1);
//	X[0][0] = x[0][0];
//	X[1][0] = x[1][0];
//	X[2][0] = x[2][0];
//	X[3][0] = 1;
//	X = invP * X;
//
//	// 3d homo coordinatein the org frame
//	p.setWorldCoordinates(X[0][0], X[1][0], X[2][0]);
//}

inline void
superResolutionTracker::backProj(
		const vpMatrix& invK, 
		const vpMatrix& invP, 
		float 			depth, 
		float 			ix, 
		float 			iy, 
		vpPoint&        p)
{
	// 2d coordinate in org frame
	vpMatrix x(3, 1);
	// 3d coordinate without homo
	x[0][0] = depth * (ix * invK[0][0] + iy * invK[0][1] + invK[0][2]);
	x[1][0] = depth * (ix * invK[1][0] + iy * invK[1][1] + invK[1][2]);
	x[2][0] = depth * (ix * invK[2][0] + iy * invK[2][1] + invK[2][2]);

	// 3d coordinate homo
	vpMatrix X(4, 1);
	X[0][0] = x[0][0] * invP[0][0] + x[1][0] * invP[0][1] + x[2][0] * invP[0][2] + invP[0][3];
	X[1][0] = x[0][0] * invP[1][0] + x[1][0] * invP[1][1] + x[2][0] * invP[1][2] + invP[1][3];
	X[2][0] = x[0][0] * invP[2][0] + x[1][0] * invP[2][1] + x[2][0] * invP[2][2] + invP[2][3];

	// 3d homo coordinatein the org frame
	p.setWorldCoordinates(X[0][0], X[1][0], X[2][0]);
}

// old version backup, new version can be found in the next function
//
//inline void
//superResolutionTracker::projPoint(
//		const vpMatrix& invK, 
//		const vpMatrix& invP, 
//		const vpHomogeneousMatrix& P, 
//		const vpMatrix& K, 
//		float 			depth, 
//		float 			ix, 
//		float 			iy, 
//		int& 			xc, 
//		int& 			yc, 
//		float& 			Z)
//{
//	vpPoint p;
//	backProj(invK, invP, depth, ix, iy, p);
//	// 3d homo coordinate in the target frame
//	p.changeFrame(P);
//	// 3d coordinate in the target frame
//	vpMatrix X(3, 1); 
//	X[0][0] = p.get_X();
//	X[1][0] = p.get_Y();
//	X[2][0] = p.get_Z();
//	// 2d coordinate in the target frame
//	X = K * X;
//	xc = X[0][0] / X[2][0];
//	yc = X[1][0] / X[2][0];
//
//	// final depth required
//	Z = p.get_Z();
//}

inline void
superResolutionTracker::projPoint(
		const vpMatrix& invK, 
		const vpMatrix& PinvP, 
		const vpMatrix& K, 
		float 			depth, 
		float 			ix, 
		float 			iy, 
		int& 			xc, 
		int& 			yc, 
		float& 			Z)
{
	// 2d coordinate in org frame
	vpMatrix x(3, 1);
	// 3d coordinate without homo
	x[0][0] = depth * (ix * invK[0][0] + iy * invK[0][1] + invK[0][2]);
	x[1][0] = depth * (ix * invK[1][0] + iy * invK[1][1] + invK[1][2]);
	x[2][0] = depth * (ix * invK[2][0] + iy * invK[2][1] + invK[2][2]);

	// 3d coordinate homo
	vpMatrix X(3, 1);
	X[0][0] = x[0][0] * PinvP[0][0] + x[1][0] * PinvP[0][1] + x[2][0] * PinvP[0][2] + PinvP[0][3];
	X[1][0] = x[0][0] * PinvP[1][0] + x[1][0] * PinvP[1][1] + x[2][0] * PinvP[1][2] + PinvP[1][3];
	X[2][0] = x[0][0] * PinvP[2][0] + x[1][0] * PinvP[2][1] + x[2][0] * PinvP[2][2] + PinvP[2][3];

	// 2d coordinate in the target frame
	X = K * X;
	float tmp = 1 / X[2][0];
	xc = X[0][0] * tmp;
	yc = X[1][0] * tmp;

	// final depth required
	Z = X[2][0];
}

// note this function is obsoleted
/*
   float
   superResolutionTracker::calcDepth(
   const std::vector<cv::Point>& p, 
   const std::vector<float>& depth,
   cv::Point cp)
   {
// solve the equ. Ax = b, x = A\b;
cv::Mat A(2, 2, CV_32FC1);
float* a = (float*) (A.data);
 *(a + 0) = p[1].x - p[0].x;
 *(a + 1) = p[1].x - p[2].x;
 *(a + 2) = p[1].y - p[0].y;
 *(a + 3) = p[1].y - p[2].y;

 cv::Mat b(2, 1, CV_32FC1);
 float* pb = (float*) (b.data);
 *(pb + 0) = cp.x + p[1].x - p[0].x - p[2].x;
 *(pb + 1) = cp.y + p[1].y - p[0].y - p[2].y;

 cv::Mat x(2, 1, CV_32FC1);
 x = A.inv() * b;

// calculate the depth
float* xb = (float*) (x.data);
float alpha = *(xb + 0);
float beta  = *(xb + 1);
float dp = (alpha + beta - 1) * depth[1] + (1 - alpha) * depth[0] + (1 - beta) * depth[2];
return dp;
}
*/

inline cv::Point
superResolutionTracker::meanPoint(const std::vector<cv::Point>& p)
{
	cv::Point cvp;
	for (size_t i = 0; i < p.size(); i++)
	{
		cvp.x += p[i].x;
		cvp.y += p[i].y;
	}
	cvp.x /= p.size();
	cvp.y /= p.size();
	return cvp;
}

void 
superResolutionTracker::optimizePose(cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches)
{

	// find the scale of the target
	float tarScale = 0;
	int faceCount = 0;
	for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
		if (isVisible[i])
			if(!prePatch[i].empty())
			{
				tarScale += prePatch[i].front().patchScale;
				++faceCount;
			}
	tarScale /= faceCount;

	p_cMo = cMo;
	res = getPose(I, tarScale, img, prePatch, dataPatches);

	////
	//// compute pose
	//featuresComputePose.setLambda(0.6);
	//p_cMo = cMo;
	//try
	//{
	//	featuresComputePose.computePose(cMo, vpPoseFeatures::ROBUST_VIRTUAL_VS);
	//}
	//catch(...) // catch all kinds of Exceptions
	//{
	//	std::cout<<"Exception raised in computePose"<<std::endl;
	//}
}

void
superResolutionTracker::deepCopyPrePatch(patch& src, patch& p)
{
	p.patchScale 	= src.patchScale;
	p.orgPatch 		= src.orgPatch.clone();
	p.mask 			= src.mask.clone();
	p.depth 		= src.depth.clone();
	p.patchRect 	= src.patchRect;
	p.faceID 		= src.faceID;
	p.pose 			= src.pose;
}

// TODO: complete this function
bool
superResolutionTracker::isKeyFrame(void)
{
	float th1 = 0.001;
	float th2 = 0.0035;  // for stick // back up
	//float th2 = 0.004;  // for stick
	//float th2 = 0.002; // for cracker
	float th3 = 10;
	bool frameDist = true, 
		 poseDiff  = true, 
		 matchness = true,
		 ncc 	   = true;

	// frameDist
	frameDist = frameCount > minFrameCount ? true : false;
	if (!frameDist)
		return false;

	// the residual
	std::cout<<"res = "<<res<<std::endl;
	matchness = res < th2 ? true : false;
	if (!matchness)
		return false;

	// poseDiff
	vpPoseVector pose;
	pose.buildFrom(cMo);
	omp_set_lock(&dataLock);
	float diff = 0;
	for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
		for (std::list<patch>::iterator itr = dataset[i].begin(); itr != dataset[i].end(); ++itr)
		{
			vpPoseVector curPose;
			curPose.buildFrom(itr->pose);
			diff = 0;
			for (int j = 0; j < 6; j++)
				diff += fabs(pose[j] - curPose[j]);
			//std::cout<<"diff1 = "<<diff<<std::endl;
			if (diff < 6 * th1)
			{
				poseDiff = false;
				break;
			}
		}
	omp_unset_lock(&dataLock);
	if (!poseDiff)
		return false;

	/* ncc or mad
	// ncc
	// required information
	cv::Mat AllprojImg = cv::Mat::zeros(rows, cols, CV_8UC1);		
	cv::Mat Allmask    = cv::Mat::zeros(rows, cols, CV_8UC1);		
	vpHomogeneousMatrix P = cMo;
	vpMatrix PP= cMo;
	vpMatrix K = cam.get_K();
	// project prePatch using the current pose
	int len = vpMbEdgeTracker::faces.getPolygon().size();
	for (int i = 0; i < len; i++)
		if (isVisible[i])
		{
			if (dataPatches[i].empty())
				continue;
			// orgPatch
			patch& orgPatch = dataPatches[i].front();
			int hs = orgPatch.highestConfidenceScale;
			vpMatrix invK = faceScaleInfo[i].Ks[hs].inverseByLU();
			vpMatrix invP = faceScaleInfo[i].cMo.inverseByLU();
			vpMatrix PinvP = PP * invP;
			float dp = faceScaleInfo[i].depth;

			// define the corresponding points
			std::vector<cv::Point2f> orgPoints(4);
			std::vector<cv::Point2f> transPoints(4);
			std::vector<cv::Point>   corners(4);
			// find the four corners in the org rectangle patch
			cv::Size pSize = faceScaleInfo[i].faceSizes[hs];
			for (int i = 0; i < 4; i++)
			{
				int x, y;
				switch (i)
				{
					case 0:
						x = 0;
						y = 0;
						break;
					case 1:
						x = 0;
						y = pSize.height - 1;
						break;
					case 2:
						x = pSize.width  - 1;
						y = pSize.height - 1;
						break;
					case 3:
						x = pSize.width  - 1;
						y = 0;
						break;
					default:
						break;
				}

				// project the corners to the target quadrangle patch
				int xc, yc;
				float Z;
				projPoint(invK, PinvP, K, dp, x, y, xc, yc, Z);

				// save the corresponding points
				orgPoints[i]   = cv::Point2f(x, y);
				transPoints[i] = cv::Point2f(xc, yc);
				corners[i]     = cv::Point(xc, yc);
			}
			cv::Mat projImg = cv::Mat::zeros(rows, cols, CV_8UC1);		
			cv::Mat mask    = cv::Mat::zeros(rows, cols, CV_8UC1);		
			// fill the mask
			for (size_t i = 0; i < 4; i++)
				cv::line(mask, transPoints[i], transPoints[(i+1) % 4], 255);
			cv::floodFill(mask, meanPoint(corners), 255);

			// find the perspective transform
			cv::Mat warp_matrix = cv::getPerspectiveTransform(orgPoints, transPoints);
			// perspective transform the patch
			cv::warpPerspective(orgPatch.scaledPatch[hs], projImg, warp_matrix, cv::Size(cols, rows));
			AllprojImg += projImg;
			Allmask    += mask;
		}

	//DEBUG only
	//cv::imshow("projImg", AllprojImg);
	//cv::waitKey();
	//end of DEBUG
	
	// calculate the MAD
	//double score = CorrNormedWithMask(AllprojImg, curImg, Allmask);
	double score = MADWithMask(AllprojImg, curImg, Allmask);
	std::cout<<"MAD = "<<score<<std::endl;
	ncc = score < th3 ? true : false;

	end of ncc or mad*/

	return frameDist & poseDiff & matchness & ncc;
}

void 
superResolutionTracker::initialization(cv::Mat& src, std::string config_file, std::string modelName, std::string initName)
{
	cv::cvtColor(src, curImg, CV_RGB2GRAY);	
	loadConfigFile(config_file);
	virtualCam = getVirtualCam();
	invVirtualCam = virtualCam.inverseByLU();
	// Load the 3d model in cao format. No 3rd party library is required
	loadModel(modelName); 

	// Initialise manually the pose by clicking on the image points associated to the 3d points contained in the cube.init file.
	vpImageConvert::convert(curImg,I);
	disp.init(I);
	initClick(I, initName); 

	// initialize the member variables
	rows = curImg.rows;
	cols = curImg.cols;

	for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
		isVisible.push_back(false);
	// once the model has been read, we can init this important structure
	initFaceScaleInfo();

	// for first frame tracking
	int i = 0;
	for (std::vector<vpMbtPolygon*>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin(); itr != vpMbEdgeTracker::faces.getPolygon().end(); ++itr)
	{
		isVisible[i] = (*itr)->isVisible(cMo, vpMath::rad(faceAngle));
		++i;
	}
	for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
	{
		prePatch[i].clear();
		if (isVisible[i])
		{
			patch p;
			obtainPatch(i, p);
			prePatch[i].push_back(p);
			// the only absolute information is the first frame initialized manually
			patch pb;
			deepCopyPrePatch(p, pb);
			pushDataIntoBuff(pb);
		}
	}
}

void
superResolutionTracker::initFaceScaleInfo(void)
{
	// for faceSizes
	initFaceScaleSize();

	// for virtual camera
	initFaceScaleVirtualCam();

	// for pose
	initFaceScalePose();

	// for depth
	initFaceScaleDepth();
}

void
superResolutionTracker::initFaceScaleDepth(void)
{
	int i = 0;
	for (std::vector<vpMbtPolygon*>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin(); itr != vpMbEdgeTracker::faces.getPolygon().end(); ++itr)
	{
		vpPoint p = (*itr)->p[0];
		p.changeFrame(faceScaleInfo[i].cMo);
		faceScaleInfo[i].depth = p.get_Z();
		++i;
	}
}

void
superResolutionTracker::initFaceScalePose(void)
{
	int i = 0;
	for (std::vector<vpMbtPolygon*>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin(); itr != vpMbEdgeTracker::faces.getPolygon().end(); ++itr)
	{
		vpPose pose;
		// 4 points are enough for computing the pose
		for (int j = 0; j < 4; j++)
		{
			vpPoint p = (*itr)->p[j];
			double x, y;
			double u, v;
			switch (j)
			{
				case 0:
					u = 0;
					v = 0;
					break;
				case 1:
					u = 0;
					v = faceScaleInfo[i].faceSizes[numOfPatchScale-1].height;
					break;                                        
				case 2:                                           
					u = faceScaleInfo[i].faceSizes[numOfPatchScale-1].width;
					v = faceScaleInfo[i].faceSizes[numOfPatchScale-1].height;
					break;                                        
				case 3:                                           
					u = faceScaleInfo[i].faceSizes[numOfPatchScale-1].width;
					v = 0;
					break;
				default:
					break;
			}
			vpPixelMeterConversion::convertPoint(cam, u, v, x, y);
			p.set_x(x);
			p.set_y(y);
			p.set_w(1);
			pose.addPoint(p);
		}
		pose.computePose(vpPose::DEMENTHON_VIRTUAL_VS, faceScaleInfo[i].cMo);

		++i;
	}
}

void
superResolutionTracker::initFaceScaleSize(void)
{
	float rate = getRate(maxDownScale, numOfPatchScale);
	float aspectRatio = (float) cols / (float) rows;

	int i = 0;
	for (std::vector<vpMbtPolygon*>::iterator itr = vpMbEdgeTracker::faces.getPolygon().begin(); itr != vpMbEdgeTracker::faces.getPolygon().end(); ++itr)
	{
		// width / height
		float height = pointDistance3D((*itr)->p[0], (*itr)->p[1]);
		float width  = pointDistance3D((*itr)->p[1], (*itr)->p[2]);
		float curAR = width / height;
		if (curAR > aspectRatio)
		{
			faceScaleInfo[i].faceSizes[numOfPatchScale-1].width = cols;
			faceScaleInfo[i].faceSizes[numOfPatchScale-1].height = cols / curAR;
		}
		else
		{
			faceScaleInfo[i].faceSizes[numOfPatchScale-1].height = rows;
			faceScaleInfo[i].faceSizes[numOfPatchScale-1].width = rows * curAR;
		}

		int maxHeight = faceScaleInfo[i].faceSizes[numOfPatchScale-1].height;
		int maxWidth = faceScaleInfo[i].faceSizes[numOfPatchScale-1].width;

		for (int j = 0; j < numOfPatchScale - 1; j++)
		{
			faceScaleInfo[i].faceSizes[j].width = maxWidth / pow(rate, numOfPatchScale -1 - j);
			faceScaleInfo[i].faceSizes[j].height = maxHeight / pow(rate, numOfPatchScale -1 - j);
		}

		++i;
	}
}

void
superResolutionTracker::initFaceScaleVirtualCam(void)
{
	float rate = getRate(maxDownScale, numOfPatchScale);
	for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
	{
		faceScaleInfo[i].Ks[numOfPatchScale-1] = cam.get_K(); 
		for (int j = 0; j < numOfPatchScale-1; j++)
			faceScaleInfo[i].Ks[j] = scaleCam(cam.get_K(), pow(rate, numOfPatchScale -1 - j)); 
	}
}

vpMatrix
superResolutionTracker::scaleCam(vpMatrix K, float rate)
{
	K[0][0] /= rate;
	K[0][2] /= rate;
	K[1][1] /= rate;
	K[1][2] /= rate;

	return K;
}

	inline float 
superResolutionTracker::pointDistance3D(const vpPoint& p1, const vpPoint& p2)
{
	float x1 = p1.get_oX();
	float y1 = p1.get_oY();
	float z1 = p1.get_oZ();
	float x2 = p2.get_oX();
	float y2 = p2.get_oY();
	float z2 = p2.get_oZ();
	float dist = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
	return dist;
}

	float
superResolutionTracker::calcDepth(
		vpMatrix& A,
		vpMatrix& b,
		float* coefficient,
		vpMatrix invP,
		vpMatrix invK,
		cv::Point cp
		)
{
	// NOTE: only the first column of A can be changed here
	//
	vpMatrix X(3, 1);
	X[0][0] = cp.x;
	X[1][0] = cp.y;
	X[2][0] = 1;
	X = invK * X;

	float detA = coefficient[0] * X[0][0] 
		+ coefficient[1] * X[2][0] 
		+ coefficient[2] * X[1][0] 
		- coefficient[3] * X[0][0] 
		- coefficient[4] * X[1][0] 
		- coefficient[5] * X[2][0];

	// for detx
	float detx = coefficient[0] * b[0][0] 
		+ coefficient[1] * b[2][0] 
		+ coefficient[2] * b[1][0] 
		- coefficient[3] * b[0][0] 
		- coefficient[4] * b[1][0] 
		- coefficient[5] * b[2][0];

	float depth = detx / detA;

	return depth;
}

	void
superResolutionTracker::prepCalcDepth(
		vpPoint*  vp, 
		vpMatrix& A, 
		vpMatrix& b,
		float* coefficient)
{
	// fill b
	b[0][0] = -vp[1].get_X() + vp[0].get_X() + vp[2].get_X();
	b[1][0] = -vp[1].get_Y() + vp[0].get_Y() + vp[2].get_Y();
	b[2][0] = -vp[1].get_Z() + vp[0].get_Z() + vp[2].get_Z();
	// fill A
	// second column
	A[0][1] = -(vp[1].get_X() - vp[0].get_X());
	A[1][1] = -(vp[1].get_Y() - vp[0].get_Y());
	A[2][1] = -(vp[1].get_Z() - vp[0].get_Z());
	// third column
	A[0][2] = -(vp[1].get_X() - vp[2].get_X());
	A[1][2] = -(vp[1].get_Y() - vp[2].get_Y());
	A[2][2] = -(vp[1].get_Z() - vp[2].get_Z());

	// pre-compute the coefficient for performance
	//	a11a22
	//  a01a12
	//  a02a21
	//  a12a21
	//  a01a22		
	//  a02a11
	coefficient[0] = A[1][1] * A[2][2];
	coefficient[1] = A[0][1] * A[1][2];
	coefficient[2] = A[0][2] * A[2][1];
	coefficient[3] = A[1][2] * A[2][1];
	coefficient[4] = A[0][1] * A[2][2];
	coefficient[5] = A[0][2] * A[1][1];
}

void
superResolutionTracker::findMinMax(std::vector<cv::Point> corners, cv::Point& lu, cv::Point& rb)
{
	lu = corners[0];
	rb = corners[0];
	for (size_t i = 1; i < corners.size(); i++)
	{
		if (lu.x > corners[i].x)
			lu.x = corners[i].x;
		if (lu.y > corners[i].y)
			lu.y = corners[i].y;
		if (rb.x < corners[i].x)
			rb.x = corners[i].x;
		if (rb.y < corners[i].y)
			rb.y = corners[i].y;
	}
}

bool 
comparator(const std::pair<float, int>& l, const std::pair<float, int>& r)
{ 
	return l.first < r.first; 
}

void 
superResolutionTracker::findStableFeatures(
		std::vector<bool>& 		    		finalStatus, 
		const std::vector<cv::Point2f>& 	corners, 
		const std::vector<cv::Point2f>& 	bFeatures,
		const std::vector<uchar>& 			fStatus, 
		const std::vector<uchar>& 			bStatus,
		const std::vector<float>& 			fErr,
		float 								th,
		float 								rate
		)
{
	for (size_t i = 0; i < corners.size(); i++)
	{
		if (fStatus[i] && bStatus[i])
		{
			double dist = pointDistance2D(corners[i], bFeatures[i]);;
			if (dist < th)
				finalStatus[i] = true;
			else
				finalStatus[i] = false;
		}
	}

	int len = fErr.size();
	std::vector<std::pair<float, int> > pErr(len);

#pragma omp parallel for num_threads(2)
	for (int i = 0; i < len; i++)
	{
		pErr[i].first = fErr[i];
		pErr[i].second = i;
	}

	std::sort(pErr.begin(), pErr.end(), comparator);

	for (int i = 0, count = 0; i < len; i++)
	{
		if (count < rate * len)
			if (fStatus[pErr[i].second] == 1 && fStatus[pErr[i].second] == 1 && finalStatus[pErr[i].second])
			{
				++count;
				finalStatus[pErr[i].second] = true;
			}
			else
				finalStatus[pErr[i].second] = false;
		else
			finalStatus[pErr[i].second] = false;
	}
}

void
superResolutionTracker::findStableFeaturesWithRate(
		const std::vector<float>& 	 	fErr, 
		const std::vector<uchar>& 	 	fStatus, 
		std::vector<bool>& 			 	finalStatus, 
		const std::vector<cv::Point2f>& corners,
		const std::vector<cv::Point2f>& cFeatures,
		float 						 	rate)
{
	int len = fErr.size();
	std::vector<std::pair<float, int> > pErr(len);

	double dist = 0;
	double maxRate = 3; // the actual rate is sqrt(3), since we don't take the square root while calculating the distance
	std::vector<double> pd;
	for (int i = 0; i < len; i++)
		if (fStatus[i] == 1)
			pd.push_back(pointDistance2D(corners[i], cFeatures[i]));
	double minD;
	if (pd.empty())
	{
		for (int i = 0; i < len; i++)
			finalStatus[i] = false;
		return;
	}
	else
	{
		size_t n = pd.size() / 2;
		std::nth_element(pd.begin(), pd.begin()+n, pd.end());
		minD = (*std::min_element(pd.begin(), pd.end()) + 1) * maxRate; // avoid the zero distance
	}
	// DEBUG only
	//std::cout<<"maxD = "<<maxD<<std::endl;
	// END OF DEBUG


#pragma omp parallel for num_threads(2)
	for (int i = 0; i < len; i++)
	{
		pErr[i].first = fErr[i];
		pErr[i].second = i;
	}

	std::sort(pErr.begin(), pErr.end(), comparator);

	for (int i = 0, count = 0; i < len; i++)
	{
		if ((count < rate * len) && (fStatus[pErr[i].second] == 1) && (pointDistance2D(corners[pErr[i].second], cFeatures[pErr[i].second]) < minD))
		{
			++count;
			finalStatus[pErr[i].second] = true;
		}
		else
			finalStatus[pErr[i].second] = false;
	}
}

void
superResolutionTracker::findClosestPatch(
		vpPoseVector& 		vp, 
		std::vector<int>& 	id, 
		std::vector<int>& 	idx, 
		int 				numOfPatchesUsed,
		int 				faceID)
{
	int len = id.size();
	std::vector<float> diff(len);
#pragma omp parallel for num_threads(4)
	for (int j = 0; j < len; j++)
	{
		std::list<patch>::iterator itr = dataset[faceID].begin();
		for (int k = 0; k < id[j] ; k++)
			++itr;
		vpPoseVector p;
		p.buildFrom(itr->pose);
		diff[j] = fabs(p[0] - vp[0])
			+ fabs(p[1] - vp[1])
			+ fabs(p[2] - vp[2]);
	}

	for (int i = 0; i < numOfPatchesUsed; i++)
	{
		float min = 1e5;
		int minIdx = 0;
		for (int j = 0; j < len; j++)
			if (min > diff[j])
			{
				min = diff[j];
				idx[i] = id[j];
				minIdx = j;
			}
		diff[minIdx] = 1e5;
	}
}

	inline int 
superResolutionTracker::findMinCost(float tar, int pos, const cv::Mat& img , int rows, int cols, int ws)
{
	int r = pos / cols;
	int c = pos % cols;
	cv::Mat subMat = img(
			cv::Range(std::max(r-ws, 0), std::min(r+ws+1, rows)), 
			cv::Range(std::max(c-ws, 0), std::min(c+ws+1, cols))
			).clone();
	if (tar > 255)
		tar = 255;
	subMat = cv::abs(subMat - (uchar)tar);	
	return *std::min_element(subMat.begin<uchar>(), subMat.end<uchar>());
}

// virtual inherit the vpMbTracker class, so only one cMo is available here
// 90% of the codes here are copied from the vpMbEdgeKltTracker class
double
superResolutionTracker::computeVVS(
		unsigned int _nbInfos,
		const vpImage<unsigned char>& I, 
		vpPoseFeatures& pf,
		vpColVector &w_mbt, 
		vpColVector &w_klt, 
		float scale_,
		const unsigned int lvl)
{
	vpColVector factor;
	unsigned int nbrow = trackFirstLoop(I, factor, lvl);

	// for the klt tracker
	vpMatrix J_mbt, J_klt; // interaction matrix
	vpColVector R_mbt, R_klt; // residu
	pf.error_and_interaction(cMo, R_klt, J_klt);
	int nbInfos_ = R_klt.size() / 2;
	int nbInfos  = nbInfos_ + _nbInfos;

	if(nbrow < 4 && nbInfos < 4)
	{
		vpERROR_TRACE("\n\t\t Error-> not enough data") ;
		throw vpTrackingException(vpTrackingException::notEnoughPointError, "\n\t\t Error-> not enough data");
	}
	else if(nbrow < 4)
		nbrow = 0;

	double residu = 0;
	double residu_1 = -1;
	unsigned int iter = 0;

	vpMatrix *J;
	vpColVector *R;
	vpMatrix J_true;
	vpColVector R_true;
	vpColVector w_true;

	if(nbrow != 0)
	{
		J_mbt.resize(nbrow,6);
		R_mbt.resize(nbrow);
	}

	if(nbInfos != 0){
		J_klt.resize(2*nbInfos,6);
		R_klt.resize(2*nbInfos);
	}

	vpColVector w; // weight from MEstimator
	vpColVector v; // "speed" for VVS
	vpRobust robust_mbt(0), robust_klt(0);
	vpHomography H;

	vpMatrix JTJ, JTR;

	double factorMBT = 0.35;
	double factorKLT = 0.65;

	//More efficient weight repartition for hybrid tracker should come soon...
	//factorMBT = 1.0 - (double)nbrow / (double)(nbrow + nbInfos);
	//factorKLT = 1.0 - factorMBT;
	//if (scale_ < 4)
	//{
	//	factorMBT = 0.2;
	//	factorKLT = 0.8;
	//}
	//else
	//{
	//	factorMBT = 0.1;
	//	factorKLT = 0.9;
	//}
	// for DEBUG only

	//std::cout<<"scale_ = "<<scale_<<std::endl;
	// END OF DEBUG

	double residuMBT = 0;
	double residuKLT = 0;

	while( ((int)((residu - residu_1)*1e8) !=0 ) && (iter<maxIter) )
	{
		J = new vpMatrix();
		R = new vpColVector();

		if(nbrow >= 4)
			trackSecondLoop(I,J_mbt,R_mbt,cMo,lvl);

		if(nbInfos >= 4)
		{
			unsigned int shift = 0;
			for (unsigned int i = 0; i < vpMbKltTracker::faces.size(); i++)
				if(vpMbKltTracker::faces[i]->isVisible() && vpMbKltTracker::faces[i]->hasEnoughPoints())
				{
					vpSubColVector subR(R_klt, shift, 2*vpMbKltTracker::faces[i]->getNbPointsCur());
					vpSubMatrix subJ(J_klt, shift, 0, 2*vpMbKltTracker::faces[i]->getNbPointsCur(), 6);
					vpMbKltTracker::faces[i]->computeHomography(ctTc0, H);
					vpMbKltTracker::faces[i]->computeInteractionMatrixAndResidu(subR, subJ);
					shift += 2*vpMbKltTracker::faces[i]->getNbPointsCur();
				}
			// the points obtained from patches
			vpSubColVector subR(R_klt, 2*_nbInfos, 2*nbInfos_), sR;
			vpSubMatrix subJ(J_klt, 2*_nbInfos, 0, 2*nbInfos_, 6), sJ;
			pf.error_and_interaction(cMo, sR, sJ);
			subR = sR;
			subJ = sJ;
		}

		if(iter == 0)
		{
			w.resize(nbrow + 2*nbInfos);
			w=1;

			w_mbt.resize(nbrow);
			w_mbt = 1;
			robust_mbt.resize(nbrow);

			w_klt.resize(2*nbInfos);
			w_klt = 1;
			robust_klt.resize(2*nbInfos);

			w_true.resize(nbrow + 2*nbInfos);
		}

		/* robust */
		if(nbrow > 3)
		{
			residuMBT = 0;
			for(unsigned int i = 0; i < R_mbt.getRows(); i++)
				residuMBT += fabs(R_mbt[i]);
			residuMBT /= R_mbt.getRows();

			robust_mbt.setIteration(iter);
			robust_mbt.setThreshold(thresholdMBT/cam.get_px());
			robust_mbt.MEstimator( vpRobust::TUKEY, R_mbt, w_mbt);
			J->stackMatrices(J_mbt);
			R->stackMatrices(R_mbt);
		}

		if(nbInfos > 3)
		{
			residuKLT = 0;
			for(unsigned int i = 0; i < R_klt.getRows(); i++)
				residuKLT += fabs(R_klt[i]);
			residuKLT /= R_klt.getRows();

			robust_klt.setIteration(iter);
			robust_klt.setThreshold(thresholdKLT/cam.get_px());
			robust_klt.MEstimator( vpRobust::TUKEY, R_klt, w_klt);

			J->stackMatrices(J_klt);
			R->stackMatrices(R_klt);
		}

		unsigned int cpt = 0;
		while(cpt< (nbrow+2*nbInfos))
		{
			if(cpt<(unsigned)nbrow)
				w[cpt] = ((w_mbt[cpt] * factor[cpt]) * factorMBT) ;
			else
				w[cpt] = (w_klt[cpt-nbrow] * factorKLT);
			cpt++;
		}

		if(computeCovariance){
			R_true = (*R);
			J_true = (*J);
		}

		residu_1 = residu;
		residu = 0;
		double num = 0;
		double den = 0;
		for (unsigned int i = 0; i < static_cast<unsigned int>(R->getRows()); i++){
			num += w[i]*vpMath::sqr((*R)[i]);
			den += w[i];

			w_true[i] = w[i]*w[i];
			(*R)[i] *= w[i];
			if(compute_interaction){
				for (unsigned int j = 0; j < 6; j += 1){
					(*J)[i][j] *= w[i];
				}
			}
		}

		residu = sqrt(num/den);

		JTJ = J->AtA();
		computeJTR(*J, *R, JTR);
		v = -lambda * JTJ.pseudoInverse() * JTR;
		cMo = vpExponentialMap::direct(v).inverse() * cMo;
		ctTc0 = vpExponentialMap::direct(v).inverse() * ctTc0;

		iter++;

		delete J;
		delete R;
	}

	if(computeCovariance){
		vpMatrix D;
		D.diag(w_true);
		covarianceMatrix = vpMatrix::computeCovarianceMatrix(J_true,v,-lambda*R_true,D);
	}

	return residu;
}

// 90% of the codes of this function are copied from the vpMbEdgeKltTracker class
double
superResolutionTracker::getPose(const vpImage<unsigned char>& I, float scale_, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches)
{
	double res;
	unsigned int nbInfos;
	vpColVector w_klt;
	vpColVector w_mbt;

	unsigned int nbFaceUsed;
	vpMbKltTracker::preTracking(I, nbInfos, nbFaceUsed);

	if(nbInfos >= 4)
		vpMbKltTracker::computeVVS(nbInfos, w_klt);
	else
	{
		nbInfos = 0;
		// std::cout << "[Warning] Unable to init with KLT" << std::endl;
	}

	vpPoseFeatures pf;
	trackPatch(pf, img, prePatch, dataPatches);	
	//trackPatchOrb(pf, img, prePatch, dataPatches);	
	//trackPatchSIFT(pf, img, prePatch, dataPatches);	
	//trackPatchSURF(pf, img, prePatch, dataPatches);	

	vpMbEdgeTracker::trackMovingEdge(I);

	res = computeVVS(nbInfos, I, pf, w_mbt, w_klt, scale_);
	vpSubColVector sub_w_klt(w_klt, 0, 2*nbInfos);

	if(postTracking(I, w_mbt, sub_w_klt))
	{
		vpMbKltTracker::reinit(I);

		initPyramid(I, Ipyramid);

		unsigned int n = 0;
		for(unsigned int i = 0; i < vpMbKltTracker::faces.size() ; i++)
			if(vpMbKltTracker::faces[i]->isVisible())
			{
				vpMbEdgeTracker::faces[i]->isvisible = true;
				n++;
			}
			else
				vpMbEdgeTracker::faces[i]->isvisible = false;

		vpMbEdgeTracker::nbvisiblepolygone = n;

		unsigned int i = (unsigned int)scales.size();
		do {
			i--;
			if(scales[i]){
				downScale(i);
				initMovingEdge(*Ipyramid[i], cMo);
				upScale(i);
			}
		} while(i != 0);

		cleanPyramid(Ipyramid);
	}
	return res;
}


/**
 * @brief for efficiency, no square root is computed here
 *
 * @param p1
 * @param p2
 *
 * @return 
 */
inline double
superResolutionTracker::pointDistance2D(const cv::Point& p1, const cv::Point& p2)
{
	double dx = p1.x - p2.x;
	double dy = p1.y - p2.y;
	//return std::sqrt(dx * dx + dy * dy);
	return (dx * dx + dy * dy);
}

void
superResolutionTracker::trackPatch(vpPoseFeatures& featuresComputePose, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches)
{
	//vpMbEdgeKltTracker::track(I);

	// build the image pyramid for tracking
	std::vector<cv::Mat> cPyramid;
	std::vector<cv::Mat> upScaleCPyramid;
	std::vector<cv::Mat> pPyramid;
#pragma omp parallel sections
	{
#pragma omp section
		cv::buildOpticalFlowPyramid(curImg, cPyramid, cv::Size(21, 21), 3);
#pragma omp section
		cv::buildOpticalFlowPyramid(img, upScaleCPyramid, cv::Size(winSize, winSize), maxLevel);
#pragma omp section
		cv::buildOpticalFlowPyramid(preImg, pPyramid, cv::Size(21, 21), 3);
	}

	vpMatrix invP = cMo.inverseByLU();

	// erode the mask so the feature points detected will not be at the border
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15), cv::Point(7, 7)); 

	// for better parallel
	std::list<vpPoint> pSetsOne;
	std::list<vpPoint> pSetsTwo;
#pragma omp parallel sections
	{
#pragma omp section
		{
			// for pre-frame
			vpMatrix invK = cam.get_K().inverseByLU();
			// for pre-frame
			for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
				if (isVisible[i])
					if(!prePatch[i].empty())
					{
						patch& pp = prePatch[i].front();
						// restore the patch to the image size
						cv::Mat mask 	 = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						cv::erode(pp.mask, pp.mask, element);
						pp.mask.copyTo(
								mask(
									cv::Range(pp.patchRect.y, pp.patchRect.y + pp.patchRect.height),
									cv::Range(pp.patchRect.x, pp.patchRect.x + pp.patchRect.width)
									)
								);
						// detect good features in pre-patch
						std::vector<cv::Point2f> corners;
						cv::goodFeaturesToTrack(preImg, corners, 50, 0.01, 5, mask);
						// DEBUG only
						//cv::imshow("mask", mask);
						//cv::Mat pImg = orgPatch.clone();
						//for (size_t j = 0; j < corners.size(); j++)
						//	cv::circle(pImg, corners[j], 3, cv::Scalar(255, 0, 0));
						//cv::imshow("pre-features", pImg);
						// END OF DEBUG
						// if not corners are detection in this face, then we won't do the klt tracking procedure, and simply move to another face
						if (corners.empty())
							continue;

						// KLT search them in cur-image
						std::vector<cv::Point2f> cFeatures, bFeatures;
						std::vector<float> fErr, bErr;
						std::vector<uchar> fStatus, bStatus;
						// forward-backward track
						cv::calcOpticalFlowPyrLK(pPyramid, cPyramid, corners, cFeatures, fStatus, fErr, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 20, 0.03));
						cv::calcOpticalFlowPyrLK(cPyramid, pPyramid, cFeatures, bFeatures, bStatus, bErr, cv::Size(21, 21), 3, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 20, 0.03));
						std::vector<bool> finalStatus(corners.size());
						float th = 0.5 * 0.5; // the threshold of distance is 0.5
						float rate = 0.2;
						findStableFeatures(finalStatus, corners, bFeatures, fStatus, bStatus, fErr, th, rate);
						// DEBUG only
						//cv::Mat cImg = curImg.clone();
						//for (size_t j = 0; j < cFeatures.size(); j++)
						//	if (finalStatus[j]) // only the tracked features are used
						//		cv::circle(cImg, cFeatures[j], 3, cv::Scalar(255, 0, 0));
						//cv::imshow("cur features", cImg);
						//cv::waitKey(30);
						// END OF DEBUG

						// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
						// push back to compute pose
						int dx = pp.patchRect.x;
						int dy = pp.patchRect.y;
						for (size_t j = 0; j < corners.size(); j++)
							if (finalStatus[j]) // only the forward-backward stable features are used
							{
								vpPoint p;	
								// 2d point
								double u, v;
								vpPixelMeterConversion::convertPoint(cam, cFeatures[j].x, cFeatures[j].y, u, v);
								p.set_x(u);
								p.set_y(v);
								p.set_w(1);
								// 3d point
								float depth = pp.depth.at<float>(corners[j].y - dy, corners[j].x - dx);
								if (depth < 1e-5)
									continue;
								backProj(invK, invP, depth, corners[j].x, corners[j].y, p);
								pSetsOne.push_back(p);
							}
					}
		}

#pragma omp section
		{
			// erode the mask so the feature points detected will not be at the border
			cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15), cv::Point(7, 7)); 
			// for dataset
			vpMatrix invVirtualK = invVirtualCam;
			//
			// detect good features in dataPatches
			// virtual camera is used here
			vpCameraParameters vc;
			vc.initFromCalibrationMatrix(virtualCam);
			for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
				if (isVisible[i])
					for (std::list<patch>::iterator pp = dataPatches[i].begin(); pp != dataPatches[i].end(); ++pp)
					{
						// detect good features in pp-patch
						// corners: features in previous frame
						std::vector<cv::Point2f> corners;
						cv::erode(pp->mask, pp->mask, element);
						cv::goodFeaturesToTrack(pp->orgPatch, corners, 50, 0.01, 5, pp->mask);
						if (corners.empty())
							continue;
						// DEBUG only
						//cv::Mat pImg = pp->orgPatch.clone();
						//for (size_t j = 0; j < corners.size(); j++)
						//	cv::circle(pImg, corners[j], 3, cv::Scalar(255, 0, 0));
						//cv::imshow("orgPatch", pImg);
						// END OF DEBUG

						// KLT search them in cur-image
						std::vector<cv::Mat> prePatchPyr;
						std::vector<cv::Point2f> cFeatures;
						std::vector<float> fErr;
						std::vector<uchar> fStatus;
						cv::buildOpticalFlowPyramid(pp->orgPatch, prePatchPyr, cv::Size(winSize, winSize), maxLevel);
						cv::calcOpticalFlowPyrLK(prePatchPyr, upScaleCPyramid, corners, cFeatures, fStatus, fErr, cv::Size(winSize, winSize), maxLevel, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 20, 0.05));
						float rate = 3;
						std::vector<bool> finalStatus(corners.size());
						findStableFeaturesWithRate(fErr, fStatus, finalStatus, corners, cFeatures, rate);
						// DEBUG only
						//cv::Mat cImg = img.clone();
						//for (size_t j = 0; j < cFeatures.size(); j++)
						//	if (finalStatus[j]) // only the tracked features are used
						//		cv::circle(cImg, cFeatures[j], 3, cv::Scalar(255, 0, 0));
						//cv::imshow("patch features", cImg);
						//cv::waitKey(30);
						// END OF DEBUG

						// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
						// push back to compute pose
						for (size_t j = 0; j < corners.size(); j++)
							if (finalStatus[j])
							{
								vpPoint p;	
								// 2d point
								double u, v;
								vpPixelMeterConversion::convertPoint(vc, cFeatures[j].x,cFeatures[j].y,u,v);
								p.set_x(u);
								p.set_y(v);
								p.set_w(1);
								// 3d point
								float depth = pp->depth.at<float>(corners[j].y, corners[j].x);
								if (depth < 1e-5)
									continue;
								backProj(invVirtualK, invP, depth, corners[j].x, corners[j].y, p);
								pSetsTwo.push_back(p);
							}
					}
		}
	}
	for (std::list<vpPoint>::iterator itr = pSetsOne.begin(); itr != pSetsOne.end(); ++itr)
		featuresComputePose.addFeaturePoint(*itr);
	for (std::list<vpPoint>::iterator itr = pSetsTwo.begin(); itr != pSetsTwo.end(); ++itr)
		featuresComputePose.addFeaturePoint(*itr);
}

// TODO: good_matches not added in this tracker, ref trackPatchSIFT
void
superResolutionTracker::trackPatchOrb(vpPoseFeatures& featuresComputePose, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches)
{
	//vpMbEdgeKltTracker::track(I);

	// erode the mask so the feature points detected will not be at the border
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15), cv::Point(7, 7)); 

	vpMatrix invP = cMo.inverseByLU();

	// the allowed movement range between two consecutive frames
	int erodeSize = 50;

	// for better parallel
	std::vector<vpPoint> pSetsOne;
	std::vector<vpPoint> pSetsTwo;
#pragma omp parallel sections
	{
#pragma omp section
		{
			// for pre-frame
			vpMatrix invK = cam.get_K().inverseByLU();
			// for pre-frame
			for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
				if (isVisible[i])
					if(!prePatch[i].empty())
					{
						patch& pp = prePatch[i].front();
						// restore the patch to the image size
						cv::Mat orgPatch = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						cv::Mat mask 	 = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						cv::Mat curMask  = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						pp.orgPatch.copyTo(
								orgPatch(
									cv::Range(pp.patchRect.y, pp.patchRect.y + pp.patchRect.height),
									cv::Range(pp.patchRect.x, pp.patchRect.x + pp.patchRect.width)
									)
								);
						cv::erode(pp.mask, pp.mask, element);
						pp.mask.copyTo(
								mask(
									cv::Range(pp.patchRect.y, pp.patchRect.y + pp.patchRect.height),
									cv::Range(pp.patchRect.x, pp.patchRect.x + pp.patchRect.width)
									)
								);
						// generate the mask for current image, since we don't know where the object is, the mask should be relatively large
						cv::Point upLeft, rightbottom;
						findCorner(mask, upLeft, rightbottom);
						genMask(curMask, upLeft, rightbottom, erodeSize);
						
						// Orb detection and matching
						// ORB::ORB(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)
						// init the detector
						cv::ORB orb(100, 1.2f, 3, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);  
						std::vector<cv::KeyPoint> keyPoints_1, keyPoints_2;  
						cv::Mat descriptors_1, descriptors_2;  
						// detect
						orb(orgPatch,  mask, keyPoints_1, descriptors_1);  
						orb(curImg, curMask, keyPoints_2, descriptors_2);  
						// match
						cv::BruteForceMatcher<cv::HammingLUT> matcher;  
						std::vector<cv::DMatch> matches;  
						matcher.match(descriptors_1, descriptors_2, matches);  

						// DEBUG only
						cv::Mat img_matches;  
						cv::drawMatches(orgPatch, keyPoints_1, curImg, keyPoints_2,  
								matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),  
								std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
						cv::imshow("Curmask", curMask);
						cv::imshow( "Match", img_matches);  
						cv::waitKey();  
						// end of DEBUG

						// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
						// push back to compute pose
						// extract the matched points
						int dx = pp.patchRect.x;
						int dy = pp.patchRect.y;
						for (size_t j = 0; j < matches.size(); j++)
						{
							// extract the two 2d points
							float px = keyPoints_1[matches[j].queryIdx].pt.x;
							float py = keyPoints_1[matches[j].queryIdx].pt.y;
							float cx = keyPoints_2[matches[j].queryIdx].pt.x;
							float cy = keyPoints_2[matches[j].queryIdx].pt.y;
							
							vpPoint p;	
							// 2d point in current frame
							double u, v;
							vpPixelMeterConversion::convertPoint(cam, cx, cy, u, v);
							p.set_x(u);
							p.set_y(v);
							p.set_w(1);
							// 3d point in previous frame
							float depth = pp.depth.at<float>(py - dy, px - dx);
							if (depth < 1e-5)
								continue;
							backProj(invK, invP, depth, px, py, p);
							pSetsOne.push_back(p);
						}
					}
		}

#pragma omp section
		{
			// for dataset
			vpMatrix invVirtualK = invVirtualCam;
			//
			// detect good features in dataPatches
			// virtual camera is used here
			vpCameraParameters vc;
			vc.initFromCalibrationMatrix(virtualCam);
			for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
				if (isVisible[i])
					for (std::list<patch>::iterator pp = dataPatches[i].begin(); pp != dataPatches[i].end(); ++pp)
					{
						cv::erode(pp->mask, pp->mask, element);
						cv::ORB orb(100, 1.2f, 3, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);  
						std::vector<cv::KeyPoint> keyPoints_1, keyPoints_2;  
						cv::Mat descriptors_1, descriptors_2;  
						// detect
						orb(pp->orgPatch, pp->mask, keyPoints_1, descriptors_1);  
						// TODO: check the scale of patches here
						cv::Mat curMask = pp->mask.clone();
						// generate the mask for current image, since we don't know where the object is, the mask should be relatively large
						cv::Point upLeft, rightbottom;
						findCorner(pp->mask, upLeft, rightbottom);
						genMask(curMask, upLeft, rightbottom, erodeSize);
						orb(img, curMask, keyPoints_2, descriptors_2);  
						// match
						cv::BruteForceMatcher<cv::HammingLUT> matcher;  
						std::vector<cv::DMatch> matches;  
						matcher.match(descriptors_1, descriptors_2, matches);  
						
						// DEBUG only
						cv::Mat img_matches;  
						cv::drawMatches(pp->orgPatch, keyPoints_1, img, keyPoints_2,  
								matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),  
								std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
						cv::imshow( "Match", img_matches);  
						cv::imshow("Curmask", curMask);
						cv::waitKey();  
						// end of DEBUG


						// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
						// push back to compute pose
						for (size_t j = 0; j < matches.size(); j++)
							{
							// extract the two 2d points
							float px = keyPoints_1[matches[j].queryIdx].pt.x;
							float py = keyPoints_1[matches[j].queryIdx].pt.y;
							float cx = keyPoints_2[matches[j].queryIdx].pt.x;
							float cy = keyPoints_2[matches[j].queryIdx].pt.y;

								vpPoint p;	
								// 2d point
								double u, v;
								vpPixelMeterConversion::convertPoint(vc, cx,cy,u,v);
								p.set_x(u);
								p.set_y(v);
								p.set_w(1);
								// 3d point
								float depth = pp->depth.at<float>(py, px);
								if (depth < 1e-5)
									continue;
								backProj(invVirtualK, invP, depth, px, py, p);
								pSetsTwo.push_back(p);
							}
					}
		}
	}
	for (size_t i = 0; i < pSetsOne.size(); i++)
		featuresComputePose.addFeaturePoint(pSetsOne[i]);
	for (size_t i = 0; i < pSetsTwo.size(); i++)
		featuresComputePose.addFeaturePoint(pSetsTwo[i]);
}

void
superResolutionTracker::trackPatchSIFT(vpPoseFeatures& featuresComputePose, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches)
{
	//vpMbEdgeKltTracker::track(I);

	// erode the mask so the feature points detected will not be at the border
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15), cv::Point(7, 7)); 

	vpMatrix invP = cMo.inverseByLU();

	// the allowed movement range between two consecutive frames
	int erodeSize = 50;

	// for better parallel
	std::vector<vpPoint> pSetsOne;
	std::vector<vpPoint> pSetsTwo;
#pragma omp parallel sections
	{
#pragma omp section
		{
			// for pre-frame
			vpMatrix invK = cam.get_K().inverseByLU();
			// for pre-frame
			for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
				if (isVisible[i])
					if(!prePatch[i].empty())
					{
						patch& pp = prePatch[i].front();
						// restore the patch to the image size
						cv::Mat orgPatch = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						cv::Mat mask 	 = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						cv::Mat curMask  = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						pp.orgPatch.copyTo(
								orgPatch(
									cv::Range(pp.patchRect.y, pp.patchRect.y + pp.patchRect.height),
									cv::Range(pp.patchRect.x, pp.patchRect.x + pp.patchRect.width)
									)
								);
						cv::erode(pp.mask, pp.mask, element);
						pp.mask.copyTo(
								mask(
									cv::Range(pp.patchRect.y, pp.patchRect.y + pp.patchRect.height),
									cv::Range(pp.patchRect.x, pp.patchRect.x + pp.patchRect.width)
									)
								);
						// generate the mask for current image, since we don't know where the object is, the mask should be relatively large
						cv::Point upLeft, rightbottom;
						findCorner(mask, upLeft, rightbottom);
						genMask(curMask, upLeft, rightbottom, erodeSize);
						
						// init the detector
						cv::SIFT sift;
						std::vector<cv::KeyPoint> keyPoints_1, keyPoints_2;  
						cv::Mat descriptors_1, descriptors_2;  
						// detect
						sift(orgPatch,  mask, keyPoints_1, descriptors_1);  
						sift(curImg, curMask, keyPoints_2, descriptors_2);  
						// match
						cv::BruteForceMatcher<cv::L2<float> >  matcher;	
						std::vector<cv::DMatch> matches;  
						matcher.match(descriptors_1, descriptors_2, matches);  

						double max_dist = 0; double min_dist = 100;  
						//-- Quick calculation of max and min distances between keypoints  
						for( int i = 0; i < descriptors_1.rows; i++ )  
						{   
							double dist = matches[i].distance;  
							if( dist < min_dist ) min_dist = dist;  
							if( dist > max_dist ) max_dist = dist;  
						}  
						//printf("-- Max dist : %f \n", max_dist );  
						//printf("-- Min dist : %f \n", min_dist );  
						//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
						//-- PS.- radiusMatch can also be used here.  
						std::vector< cv::DMatch > good_matches;  
						for( int i = 0; i < descriptors_1.rows; i++ )  
						{   
							if( matches[i].distance < 0.3*max_dist )  
							{   
								good_matches.push_back( matches[i]);   
							}  
						}  

						// DEBUG only
						//cv::Mat img_matches;  
						//cv::drawMatches(orgPatch, keyPoints_1, curImg, keyPoints_2,  
						//		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),  
						//		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
						//cv::imshow("Curmask", curMask);
						//cv::imshow( "Match", img_matches);  
						//cv::waitKey();  
						// end of DEBUG

						// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
						// push back to compute pose
						// extract the matched points
						int dx = pp.patchRect.x;
						int dy = pp.patchRect.y;
						for (size_t j = 0; j < good_matches.size(); j++)
						{
							// extract the two 2d points
							float px = keyPoints_1[good_matches[j].queryIdx].pt.x;
							float py = keyPoints_1[good_matches[j].queryIdx].pt.y;
							float cx = keyPoints_2[good_matches[j].queryIdx].pt.x;
							float cy = keyPoints_2[good_matches[j].queryIdx].pt.y;
							
							vpPoint p;	
							// 2d point in current frame
							double u, v;
							vpPixelMeterConversion::convertPoint(cam, cx, cy, u, v);
							p.set_x(u);
							p.set_y(v);
							p.set_w(1);
							// 3d point in previous frame
							float depth = pp.depth.at<float>(py - dy, px - dx);
							if (depth < 1e-5)
								continue;
							backProj(invK, invP, depth, px, py, p);
							pSetsOne.push_back(p);
						}
					}
		}

#pragma omp section
		{
			// for dataset
			vpMatrix invVirtualK = invVirtualCam;
			//
			// detect good features in dataPatches
			// virtual camera is used here
			vpCameraParameters vc;
			vc.initFromCalibrationMatrix(virtualCam);
			for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
				if (isVisible[i])
					for (std::list<patch>::iterator pp = dataPatches[i].begin(); pp != dataPatches[i].end(); ++pp)
					{
						cv::erode(pp->mask, pp->mask, element);
						cv::SIFT sift;
						std::vector<cv::KeyPoint> keyPoints_1, keyPoints_2;  
						cv::Mat descriptors_1, descriptors_2;  
						// detect
						sift(pp->orgPatch, pp->mask, keyPoints_1, descriptors_1);  
						// TODO: check the scale of patches here
						cv::Mat curMask = pp->mask.clone();
						// generate the mask for current image, since we don't know where the object is, the mask should be relatively large
						cv::Point upLeft, rightbottom;
						findCorner(pp->mask, upLeft, rightbottom);
						genMask(curMask, upLeft, rightbottom, erodeSize);
						sift(img, curMask, keyPoints_2, descriptors_2);  
						// match
						cv::BruteForceMatcher<cv::L2<float> >  matcher;
						std::vector<cv::DMatch> matches;  
						matcher.match(descriptors_1, descriptors_2, matches);  

						double max_dist = 0; double min_dist = 100;  
						//-- Quick calculation of max and min distances between keypoints  
						for( int i = 0; i < descriptors_1.rows; i++ )  
						{   
							double dist = matches[i].distance;  
							if( dist < min_dist ) min_dist = dist;  
							if( dist > max_dist ) max_dist = dist;  
						}  
						//printf("-- Max dist : %f \n", max_dist );  
						//printf("-- Min dist : %f \n", min_dist );  
						//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
						//-- PS.- radiusMatch can also be used here.  
						std::vector< cv::DMatch > good_matches;  
						for( int i = 0; i < descriptors_1.rows; i++ )  
						{   
							if( matches[i].distance < 0.3*max_dist )  
							{   
								good_matches.push_back( matches[i]);   
							}  
						}  

						// DEBUG only
						cv::Mat img_matches;  
						cv::drawMatches(pp->orgPatch, keyPoints_1, img, keyPoints_2,  
								good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),  
								std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
						cv::imshow( "Match", img_matches);  
						cv::imshow("Curmask", curMask);
						cv::waitKey();  
						// end of DEBUG


						// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
						// push back to compute pose
						for (size_t j = 0; j < good_matches.size(); j++)
							{
							// extract the two 2d points
							float px = keyPoints_1[good_matches[j].queryIdx].pt.x;
							float py = keyPoints_1[good_matches[j].queryIdx].pt.y;
							float cx = keyPoints_2[good_matches[j].queryIdx].pt.x;
							float cy = keyPoints_2[good_matches[j].queryIdx].pt.y;

								vpPoint p;	
								// 2d point
								double u, v;
								vpPixelMeterConversion::convertPoint(vc, cx,cy,u,v);
								p.set_x(u);
								p.set_y(v);
								p.set_w(1);
								// 3d point
								float depth = pp->depth.at<float>(py, px);
								if (depth < 1e-5)
									continue;
								backProj(invVirtualK, invP, depth, px, py, p);
								pSetsTwo.push_back(p);
							}
					}
		}
	}
	for (size_t i = 0; i < pSetsOne.size(); i++)
		featuresComputePose.addFeaturePoint(pSetsOne[i]);
	for (size_t i = 0; i < pSetsTwo.size(); i++)
		featuresComputePose.addFeaturePoint(pSetsTwo[i]);
}

void
superResolutionTracker::trackPatchSURF(vpPoseFeatures& featuresComputePose, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches)
{
	//vpMbEdgeKltTracker::track(I);

	// erode the mask so the feature points detected will not be at the border
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15), cv::Point(7, 7)); 

	vpMatrix invP = cMo.inverseByLU();

	// the allowed movement range between two consecutive frames
	int erodeSize = 50;

	// for better parallel
	std::list<vpPoint> pSetsOne;
	std::list<vpPoint> pSetsTwo;
#pragma omp parallel sections
	{
#pragma omp section
		{
			// for pre-frame
			vpMatrix invK = cam.get_K().inverseByLU();
			// for pre-frame
			for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
				if (isVisible[i])
					if(!prePatch[i].empty())
					{
						patch& pp = prePatch[i].front();
						// restore the patch to the image size
						cv::Mat orgPatch = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						cv::Mat mask 	 = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						cv::Mat curMask  = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
						pp.orgPatch.copyTo(
								orgPatch(
									cv::Range(pp.patchRect.y, pp.patchRect.y + pp.patchRect.height),
									cv::Range(pp.patchRect.x, pp.patchRect.x + pp.patchRect.width)
									)
								);
						cv::erode(pp.mask, pp.mask, element);
						pp.mask.copyTo(
								mask(
									cv::Range(pp.patchRect.y, pp.patchRect.y + pp.patchRect.height),
									cv::Range(pp.patchRect.x, pp.patchRect.x + pp.patchRect.width)
									)
								);
						// generate the mask for current image, since we don't know where the object is, the mask should be relatively large
						cv::Point upLeft, rightbottom;
						findCorner(mask, upLeft, rightbottom);
						genMask(curMask, upLeft, rightbottom, erodeSize);
						
						// init the detector
						cv::SURF surf;
						std::vector<cv::KeyPoint> keyPoints_1, keyPoints_2;  
						cv::Mat descriptors_1, descriptors_2;  
						// detect
						surf(orgPatch,  mask, keyPoints_1, descriptors_1);  
						surf(curImg, curMask, keyPoints_2, descriptors_2);  
						// match
						cv::BruteForceMatcher<cv::L2<float> >  matcher;	
						std::vector<cv::DMatch> matches;  
						matcher.match(descriptors_1, descriptors_2, matches);  

						double max_dist = 0; double min_dist = 100;  
						//-- Quick calculation of max and min distances between keypoints  
						for( int i = 0; i < descriptors_1.rows; i++ )  
						{   
							double dist = matches[i].distance;  
							if( dist < min_dist ) min_dist = dist;  
							if( dist > max_dist ) max_dist = dist;  
						}  
						//printf("-- Max dist : %f \n", max_dist );  
						//printf("-- Min dist : %f \n", min_dist );  
						//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
						//-- PS.- radiusMatch can also be used here.  
						std::vector< cv::DMatch > good_matches;  
						for( int i = 0; i < descriptors_1.rows; i++ )  
						{   
							if( matches[i].distance < 2*min_dist )  
							{   
								good_matches.push_back( matches[i]);   
							}  
						}  

						// DEBUG only
						//cv::Mat img_matches;  
						//cv::drawMatches(orgPatch, keyPoints_1, curImg, keyPoints_2,  
						//		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),  
						//		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
						//cv::imshow("Curmask", curMask);
						//cv::imshow( "Match", img_matches);  
						//cv::waitKey();  
						// end of DEBUG

						// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
						// push back to compute pose
						// extract the matched points
						int dx = pp.patchRect.x;
						int dy = pp.patchRect.y;
						for (size_t j = 0; j < good_matches.size(); j++)
						{
							// extract the two 2d points
							float px = keyPoints_1[good_matches[j].queryIdx].pt.x;
							float py = keyPoints_1[good_matches[j].queryIdx].pt.y;
							float cx = keyPoints_2[good_matches[j].queryIdx].pt.x;
							float cy = keyPoints_2[good_matches[j].queryIdx].pt.y;
							
							vpPoint p;	
							// 2d point in current frame
							double u, v;
							vpPixelMeterConversion::convertPoint(cam, cx, cy, u, v);
							p.set_x(u);
							p.set_y(v);
							p.set_w(1);
							// 3d point in previous frame
							float depth = pp.depth.at<float>(py - dy, px - dx);
							if (depth < 1e-5)
								continue;
							backProj(invK, invP, depth, px, py, p);
							pSetsOne.push_back(p);
						}
					}
		}

#pragma omp section
		{
			// for dataset
			vpMatrix invVirtualK = invVirtualCam;
			//
			// detect good features in dataPatches
			// virtual camera is used here
			vpCameraParameters vc;
			vc.initFromCalibrationMatrix(virtualCam);
			for (size_t i = 0; i < vpMbEdgeTracker::faces.getPolygon().size(); i++)
				if (isVisible[i])
					for (std::list<patch>::iterator pp = dataPatches[i].begin(); pp != dataPatches[i].end(); ++pp)
					{
						cv::erode(pp->mask, pp->mask, element);
						cv::SURF surf;
						std::vector<cv::KeyPoint> keyPoints_1, keyPoints_2;  
						cv::Mat descriptors_1, descriptors_2;  
						// detect
						surf(pp->orgPatch, pp->mask, keyPoints_1, descriptors_1);  
						// TODO: check the scale of patches here
						cv::Mat curMask = pp->mask.clone();
						// generate the mask for current image, since we don't know where the object is, the mask should be relatively large
						cv::Point upLeft, rightbottom;
						findCorner(pp->mask, upLeft, rightbottom);
						genMask(curMask, upLeft, rightbottom, erodeSize);
						surf(img, curMask, keyPoints_2, descriptors_2);  
						// match
						cv::BruteForceMatcher<cv::L2<float> >  matcher;
						std::vector<cv::DMatch> matches;  
						matcher.match(descriptors_1, descriptors_2, matches);  

						double max_dist = 0; double min_dist = 100;  
						//-- Quick calculation of max and min distances between keypoints  
						for( int i = 0; i < descriptors_1.rows; i++ )  
						{   
							double dist = matches[i].distance;  
							if( dist < min_dist ) min_dist = dist;  
							if( dist > max_dist ) max_dist = dist;  
						}  
						//printf("-- Max dist : %f \n", max_dist );  
						//printf("-- Min dist : %f \n", min_dist );  
						//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )  
						//-- PS.- radiusMatch can also be used here.  
						std::vector< cv::DMatch > good_matches;  
						for( int i = 0; i < descriptors_1.rows; i++ )  
						{   
							if( matches[i].distance < 2*min_dist )  
							{   
								good_matches.push_back( matches[i]);   
							}  
						}  

						// DEBUG only
						//cv::Mat img_matches;  
						//cv::drawMatches(pp->orgPatch, keyPoints_1, img, keyPoints_2,  
						//		good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),  
						//		std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);  
						//cv::imshow( "Match", img_matches);  
						//cv::imshow("Curmask", curMask);
						//cv::waitKey();  
						// end of DEBUG


						// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
						// push back to compute pose
						for (size_t j = 0; j < good_matches.size(); j++)
							{
							// extract the two 2d points
							float px = keyPoints_1[good_matches[j].queryIdx].pt.x;
							float py = keyPoints_1[good_matches[j].queryIdx].pt.y;
							float cx = keyPoints_2[good_matches[j].queryIdx].pt.x;
							float cy = keyPoints_2[good_matches[j].queryIdx].pt.y;

								vpPoint p;	
								// 2d point
								double u, v;
								vpPixelMeterConversion::convertPoint(vc, cx,cy,u,v);
								p.set_x(u);
								p.set_y(v);
								p.set_w(1);
								// 3d point
								float depth = pp->depth.at<float>(py, px);
								if (depth < 1e-5)
									continue;
								backProj(invVirtualK, invP, depth, px, py, p);
								pSetsTwo.push_back(p);
							}
					}
		}
	}

	for (std::list<vpPoint>::iterator itr = pSetsOne.begin(); itr != pSetsOne.end(); ++itr)
		featuresComputePose.addFeaturePoint(*itr);
	for (std::list<vpPoint>::iterator itr = pSetsTwo.begin(); itr != pSetsTwo.end(); ++itr)
		featuresComputePose.addFeaturePoint(*itr);
}

void
superResolutionTracker::findCorner(cv::Mat& mask, cv::Point& upLeft, cv::Point& rightbottom)
{
	upLeft.x 		= mask.cols;
	upLeft.y 		= mask.rows;
	rightbottom.x 	= 0;
	rightbottom.y 	= 0;
	for (int i = 0; i < mask.rows; i++)
		for (int j = 0; j < mask.cols; j++)
			if (mask.at<unsigned char>(i, j) == 255)
			{
				if (i < upLeft.y)
					upLeft.y = i;
				if (i > rightbottom.y)
					rightbottom.y = i;
				if (j < upLeft.x)
					upLeft.x = j;
				if (j > rightbottom.x)
					rightbottom.x = j;	
			}
}

inline void
superResolutionTracker::genMask(cv::Mat& curMask, cv::Point& upLeft, cv::Point& rightbottom, int erodeSize)
{
	int left  	= std::max(upLeft.x - erodeSize, 0);
	int right 	= std::min(rightbottom.x + erodeSize, curMask.cols);
	int up  	= std::max(upLeft.y - erodeSize, 0);
	int bottom 	= std::min(rightbottom.y + erodeSize, curMask.rows);
	curMask(cv::Range(up, bottom), cv::Range(left, right)) = 255;
}

double 
superResolutionTracker::CorrNormedWithMask(cv::Mat& src1, cv::Mat& src2, cv::Mat& mask)
{
	double dot = 0;
	double norm1 = 0;
	double norm2 = 0;
	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++)
			if (mask.at<unsigned char>(i, j) == 255)
			{
				dot   += src1.at<unsigned char>(i, j) * src2.at<unsigned char>(i, j);
				norm1 += src1.at<unsigned char>(i, j) * src1.at<unsigned char>(i, j);			
				norm2 += src2.at<unsigned char>(i, j) * src2.at<unsigned char>(i, j);		
			}
	return dot / (std::sqrt(norm1 * norm2));
}

double 
superResolutionTracker::MADWithMask(cv::Mat& src1, cv::Mat& src2, cv::Mat& mask)
{
	int iCount = 0;
	double sum = 0;
	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++)
			if (mask.at<unsigned char>(i, j) == 255)
			{
				sum += abs((int)src1.at<unsigned char>(i, j) - (int)src2.at<unsigned char>(i, j));
				iCount++;
			}
	return sum / iCount;
}
