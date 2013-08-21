/**
 * @file superResolutionTracker.cpp
 * @brief This is the implementation of the super resolution tracker.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-08-08
 */

#include "endeffector_tracking/superResolutionTracker.h"

superResolutionTracker::superResolutionTracker()
:numOfPatchScale(10)
,buffSize(20)
,numOfPatches(10)
,numOfPatchesUsed(2)
,upScale(1)
,winSize(5)
,maxLevel(1)
,maxDownScale(16)
,faceAngle(80)
,frameCount(0)
,minFrameCount(20)
{
	omp_init_lock(&buffLock);
	omp_init_lock(&dataLock);
}

// TODO: check the lock, by search dataset
// TODO: check any pointer like data in the patches shared unconsciously
void
superResolutionTracker::track(void)
{
	float rate = getRate(maxDownScale, numOfPatchScale);
	vpImageConvert::convert(curImg,I);

	// maintain the visibility test
	int i = 0;
	for (std::list<vpMbtPolygon *>::iterator itr = faces.getPolygon().begin(); itr != faces.getPolygon().end(); ++itr)
	{
		isVisible[i] = (*itr)->isVisible(cMo, vpMath::rad(faceAngle));
		++i;
	}

	// a copy of the whole dataset only including the tracking related patches
	dataset_t dataPatches;

	// upScale the src, pre, and database patches for tracking
	// src
	cv::Mat upScaleImg;
	cv::resize(curImg, upScaleImg, cv::Size(curImg.cols * rate, curImg.rows * rate));
	// dataset
	for (size_t i = 0; i < faces.getPolygon().size(); i++)
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
	optimizePose(upScaleImg, prePatch, dataPatches);
	vpMbEdgeTracker::track(I);
	//pose.buildFrom(cMo);
	//std::cout<<pose<<std::endl;

	// maintain the visibility test
	i = 0;
	for (std::list<vpMbtPolygon*>::iterator itr = faces.getPolygon().begin(); itr != faces.getPolygon().end(); ++itr)
	{
		isVisible[i] = (*itr)->isVisible(cMo, vpMath::rad(faceAngle));
		++i;
	}

	// save the patch for next frame tracking
	for (size_t i = 0; i < faces.getPolygon().size(); i++)
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
		for (size_t i = 0; i < faces.getPolygon().size(); i++)
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
	vpTime::wait(10);
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
		for (int i = 0; i < 1e5; i++)
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
	if (buff.size() < buffSize)
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
	if (dataset[patchData.faceID].size() > numOfPatches)
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

	std::list<vpMbtPolygon*>::iterator itr = faces.getPolygon().begin();
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

	for (size_t i = 0; i < npt; i++)
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
	int patchScale;
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

// TODO: check all the project, make sure whether a project will produce holes in the map
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
	vpHomogeneousMatrix P = patchData.pose;
	int shiftX = patchData.patchRect.x;
	int shiftY = patchData.patchRect.y;

	int height = si.faceSizes[patchData.patchScale].height;
	int width = si.faceSizes[patchData.patchScale].width;
	float depth = si.depth;
	//float* pd = (float*) (patchData.depth.data);
	cv::Mat sp(si.faceSizes[patchData.patchScale], CV_8UC1);
	uchar* ps = (uchar*) (sp.data);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			int xc, yc;
			float Z;
			projPoint(invK, invP, P, K, depth, j, i, xc, yc, Z);
			xc -= shiftX;
			yc -= shiftY;
			*ps++ = patchData.orgPatch.at<uchar>(yc, xc);
		}
	patchData.scaledPatch[patchData.patchScale] = sp;
	// DEBUG only
	//cv::imshow("patch", sp);
	//cv::imshow("orgPatch", patchData.orgPatch);
	//cv::waitKey();
	//END OF DEBUG
}

void
superResolutionTracker::genRestScalePatch(patch& patchData, int scaleID)
{
	if (scaleID == patchData.patchScale)
		return;
	else if (scaleID < patchData.patchScale)
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
		if (itr->confidence[scaleID])
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
	// scaleID - 1 here won't be negative
	cv::Mat lowPatch = patchData.scaledPatch[scaleID - 1];
	std::list<patch>::iterator itr = dataset[patchData.faceID].begin();
	for (int i = 0; i < patchID; i++)
		++itr;
	cv::Mat highPatch = itr->scaledPatch[scaleID];

	cv::Size lowSize(lowPatch.cols, lowPatch.rows);
	cv::Size highSize(highPatch.cols, highPatch.rows);
	cv::Mat count = cv::Mat::zeros(lowSize, CV_32FC1) + 1e-10;
	cv::Mat tarLowPatch = cv::Mat::zeros(lowSize, CV_32FC1);
	cv::Mat light = cv::Mat::ones(lowSize, CV_32FC1);
	cv::Mat high2low(lowSize, CV_32FC1);
	cv::Mat curHighPatch(highSize, CV_8UC1);
	cv::Mat xCoor(highSize, CV_32FC1);
	cv::Mat yCoor(highSize, CV_32FC1);

	// process
	// hight to low projection
	scaleInfo& si = faceScaleInfo[patchData.faceID];
	vpMatrix K = si.Ks[scaleID-1];
	vpMatrix invK = si.Ks[scaleID].inverseByLU();
	vpMatrix invP = si.cMo.inverseByLU();
	vpHomogeneousMatrix P = si.cMo;
	float depth = si.depth;
	uchar* hp = (uchar*) (highPatch.data);
	float* xp = (float*) (xCoor.data);
	float* yp = (float*) (yCoor.data);
	for (int i = 0; i < highPatch.rows; i++)
		for (int j = 0; j < highPatch.cols; j++)		
		{
			// project
			int xc, yc;
			float Z;
			projPoint(invK, invP, P, K, depth, j, i, xc, yc, Z);

			// update
			*xp++ = xc;
			*yp++ = yc;
			count.at<float>(yc, xc) += 1;
			tarLowPatch.at<float>(yc, xc) += *hp++;
		}

	cv::divide(tarLowPatch, count, tarLowPatch);

	findLight(tarLowPatch, lowPatch, light);

	// generate the super resolution patch
	hp = (uchar*) (highPatch.data);
	uchar* chp = (uchar*) (curHighPatch.data);
	xp = (float*) (xCoor.data);
	yp = (float*) (yCoor.data);
	for (int i = 0; i < highPatch.rows; i++)
		for (int j = 0; j < highPatch.cols; j++)		
			*chp++ = (*hp++) * light.at<float>(*yp++, *xp++);
	// save processed data
	patchData.scaledPatch[scaleID] = curHighPatch;	
}

// TODO: don't forget the lock while updating the database
// TODO: check all the map based structure, to see whether the loop works correctly
	void
superResolutionTracker::refreshDataset(void)
{

	// reset the isChanged flags
	for (int i = 0; i < numOfPatchScale; i++)	
		for (size_t j = 0; j < faces.getPolygon().size(); j++)
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
				itrPatch->isChanged[i] = false;

	// number of scales, start from the lowest scale
	// there is no need to update the lowest scale, so we start from scale 1 here
	for (int i = 1; i < numOfPatchScale; i++)	
		for (size_t j = 0; j < faces.getPolygon().size(); j++) // number of faces
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
				if (!itrPatch->confidence[i]) // only those without confidence should be updated
				{
					int patchID;
					if (findConfidence(i, patchID, *itrPatch))
					{
						omp_set_lock(&dataLock);
						superResolution(i, patchID, *itrPatch);
						itrPatch->confidence[i] = true;
						itrPatch->isChanged[i] = true;
						omp_unset_lock(&dataLock);
					}
					else
					{
						if (itrPatch->isChanged[i-1])
						{
							omp_set_lock(&dataLock);
							interpolate(*itrPatch, i);
							itrPatch->isChanged[i] = true;
							omp_unset_lock(&dataLock);
						}
					}
				}

	// maintain the highestConfidenceScale
	omp_set_lock(&dataLock);
	for (int i = 0; i < numOfPatchScale; i++)	
		for (size_t j = 0; j < faces.getPolygon().size(); j++)
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
	int num_labels = 10;

	// setup the weight of labels
	float low = 0.7,
		  high = 1.3;
	std::vector<float> lightWeight(num_labels);
	for (int i = 0; i < num_labels; i++)
		lightWeight[i] = low + (float)i / num_labels * (high - low);

	// FIXME: tune this value
	int smoothWeight = 10;
	try
	{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// first set up data costs individually
		uchar *tar = (uchar*) (tarLowPatch.data);
		uchar *cur = (uchar*) (lowPatch.data);
		for ( int i = 0; i < num_pixels; i++ )
			for (int l = 0; l < num_labels; l++ )
			{
				int cost = abs(lightWeight[l] * tar[i] - cur[i]);
				gc->setDataCost(i,l, cost);
			}

		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ )
			{
				int cost = smoothWeight * abs(lightWeight[l1] - lightWeight[l2]);
				gc->setSmoothCost(l1,l2,cost); 
			}

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->swap(1);
		gc->expansion(1);
		gc->swap(1);
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
	// check whether the dataset is empty, is so, return
	if (dataset[faceID].empty())
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
	int highestID;
	for (std::list<patch>::iterator itr = dataset[faceID].begin(); itr != dataset[faceID].end(); ++itr)
	{
		if (itr->highestConfidenceScale >= std::min(patchScale + upScale, numOfPatchScale-1))
			id.push_back(count);
		if (itr->highestConfidenceScale > highestScale)
		{
			highestScale = itr->highestConfidenceScale;
			highestID = count;
		}
		++count;
	}

	if (id.empty())
	{
		//the deepCopyPatch here only copy the information used by the tracker
		//std::list<patch>::iterator itr = dataset[faceID].begin();
		//for (int i = 0; i < highestID; i++)
		//	++itr;
		//for (int i = 0; i < numOfPatchesUsed; i++)
		//	deepCopyPatch(patchList, *itr);
	}
	else
	{
		cv::RNG rng;
		for (int i = 0; i < numOfPatchesUsed; i++)
		{
			int idx = rng.uniform(0, count);
			std::list<patch>::iterator itr = dataset[faceID].begin();
			for (int i = 0; i < id[idx]; i++)
				++itr;
			deepCopyPatch(patchList, *itr);
		}
	}
	omp_unset_lock(&dataLock);

	// 4. Project them on the image based on the virtual internal parameters and the pre-pose
	projPatch(pose, faceID, patchList);
}

// FIXME: some computational expensive steps in this function, try to tune it
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
		vpMatrix K = getVirtualCam();
		float dp = faceScaleInfo[id].depth;

		// find the four corners
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

			// project the corners 
			int xc, yc;
			float Z;
			projPoint(invK, invP, P, K, dp, x, y, xc, yc, Z);
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

		invK = getVirtualCam().inverseByLU();
		invP = (itr->pose).inverseByLU();
		P = faceScaleInfo[id].cMo;
		K = faceScaleInfo[id].Ks[hs];
		uchar* pm = (uchar*) (mask.data);
		float* pd = (float*) (depth.data);
		uchar* pi = (uchar*) (projImg.data);
		cv::Mat oPatch = itr->scaledPatch[hs];
		vpMatrix A(3, 3), 
				 b(3, 1);
        float coefficient[6];    
		//	a11a22
        //  a01a12
        //  a02a21
        //  a12a21
        //  a01a22		
        //  a02a11
		cv::Point lu, rb;
		findMinMax(corners, lu, rb);
		prepCalcDepth(corners3d, A, b, coefficient);
		for (int i = lu.y; i < rb.y + 1; i++)
			for (int j = lu.x; j < rb.x + 1; j++)
			{
				int offset = c * i + j;
				if (*(pm + offset) == 255)
				{
					*(pd + offset) = calcDepth(A, b, coefficient, invP, invK, cv::Point(j, i));
					int xc, yc;
					float Z;
					projPoint(invK, invP, P, K, *(pd + offset), j, i, xc, yc, Z);
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

	std::list<vpMbtPolygon*>::iterator itr = faces.getPolygon().begin();
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

	for (size_t i = 0; i < npt; i++)
		cv::line(mask, pt[i], pt[(i+1) % npt], 255);
	cv::floodFill(mask, meanPoint(pt), 255);

	// handle the mask problem here
	cv::Point lt, rb;
	lt.x = cols;
	lt.y = rows;
	rb.x = 0;
	rb.y = 0;
	for (size_t i = 0; i < npt; i++)
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
	for (int i = 0; i < mask.rows; i++)
		for (int j = 0; j < mask.cols; j++)
		{
			if (*pm++ == 255)
				*pd = calcDepth(A, b, coefficient, invP, invK, cv::Point(j, i));
			++pd;
		}


	// save the patch using the filled Project face
	//
	// patchRect
	int border = 3;
	p.patchRect.x 		= lt.x - border;
	p.patchRect.y 		= lt.y - border;
	p.patchRect.width 	= rb.x - lt.x + 1 + 2 * border;
	p.patchRect.height 	= rb.y - lt.y + 1 + 2 * border;
	// faceID
	p.faceID = faceID;
	// pose
	p.pose = cMo;
	// orgPatch
	p.orgPatch = cv::Mat(curImg, cv::Range(lt.y - border, rb.y + 1 + border), cv::Range(lt.x - border, rb.x + 1 + border)).clone();
	// mask
	p.mask = cv::Mat(mask, cv::Range(lt.y - border, rb.y + 1 + border), cv::Range(lt.x - border, rb.x + 1 + border));
	// depth
	p.depth = cv::Mat(depth, cv::Range(lt.y - border, rb.y + 1 + border), cv::Range(lt.x - border, rb.x + 1 + border));
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
	x[0][0] = ix;
	x[1][0] = iy;
	x[2][0] = 1;

	// 3d coordinate without homo
	x = depth * invK * x;
	// 3d coordinate homo
	vpMatrix X(4, 1);
	X[0][0] = x[0][0];
	X[1][0] = x[1][0];
	X[2][0] = x[2][0];
	X[3][0] = 1;
	X = invP * X;

	// 3d homo coordinatein the org frame
	p.setWorldCoordinates(X[0][0], X[1][0], X[2][0]);
}

void
superResolutionTracker::projPoint(
		const vpMatrix& invK, 
		const vpMatrix& invP, 
		const vpHomogeneousMatrix& P, 
		const vpMatrix& K, 
		float 			depth, 
		float 			ix, 
		float 			iy, 
		int& 			xc, 
		int& 			yc, 
		float& 			Z)
{
	vpPoint p;
	backProj(invK, invP, depth, ix, iy, p);
	// 3d homo coordinate in the target frame
	p.changeFrame(P);
	// 3d coordinate in the target frame
	vpMatrix X(3, 1); 
	X[0][0] = p.get_X();
	X[1][0] = p.get_Y();
	X[2][0] = p.get_Z();
	// 2d coordinate in the target frame
	X = K * X;
	xc = X[0][0] / X[2][0];
	yc = X[1][0] / X[2][0];

	// final depth required
	Z = p.get_Z();
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
	vpPoseFeatures featuresComputePose;
	// build the image pyramid for tracking
	std::vector<cv::Mat> cPyramid;
	std::vector<cv::Mat> upScaleCPyramid;
	cv::buildOpticalFlowPyramid(curImg, cPyramid, cv::Size(winSize, winSize), maxLevel);
	cv::buildOpticalFlowPyramid(img, upScaleCPyramid, cv::Size(winSize, winSize), maxLevel);

	// for dataset
	vpMatrix invVirtualK = getVirtualCam().inverseByLU();
	// for pre-frame
	vpMatrix invK = cam.get_K().inverseByLU();
	vpMatrix invP = cMo.inverseByLU();

	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9,9), cv::Point(4, 4)); // erode the mask so the feature points detected will not be at the border

	// for pre-frame
	for (size_t i = 0; i < faces.getPolygon().size(); i++)
		if (isVisible[i])
			if(!prePatch[i].empty())
			{
				patch& pp = prePatch[i].front();
				// restore the patch to the image size
				cv::Mat orgPatch = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
				cv::Mat mask 	 = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC1);
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
				// detect good features in pre-patch
				std::vector<cv::Point2f> corners;
				cv::goodFeaturesToTrack(orgPatch, corners, 30, 0.01, 5, mask);
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
				std::vector<cv::Mat> prePatchPyr;
				std::vector<cv::Point2f> cFeatures;
				std::vector<float> fErr;
				std::vector<uchar> fStatus;
				cv::buildOpticalFlowPyramid(orgPatch, prePatchPyr, cv::Size(winSize, winSize), maxLevel);
				cv::calcOpticalFlowPyrLK(prePatchPyr, cPyramid, corners, cFeatures, fStatus, fErr, cv::Size(winSize, winSize), maxLevel);
				// DEBUG only
				//cv::Mat cImg = curImg.clone();
				//for (size_t j = 0; j < cFeatures.size(); j++)
				//	if (fStatus[j] == 1) // only the tracked features are used
				//		cv::circle(cImg, cFeatures[j], 3, cv::Scalar(255, 0, 0));
				//cv::imshow("cur-features", cImg);
				//cv::waitKey();
				// END OF DEBUG

				// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
				// push back to compute pose
				int dx = pp.patchRect.x;
				int dy = pp.patchRect.y;
				for (size_t j = 0; j < corners.size(); j++)
					if (fStatus[j] == 1) // only the tracked features are used
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
						featuresComputePose.addFeaturePoint(p);
					}
			}

	//
	// detect good features in dataPatches
	// virtual camera is used here
	vpCameraParameters virtualCam;
	virtualCam.initFromCalibrationMatrix(getVirtualCam());
	for (size_t i = 0; i < faces.getPolygon().size(); i++)
		if (isVisible[i])
			for (std::list<patch>::iterator pp = dataPatches[i].begin(); pp != dataPatches[i].end(); ++pp)
			{
				// detect good features in pp-patch
				std::vector<cv::Point2f> corners;
				cv::erode(pp->mask, pp->mask, element);
				cv::goodFeaturesToTrack(pp->orgPatch, corners, 30, 0.01, 5, pp->mask);
				// DEBUG only
				//cv::imshow("mask", pp->mask);
				//cv::waitKey();
				// END OF DEBUG

				// KLT search them in cur-image
				std::vector<cv::Mat> prePatchPyr;
				std::vector<cv::Point2f> cFeatures;
				std::vector<float> fErr;
				std::vector<uchar> fStatus;
				cv::buildOpticalFlowPyramid(pp->orgPatch, prePatchPyr, cv::Size(winSize, winSize), maxLevel);
				cv::calcOpticalFlowPyrLK(prePatchPyr, upScaleCPyramid, corners, cFeatures, fStatus, fErr, cv::Size(winSize, winSize), maxLevel);

				// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
				// push back to compute pose
				for (size_t j = 0; j < corners.size(); j++)
					if (fStatus[j] == 1)
					{
						vpPoint p;	
						// 2d point
						double u, v;
						vpPixelMeterConversion::convertPoint(virtualCam, cFeatures[j].x,cFeatures[j].y,u,v);
						p.set_x(u);
						p.set_y(v);
						p.set_w(1);
						// 3d point
						float depth = pp->depth.at<float>(corners[j].y, corners[j].x);
						if (depth < 1e-5)
							continue;
						backProj(invVirtualK, invP, depth, corners[j].x, corners[j].y, p);
						featuresComputePose.addFeaturePoint(p);
					}
			}

	//
	// compute pose
	featuresComputePose.setLambda(0.6);
	try
	{
		p_cMo = cMo;
		featuresComputePose.computePose(cMo);
	}
	catch(...) // catch all kinds of Exceptions
	{
		std::cout<<"Exception raised in computePose"<<std::endl;
	}
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
	float th2 = 0.001;
	bool frameDist = true, 
		 poseDiff  = true, 
		 matchness = true;

	// frameDist
	frameDist = frameCount > minFrameCount ? true : false;

	// poseDiff
	vpPoseVector pose;
	pose.buildFrom(cMo);
	omp_set_lock(&dataLock);
	for (int i = 0; i < faces.getPolygon().size(); i++)
		for (std::list<patch>::iterator itr = dataset[i].begin(); itr != dataset[i].end(); ++itr)
		{
			vpPoseVector curPose;
			curPose.buildFrom(itr->pose);
			for (int j = 0; j < 3; j++)
				if (fabs(pose[j] - curPose[j]) < th1)
				{
					poseDiff = false;
					break;
				}
			for (int j = 3; j < 6; j++)
				if (fabs(pose[j] - curPose[j]) < th2)
				{
					poseDiff = false;
					break;
				}
		}
	omp_unset_lock(&dataLock);

	// matchness
	return frameDist & poseDiff & matchness;
}

void 
superResolutionTracker::initialization(std::string config_file, std::string modelName, std::string initName)
{
	loadConfigFile(config_file);
	// Load the 3d model in cao format. No 3rd party library is required
	loadModel(modelName); 

	// Initialise manually the pose by clicking on the image points associated to the 3d points contained in the cube.init file.
	vpImageConvert::convert(curImg,I);
	disp.init(I);
	initClick(I, initName); 

	// initialize the member variables
	rows = curImg.rows;
	cols = curImg.cols;

	for (size_t i = 0; i < faces.getPolygon().size(); i++)
		isVisible.push_back(false);
	// once the model has been read, we can init this important structure
	initFaceScaleInfo();

	// for first frame tracking
	int i = 0;
	for (std::list<vpMbtPolygon*>::iterator itr = faces.getPolygon().begin(); itr != faces.getPolygon().end(); ++itr)
	{
		isVisible[i] = (*itr)->isVisible(cMo, vpMath::rad(faceAngle));
		++i;
	}
	for (size_t i = 0; i < faces.getPolygon().size(); i++)
	{
		prePatch[i].clear();
		if (isVisible[i])
		{
			patch p;
			obtainPatch(i, p);
			prePatch[i].push_back(p);
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
	for (std::list<vpMbtPolygon*>::iterator itr = faces.getPolygon().begin(); itr != faces.getPolygon().end(); ++itr)
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
	for (std::list<vpMbtPolygon*>::iterator itr = faces.getPolygon().begin(); itr != faces.getPolygon().end(); ++itr)
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
	for (std::list<vpMbtPolygon*>::iterator itr = faces.getPolygon().begin(); itr != faces.getPolygon().end(); ++itr)
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
	for (size_t i = 0; i < faces.getPolygon().size(); i++)
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
