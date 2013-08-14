/**
 * @file superResolutionTracker.cpp
 * @brief This is the implementation of the super resolution tracker.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-08-08
 */

#include "endeffector_tracking/superResolutionTracker.h"

superResolutionTracker::superResolutionTracker()
:numOfPatchScale(5)
,buffSize(20)
,numOfPatches(10)
,numOfPatchesUsed(3)
,upScale(1)
,winSize(10)
,maxLevel(3)
,maxDownScale(16)
,faceAngle(80)
{
	omp_init_lock(&buffLock);
	omp_init_lock(&dataLock);

	for (size_t i = 0; i < faces.size(); i++)
		isVisible.push_back(false);
}

// TODO: check the lock , by search dataset
// TODO: check any pointer like data in the patches shared unconsciously
// TODO: check all the internal camera parameters used all over the codes
// TODO: check the offset of the patches while using the virtual camera
// TODO: then, check the whole track procedure
// TODO: the first frame track is crucial
void
superResolutionTracker::track(void)
{
	float rate = getRate(maxDownScale, numOfPatchScale);

	// maintain the visibility test
	for (size_t i = 0; i < faces.size(); i++)
		isVisible[i] = faces[i]->isVisible(cMo, vpMath::rad(faceAngle));

	// a copy of the whole dataset only including the tracking related patches
	dataset_t dataPatches;

	// upScale the src, pre, and database patches for tracking
	// src
	cv::Mat upScaleImg;
	cv::resize(curImg, upScaleImg, cv::Size(curImg.cols * rate, curImg.rows * rate));
	// dataset
	for (size_t i = 0; i < faces.size(); i++)
		if (isVisible[i])
		{
			int patchScale = findPatchScale(i);
			// find the corresponding patches, and copy them out of the dataset
			// then project them onto the current pose with the virtual camera
			findCopyProcessPatch(i, patchScale, cMo, dataPatches[i]);
		}

	// track
	// super resolution tracking only used in the match between the database and the current frame. The match between the pre and cur frame are based on the org scale
	optimizePose(upScaleImg, prePatch, dataPatches);

	// maintain the visibility test
	for (size_t i = 0; i < faces.size(); i++)
		isVisible[i] = faces[i]->isVisible(cMo, vpMath::rad(faceAngle));

	// save the patch for next frame tracking
	for (size_t i = 0; i < faces.size(); i++)
		if (isVisible[i])
		{
			prePatch[i].clear();
			prePatch[i].push_back(obtainPatch[i]);
		}

	// measure whether the tracked frame is the key frame
	// if true
	// push the frame into the buff
	if (isKeyFrame())
	{
		for (size_t i = 0; i < faces.size(); i++)
			if (isVisible[i])
			{
				// note: one key frame will produce several patches
				// buff lock is encapsulated in the function
				pushDataIntoBuff(deepCopyPrePatch(prePatch[i].front()));
			}
	}
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

	int npt = faces[faceID]->nbPoints;
	std::vector<cv::Point> p(npt);
	for (int i = 0; i < npt; i++)
	{
		vpPoint& vp = faces[faceID]->p[i];
		vp.changeFrame(cMo);
		vp.project();

		vpMeterPixelConversion::convertPoint(cam, vp.get_x(), vp.get_y(), p[i].x, p[i].y);
	}

	// TODO: check whether the codes here work
	for (size_t i = 0; i < npt; i++)
		cv::line(mask, p[i], p[(i+1) % npt], 255);
	cv::floodFill(mask, meanPoint(p), 255);

	// mask
	p.mask = mask;
	p.faceID = faceID;
	findPatchScale(p);

	return p.patchScale;
}

void 
superResolutionTracker::findPatchScale(patch& patchData)
{	
	float numOfPixels = cv::sum(patchData.mask);	
	numOfPixels /= 255;
	float curWidth  = 0,
		  curHeight = 0,
		  nxtWidth  = 0,
		  nxtHeight = 0;
	int patchScale;
	bool scaleFound = false;
	scaleInfo& si = faceScaleInfo[patchData.faceID];
	for (i = 0; i < numOfPatchScale; i++)
	{
		switch (i)
		{
			case 0:
				nxtWidth = si.faceSizes[i].width;
				nxtHeight = si.faceSizes[i].height;
				if (numOfPixels < nxtWidth * nxtHeight)
				{
					patchScale = i;	
					scaleFound = true;
				}
				break;
			case numOfPatchScale - 1:
				patchScale = i;	
				scaleFound = true;
				break;
			default:
				curWidth = nxtWidth;
				curHeight = nxtHeight;
				nxtWidth = si.faceSizes[i+1].width;
				nxtHeight = si.faceSizes[i+1].height;
				if (numOfPixels > curWidth * curHeight && numOfPixels < nxtWidth * nxtHeight)
				{
					patchScale = numOfPixels / (curWidth * curHeight) > (nxtWidth * nxtHeight) / numOfPixels ? i+1:i;	
					scaleFound = true;
				}
				break;
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
	vpMatrix P = patchData.pose;
	int shiftX = patchData.patchRect.x;
	int shiftY = patchData.patchRect.y;

	int height = si.faceSizes[patchData.patchScale].height;
	int width = si.faceSizes[patchData.patchScale].width;
	(float*) pd = (float*) (patchData.depth.data);
	cv::Mat sp(si.faceSizes[patchData.patchScale], CV_8UC1);
	(uchar*) ps = (uchar*) (sp.data);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			int xc, yc;
			float Z;
			projPoint(invK, invP, P, K, *pd++, j, i, xc, yc, Z);
			xc -= shiftX;
			yc -= shiftY;
			*ps++ = patchData.orgPatch.at<uchar>(yc, xc);
		}
	patchData.scaledPatch[patchData.patchScale] = sp;
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
		{
			IDs.push_back(count);
		}
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
superResolutionTracker::superResolution(scaleID, patchID, patchData)
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
	cv::Mat xCoor(highestScale, CV_32FC1);
	cv::Mat yCoor(highestScale, CV_32FC1);

	// process
	// hight to low projection
	scaleInfo& si = faceScaleInfo[patchData.faceID];
	vpMatrix K = si.Ks[scaleID-1];
	vpMatrix invK = si.Ks[scaleID].inverseByLU();
	vpMatrix invP = si.cMo.inverseByLU();
	vpMatrix P = si.cMo;
	float depth = si.depth;
	(uchar*) hp = (uchar*) (highPatch.data);
	(float*) xp = (float*) (xCoor.data);
	(float*) yp = (float*) (yCoor.data);
	for (int i = 0; i < highPatch.rows; i++)
		for (int j = 0; j < highPatch.cols; j++)		
		{
			// project
			int xc, yc;
			float Z;
			projPoint(invk, invP, P, K, depth, j, i, xc, yc, Z);

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
	(uchar*) chp = (uchar*) (curHighPatch.data);
	xp = (float*) (xCoor.data);
	yp = (float*) (yCoor.data);
	for (int i = 0; i < highPatch.rows; i++)
		for (int j = 0; j < highPatch.cols; j++)		
			*chp++ = (*hp++) * light.at<float>(*yp++, *xp++);
	// save processed data
	patchData.scaledPatch[scaleID] = curHighPatch;	
}

// TODO: don't forget the lock while updating the database
void
superResolutionTracker::refreshDataset(void)
{
	// reset the isChanged flags
	for (int i = 0; i < numOfPatchScale; i++)	
		for (size_t j = 0; j < dataset.size(); j++)
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
				itrPatch->isChanged[i] = false;

	// number of scales, start from the lowest scale
	// there is no need to update the lowest scale, so we start from scale 1 here
	for (int i = 1; i < numOfPatchScale; i++)	
	{
		// number of faces
		for (size_t j = 0; j < dataset.size(); j++)
		{
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
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
		}
	}

	// maintain the highestConfidenceScale
	for (int i = 0; i < numOfPatchScale; i++)	
		for (size_t j = 0; j < dataset.size(); j++)
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
				if (itrPatch->confidence[i])
					itrPatch->highestConfidenceScale = i;
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
		if (itr->highestConfidenceScale >= max(patchScale + upScale, numOfPatchScale-1))
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
		// the deepCopyPatch here only copy the information used by the tracker
		std::list<patch>::iterator itr = dataset[faceID].begin();
		for (int i = 0; i < highestID; i++)
			++itr;
		for (int i = 0; i < numOfPatchesUsed; i++)
			deepCopyPatch(patchList, *itr);
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
		vpMatrix P = itr->pose;
		vpMatrix K = getVirtualCam();
		float depth = faceScaleInfo[id].depth;

		cv::Size pSize = faceScaleInfo[id].faceSizes[hs];
		(uchar*) sp = (uchar*) (itr->scaledPatch[hs].data);
		for (int i = 0; i < pSize.height; i++)
			for (int j = 0; j < pSize.width; j++)
			{
				int xc, yc;
				float Z;
				projPoint(invK, invP, P, K, depth, j, i, xc, yc, Z);
				projImg.at<uchar>(yc, xc) = *sp++;
				depth.at<float>(yc, xc) = Z;
				mask.at<uchar>(yc, xc) = 255;
			}

		// copy to the patches
		itr->orgPatch = projImg;
		itr->depth = depth;
		itr->mask = mask;
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
patch
superResolutionTracker::obtainPatch(int faceID)
{
	patch p;
	// note: the patchRect should be handled here
	// project the face onto the mask image, draw the boarder lines
	// save the depth of the four corners
	// fill the projected face
	
	cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::Mat depth = cv::Mat::zeros(rows, cols, CV_32FC1);

	int npt = faces[faceID]->nbPoints;
	std::vector<cv::Point> p(npt);
	std::vector<float> depth(npt);
	for (int i = 0; i < npt; i++)
	{
		vpPoint& vp = faces[faceID]->p[i];
		vp.changeFrame(cMo);
		vp.project();

		vpMeterPixelConversion::convertPoint(cam, vp.get_x(), vp.get_y(), p[i].x, p[i].y);
		depth[i] = vp.get_Z();
	}

	// TODO: check whether the codes here work
	for (size_t i = 0; i < npt; i++)
		cv::line(mask, p[i], p[(i+1) % npt], 255);
	cv::floodFill(mask, meanPoint(p), 255);

	// handle the mask problem here
	cv::Point lt, rb;
	lt.x = cols;
	lt.y = rows;
	rb.x = 0;
	rb.y = 0;
	for (size_t i = 0; i < npt; i++)
	{
		if (p[i].x < lt.x)
			lt.x = p[i].x;
		if (p[i].y < lt.y)
			lt.y = p[i].y;
		if (p[i].x > rb.x)
			rb.x = p[i].x;
		if (p[i].y > rb.y)
			rb.y = p[i].y;
	}

	// fill the depth map using the interpolation method
	(uchar*) pm = (uchar*) (mask.data);
	(float*) pd = (float*) (depth.data);
	for (int i = 0; i < mask.rows; i++)
		for (int j = 0; j < mask.cols; j++)
		{
			if (*pm++ == 255)
				*pd++ = calcDepth(p, depth, cv::Point(j, i));
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
	p.orgPatch = cv::Mat(curImg, cv::Range(lt.y, rb.y + 1), cv::Range(lt.x, rb.x + 1));
	// mask
	p.mask = cv::Mat(mask, cv::Range(lt.y, rb.y + 1), cv::Range(lt.x, rb.x + 1));
	// depth
	p.depth = cv::Mat(depth, cv::Range(lt.y, rb.y + 1), cv::Range(lt.x, rb.x + 1));
	// scale
	findPatchScale(p);

	return p;
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

// TODO: continue here
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
	x = invK * x;
	// 3d coordinate homo
	vpMatrix X(4, 1);
	X[0][0] = x[0][0];
	X[1][0] = x[1][0];
	X[2][0] = x[2][0];
	X[3][0] = 1;
	X = depth * invP * X;

	// 3d homo coordinatein the org frame
	p.setWorldCoordinates(X[0][0], X[1][0], X[2][0]);
}

void
superResolutionTracker::projPoint(
		const vpMatrix& invK, 
		const vpMatrix& invP, 
		const vpMatrix& P, 
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
	vpMatrix tarX(3, 1); 
	tarX[0][0] = p.get_X();
	tarX[1][0] = p.get_Y();
	tarX[2][0] = p.get_Z();
	// 2d coordinate in the target frame
	X = K * X;
	int xc = X[0][0] / X[2][0];
	int yc = X[1][0] / X[2][0];

	// final depth required
	float Z = p.get_Z();
}

float
superResolutionTracker::calcDepth(
		const std::vector<cv::Point>& p, 
		const std::vector<float>& depth
		cv::Point cp)
{
	// solve the equ. Ax = b, x = A\b;
	cv::Mat A(2, 2, CV_32FC1);
	(float*) a = (float*) (A.data);
	*(a + 0) = p[1].x - p[0].x;
	*(a + 1) = p[1].x - p[2].x;
	*(a + 2) = p[1].y - p[0].y;
	*(a + 3) = p[1].y - p[2].y;

	cv::Mat b(2, 1, CV_32FC1);
	(float*) pb = (float*) (b.data);
	*(pb + 0) = cp.x + p[1].x - p[0].x - p[2].x;
	*(pb + 1) = cp.y + p[1].y - p[0].y - p[2].y;

	cv::Mat x(2, 1, CV_32FC1);
	x = A.inv() * b;

	// calculate the depth
	(float*) xb = (float*) (x.data);
	float alpha = *(xb + 0);
	float beta  = *(xb + 1);
	float dp = (alpha + beta - 1) * depth[1] + (1 - alpha) * depth[0] + (1 - beta) * depth[2];
	return dp;
}

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
	vpMatrix invK = cam.get_K();
	vpMatrix invP = cMo.inverseByLU();

	// for pre-frame
	for (size_t i = 0; i < faces.size(); i++)
		if (isVisible[i])
		{
			patch& pp = prePatch[i].front();
			// detect good features in pre-patch
			std::vector<cv::Point2f> corners;
			cv::goodFeaturesToTrack(pp.orgPatch, corners, 20, 0.01, 10, pp.mask);

			// KLT search them in cur-image
			std::vector<cv::Mat> prePatchPyr;
			std::vector<cv::Point2f> fErr, cFeatures;
			std::vector<uchar> fStatus;
			cv::buildOpticalFlowPyramid(pp.orgPatch, prePatchPyr, cv::Size(winSize, winSize), maxLevel);
			int dx = pp.patchRect.x;
			int dy = pp.patchRect.y;
			std::vector<cv::Point2f>::iterator tar = cFeatures.begin();
			for (std::vector<cv::Point2f>::iterator itr = corners.begin(); itr != corners.end(); ++itr) 
			{
				tar->x = itr->x + dx;
				tar->y = itr->y + dy;
				++tar;
			}
			cv::calcOpticalFlowPyrLK(prePatchPyr, cPyramid, corners, cFeatures, fStatus, fErr, cv::Size(winSize, winSize), maxLevel, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), OPTFLOW_USE_INITIAL_FLOW);

			// find the 3d point: based on the tracked features and the 3d point calculated based on the pre-features
			// push back to compute pose
			for (size_t j = 0; j < corners.size(); j++)
				if (fStatus[j] == 1)
				{
					vpPoint p;	
					// 2d point
					double u, v;
					vpPixelMeterConversion::convertPoint(cam, cFeatures[j].x,cFeatures[j].y,u,v);
					p.set_x(u);
					p.set_y(v);
					// 3d point
					float depth = pp.depth.at<float>(corners[j].y, corners[j].x);
					backProj(invK, invP, depth, corners[j].x + dx, corners[j].y + dy, p);
					featuresComputePose.addFeaturePoint(p);
				}
		}

	//
	// detect good features in dataPatches
	// virtual camera is used here
	vpCameraParameters virtualCam;
	virtualCam.initFromCalibrationMatrix(getVirtualCam());
	for (size_t i = 0; i < faces.size(); i++)
		if (isVisible[i])
			for (std::list<patch>::iterator pp = dataPatches[i].begin(); pp != dataPatches[i].end(); ++pp)
			{
				// detect good features in pp-patch
				std::vector<cv::Point2f> corners;
				cv::goodFeaturesToTrack(pp->orgPatch, corners, 20, 0.01, 10, pp->mask);

				// KLT search them in cur-image
				std::vector<cv::Mat> prePatchPyr;
				std::vector<cv::Point2f> fErr, cFeatures;
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
						// 3d point
						float depth = pp->depth.at<float>(corners[j].y, corners[j].x);
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

patch
superResolutionTracker::deepCopyPrePatch(const patch& src)
{
	patch p;
	p.patchScale = src.patchScale;
	p.orgPatch = src.orgPatch.clone();
	p.mask = src.mask.clone();
	p.depth = src.depth.clone();
	p.patchRect = src.patchRect;
	p.faceID = src.faceID;
	p.pose = src.pose;

	return p;
}

// TODO: complete this function
bool
superResolutionTracker::isKeyFrame(void)
{
	return true;
}

void 
superResolutionTracker::initialization(vpCameraParameters& cam, std::string modelName, std::string initName)
{
	// Get the camera parameters used by the tracker (from the configuration file).
	getCameraParameters(cam); 
	// Load the 3d model in cao format. No 3rd party library is required
	loadModel(modelName); 

	// Initialise manually the pose by clicking on the image points associated to the 3d points contained in the cube.init file.
	vpDisplayX display;
	vpImage<uchar> I;
	vpImageConvert::convert(curImg,I);
	display.init(I);
	initClick(I, initName.c_str()); 

	// initialize the member variables
	rows = curImg.rows;
	cols = curImg.cols;

	// once the model has been read, we can init this important structure
	initFaceScaleInfo();
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
	for (size_t i = 0; i < faces.size(); i++)
	{
		vpPoint p = faces[i]->p[0];
		p.changeFrame(faceScaleInfo[i].cMo);
		faceScaleInfo[i].depth = p.get_Z();
	}
}

void
superResolutionTracker::initFaceScalePose(void)
{
	for (size_t i = 0; i < faces.size(); i++)
	{
		vpCameraParameters cam;
		cam.initFromCalibrationMatrix(faceScaleInfo[i].Ks[numOfPatchScale-1]);

		vpPose pose;
		for (int j = 0; j < 4; j++)
		{
			vpPoint p = faces[i]->p[j];
			double x, y;
			int u, v;
			switch (j)
			{
				case 0:
					u = 0;
					v = 0;
					break;
				case 1:
					u = 0;
					v = faceScaleInfo[i].faceSizes[j].height;
					break;
				case 2:
					u = faceScaleInfo[i].faceSizes[j].width;
					v = faceScaleInfo[i].faceSizes[j].height;
					break;
				case 3:
					u = faceScaleInfo[i].faceSizes[j].width;
					v = 0;
					break;
				default:
					break;
			}
			vpPixelMeterConversion::convertPoint(cam, u, v, x, y);
			p.set_x(x);
			p.set_y(y);
			pose.addPoint(p);
		}
		pose.computePose(vpPose::DEMENTHON_VIRTUAL_VS, faceScaleInfo[i].cMo);
	}
}

void
superResolutionTracker::initFaceScaleSize(void)
{
	float rate = getRate(maxDownScale, numOfPatchScale);
	for (size_t i = 0; i < faces.size(); i++)
	{
		// width / height
		float aspectRatio = (float) cols / (float) rows;
		float height = pointDistance3D(faces[i]->p[0], faces[i]->p[1]);
		float width  = pointDistance3D(faces[i]->p[1], faces[i]->p[2]);
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
			faceScaleInfo[i].faceSizes[j].width = cols / pow(rate, numOfPatchScale -1 - j);
			faceScaleInfo[i].faceSizes[j].height = cols / curAR / pow(rate, numOfPatchScale -1 - j);
		}
	}
}

void
superResolutionTracker::initFaceScaleVirtualCam(void)
{
	float rate = getRate(maxDownScale, numOfPatchScale);
	for (size_t i = 0; i < faces.size(); i++)
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
