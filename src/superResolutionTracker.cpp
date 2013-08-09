/**
 * @file superResolutionTracker.cpp
 * @brief This is the implementation of the super resolution tracker.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-08-08
 */

#include "endeffector_tracking/superResolutionTracker.h"

superResolutionTracker::superResolutionTracker()
{
	omp_init_lock(&buffLock);
	omp_init_lock(&dataLock);
}

void
superResolutionTracker::track(void)
{
	omp_set_lock(&dataLock);
	// find the corresponding patches, and copy them out of the dataset
	omp_unset_lock(&dataLock);

	// track
	optimizePose();
	// measure whether the tracked frame is the key frame
	// if true
	// push the frame into the buff
	if (isKeyFrame())
	{
		// one key frame will produce several patches
		// buff lock is encapsulated in the function
		pushDataIntoBuff();
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
	// find the number of pixels in the patch, and then decide which scale the patch belongs to.
	findPatchScale(patchData);
	genOrgScalePatch(patchData);
	//
	// loop: until all the patches in all the scales generated
	for (int i = 0; i < numOfPatchScale; i++)
		genRestScalePatch(patchData, i);
}

void 
superResolutionTracker::findPatchScale(patch& patchData)
{	
	float numOfPixels = cv::sum(mask);	
	float curWidth  = 0,
		  curHeight = 0,
		  nxtWidth  = 0,
		  nxtHeight = 0;
	int patchScale;
	bool scaleFound = false;
	for (i = 0; i < numOfPatchScale; i++)
	{
		switch (i)
		{
			case 0:
				nxtWidth = faceScaleInfo[patchData.faceID].faceSizes[i].width;
				nxtHeight = faceScaleInfo[patchData.faceID].faceSizes[i].height;
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
				nxtWidth = faceScaleInfo[patchData.faceID].faceSizes[i+1].width;
				nxtHeight = faceScaleInfo[patchData.faceID].faceSizes[i+1].height;
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

void
superResolutionTracker::genOrgScalePatch(patch& patchData)
{
	// set the confidence
	patchData.confidence[scaleID] = true;
	// gen the patch
	vpMatrix K = cam.get_K();
	vpMatrix invK = K.inverseByLU();
	vpMatrix invP = faceScaleInfo[patchData.faceID].cMos[patchData.patchScale].inverseByLU();
	vpMatrix P = patchData.pose;
	float depth = 1 / faceScaleInfo[patchData.faceID].invDepth[patchData.patchScale];
	vpMatrix trans = K * P * invP * depth * invK;
	int shiftX = patchData.patchRect.x;
	int shiftY = patchData.patchRect.y;
	for (int i = 0; i < faceScaleInfo[patchData.faceID].faceSizes[patchData.patchScale].height; i++)
		for (int j = 0; j < faceScaleInfo[patchData.faceID].faceSizes[patchData.patchScale].width; j++)
		{
			vpMatrix x(3, 1);
			x[0][0] = j;
			x[1][0] = i;
			x[2][0] = 1;
			vpMatrix X = trans * x;
			int xc = X[0][0] / X[2][0] - shiftX;
			int yc = X[1][0] / X[2][0] - shiftY;
			// TODO: the scaledPatch should be initialized before
			patchData.scaledPatch[patchData.patchScale].at<uchar>(i, j) =
				patchData.orgPatch.at<uchar>(yc, xc);
		}
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
	cv::resize(patchData.scaledPatch[patchData.patchScale], 
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
		if (*itr.confidence[scaleID])
		{
			IDs.push_back(count);
			count++;
		}

	if (IDs.empty())
		return false;
	else
	{
		cv::RNG rng;
		patchID = rng.uniform(0, (int)IDs.size());
		return true;
	}
}

void
superResolutionTracker::superResolution(scaleID, patchID, patchData)
{
	cv::Mat lowPatch = patchData.scaledPatch[scaleID - 1];
	cv::Mat highPatch = (dataset[patchData.faceID].begin() + patchID)->scaledPatch[scaleID];

	cv::Size lowSize(lowPatch.cols, lowPatch.rows);
	cv::Size highSize(highPatch.cols, highPatch.rows);
	cv::Mat count = cv::Mat::zeros(lowSize, CV_32FC1) + 1e-10;
	cv::Mat tarLowPatch = cv::Mat::zeros(lowSize, CV_32FC1);
	cv::Mat light = cv::Mat::ones(lowSize, CV_32FC1);
	cv::Mat high2low(lowSize, CV_32FC1);
	cv::Mat curHighPatch(highSize, CV_8UC1);

	// process
	//
	vpMatrix K = cam.get_K();
	vpMatrix invK = K.inverseByLU();
	vpMatrix invP = faceScaleInfo[patchData.faceID].cMos[scaleID].inverseByLU();
	vpMatrix P = faceScaleInfo[patchData.faceID].cMos[scaleID-1].inverseByLU();
	float depth = 1 / faceScaleInfo[patchData.faceID].invDepth[scaleID];
	vpMatrix trans = K * P * invP * depth * invK;
	for (int i = 0; i < highPatch.rows; i++)
		for (int j = 0; j < highPatch.cols; j++)		
		{
			// project
			vpMatrix x(3, 1);
			x[0][0] = j;
			x[1][0] = i;
			x[2][0] = 1;
			vpMatrix X = trans * x;
			int xc = X[0][0] / X[2][0];
			int yc = X[1][0] / X[2][0];

			// update
			count.at<float>(yc, xc) += 1;
			tarLowPatch.at<float>(yc, xc) += highPatch.at<uchar>(i, j);
		}

	cv::divide(tarLowPatch, count, tarLowPatch);

	findLight(tarLowPatch, lowPatch, light);

	// generate the super resolution patch
	for (int i = 0; i < highPatch.rows; i++)
		for (int j = 0; j < highPatch.cols; j++)		
		{
			// project
			vpMatrix x(3, 1);
			x[0][0] = j;
			x[1][0] = i;
			x[2][0] = 1;
			vpMatrix X = trans * x;
			int xc = X[0][0] / X[2][0];
			int yc = X[1][0] / X[2][0];

			// update
			curHighPatch.at<uchar>(i, j) = highPatch.at<uchar>(i, j) * light.at<float>(yc, xc);
		}
	// save processed data
	patchData.scaledPatch[scaleID] = curHighPatch;	
}

// TODO: don't forget the lock while updating the database
// TODO: check whether the tracking procedure requires the information saved in the scaleInfo structure
void
superResolutionTracker::refreshDataset()
{
	// number of scales, start from the lowest scale
	for (int i = 0; i < numOfPatchScale; i++)	
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
					*itrPatch.confidence[i] = true;
					*itrPatch.isChanged[i] = true;
					omp_unset_lock(&dataLock);
				}
				else
				{
					if (*itrPatch.isChanged[i-1])
					{
						omp_set_lock(&dataLock);
						interpolate(*itrPatch, i);
						*itrPatch.isChanged[i] = true;
						omp_unset_lock(&dataLock);
					}
				}
			}
		}
	}

	// reset the isChanged flags
	for (int i = 0; i < numOfPatchScale; i++)	
		for (size_t j = 0; j < dataset.size(); j++)
			for (std::list<patch>::iterator itrPatch = dataset[j].begin(); itrPatch != dataset[j].end(); ++itrPatch)
				*itrPatch.isChanged[i] = true;
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
	catch (GCException e){
		e.Report();
	}
}

// TODO: use the pointer gramma rewrite all the cv::Mat related codes
