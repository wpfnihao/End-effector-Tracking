/**
 * @file superResolutionTracker.h
 * @brief this the header file of the superResolutionTracker class
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-08-08
 */


// ViSP
#include <visp/vpMbEdgeTracker.h>
#include <visp/vpMbEdgeKltTracker.h>
#include <visp/vpImageConvert.h>
#include "visp/vpMeterPixelConversion.h"
#include "visp/vpPixelMeterConversion.h"
#include "visp/vpPose.h"
#include "visp/vpDisplayX.h"
#include "visp/vpPoseFeatures.h"
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/nonfree/features2d.hpp>
// OpenMP
#include <omp.h>
// Graph Cut Optimization
#include "gco/GCoptimization.h"

#include <stdlib.h>

// TODO; change the two structs into class with embedded member functions
// TODO: the klt tracker seems not very stable
class superResolutionTracker: public vpMbEdgeKltTracker
{
	// the definition of the most important structures in the program
	public:
		/**
		 * @brief a key frame is saved in this structure, with its pose when obtained saved in the vpHomogeneousMatrix, and the down-scaled and up=scaled patched saved in the scaledPatch, the int stands for the scale.
		 */
		typedef struct patch_
		{
			// the patch push back into the dataset will have all the following fields filled
			/**
			 * @brief the patch under different scales
			 */
			std::map<int, cv::Mat> scaledPatch;
			/**
			 * @brief the confidence of the patch in the corresponding scale
			 */
			std::map<int, bool> confidence;
			std::map<int, bool> isChanged;

			/**
			 * @brief TODO: highestConfidenceScale is assumed to be handled in the dataset update procedure, check whether it is.
			 */
			int highestConfidenceScale;

			// the patch extracted from the key frame only have the following fields filled
			/**
			 * @brief only the pixels in the patchRect are saved, and the pixels corresponding to the model face are labeled by the mask.
			 */
			int patchScale;
			cv::Mat orgPatch, mask, depth;
			/**
			 * @brief the patchRect here is only used for the orgPatch, for scaledPatch, refer to the faceScaleInfo.faceSizes, and there is offset for scaledPatch, so cv::Size is enough.
			 */
			cv::Rect patchRect;

			int faceID;
			vpHomogeneousMatrix pose;
		}patch;

		/**
		 * @brief save the info for faces under different scales
		 * once the information in this structure is initialized, it should never be changed.
		 * As a result, no data lock is required while processing the structure.
		 * Once initialized, this structure can't be modified
		 */
		typedef struct scaleInfo_
		{
			/**
			 * @brief the pose under which the image of the target face is under the desired scale and position.
			 */
			vpHomogeneousMatrix cMo;
			float depth;
			std::map<int, vpMatrix> Ks;
			/**
			 * @brief the desired image size of the face under the very scale
			 */
			std::map<int, cv::Size> faceSizes;
		}scaleInfo;

		/**
		 * @brief several patches corresponding to a single face are saved in the face structure. For quick accessing and performance while maintaining the structure, pointers are saved. Cares should be taken while processing this structure.
		 */
		typedef std::list<patch> face_t;

		typedef std::map<int, face_t> dataset_t;
	// public member functions
	public:
		superResolutionTracker();
		void track(void);
		void updateDataBase(void);
		void initDataset(void);
		void initialization(cv::Mat& src, std::string config_file, std::string modelName, std::string initName);

		/**
		 * @brief retrieve image and then convert it to gray scale
		 *
		 * @param src
		 */
		inline void retrieveImage(const cv::Mat& src)
		{
			++frameCount;
			preImg = curImg.clone();
			cv::cvtColor(src, curImg, CV_RGB2GRAY);	
			colorImg = src.clone();
		}


		// private member variables
	protected:
		omp_lock_t buffLock;
		omp_lock_t dataLock;

		/**
		 * @brief dataset here is the structure saving all the patches.
		 * the face ID here should be consistent with the index in the polygon class
		 */
		dataset_t dataset;

		cv::Mat curImg, preImg;
		cv::Mat colorImg;

		/**
		 * @brief the size and the pose under which can obtain the size of faces is saved in this structure
		 * This structure should be initialized as soon as the cad model of the target has been loaded.
		 */
		std::map<int, scaleInfo> faceScaleInfo;

		/**
		 * @brief the original patches obtained from the image frame are pushed back into the buff, and wait for processing. Each patch here only has one scale (the original patch).
		 */
		face_t buff;

		/**
		 * @brief face visibility maintained in this structure and avoid compute the visibility every time
		 */
		std::vector<bool> isVisible;

		int numOfPatchScale;
		int maxDownScale;
		int buffSize;
		int numOfPatches;
		int numOfPatchesUsed;
		int rows, cols;
		int superScale;
		int winSize;
		int maxLevel;
		int minFrameCount;
		int frameCount;
		int frame_num;
		/**
		 * @brief the angle define the which face can be identified as visible
		 * the angle is in degree (not rad)
		 */
		int faceAngle;
		double res;


		vpHomogeneousMatrix p_cMo;

		vpMatrix virtualCam, invVirtualCam;

		dataset_t prePatch;
		/**
		 * @brief a copy of the whole dataset only including the tracking related patches
		 */
		dataset_t dataPatches;

		// for display only
		vpDisplayX disp;
		vpImage<uchar> I;

	// private member functions	
	protected:
		void pushDataIntoBuff(patch& patchData);
		bool getDataFromBuff(patch& patchData);	
		//TODO: check lock
		void updataDataset(patch& patchData);

		/**
		 * @brief in this function, there is no need to lock the dataset, since the function only read the data in the dataset, and won't make any change.
		 *
		 * @param patchData
		 */
		void processKeyPatch(patch& patchData);

		/**
		 * @brief only the cv::Mat mask (which actually means this orgPatch and depth are also obtained while initialization) information should be filled before call this function, please make sure this.
		 *
		 * @param patchData the patch requires for its scale information
		 */
		void findPatchScale(patch& patchData);

		/**
		 * @brief only find the scale of the patch i, without obtaining the whole patch
		 *
		 * @param i
		 *
		 * @return 
		 */
		int findPatchScale(int i);

		/**
		 * @brief the initial scale patch based on the original patch scale found
		 *
		 * @param patchData
		 */
		void genOrgScalePatch(patch& patchData);

		/**
		 * @brief generate all the scale patches using the org scale patch
		 *
		 * @param patchData
		 * @param scaleID
		 */
		void genRestScalePatch(patch& patchData, int scaleID);

		/**
		 * @brief if the scaleID of the curPatch is smaller than the org scale patch, we simply interpolate the patch
		 *
		 * @param patchData
		 * @param scaleID
		 */
		inline void interpolate(patch& patchData, int scaleID);

		/**
		 * @brief use the patch database to super resolute the current patch
		 *
		 * @param scaleID
		 * @param patchID
		 * @param patchData
		 */
		void superResolution(int scaleID, int patchID, patch& patchData);

		/**
		 * @brief if new patch has been added to the database, we should refresh the whole database based on the new patch.
		 * TODO: check lock
		 */
		void refreshDataset(void);

		/**
		 * @brief the key function based on the graph cut optimization method to super resolute the current patch
		 *
		 * @param tarLowPatch NOTE here the tarLowPatch is CV_32FC1
		 * @param lowPatch    NOTE here the lowPatch    is CV_8UC1
		 * @param light
		 */
		void findLight(const cv::Mat& tarLowPatch,const cv::Mat& lowPatch, cv::Mat& light);

		/**
		 * @brief find any patches in the database has the larger confidence scale than the orgPatch, if so, return one of the patch id and true, else, return false.
		 *
		 * @param scaleID
		 * @param patchID
		 * @param patchData
		 *
		 * @return 
		 */
		bool findConfidence(int scaleID, int& patchID, patch& patchData);

		/**
		 * @brief // find the corresponding patches, and copy them out of the dataset // then project them onto the current pose with the virtual camera
		 *
		 * @param faceID 		the id of the face under investigation
		 * @param patchScale 	scale of current patch in the frame, used to choose the patches in the database with higher scale
		 * @param pose 			current pose
		 * @param patchList 	where the found, projected patches are copied to 
		 */
		void findCopyProcessPatch(
				int faceID,
				int patchScale,
				vpHomogeneousMatrix pose,
				std::list<patch>& patchList);

		/**
		 * @brief the following information has been copied:
		 * highestConfidenceScale patch
		 * highestConfidenceScale
		 * patchScale
		 * faceID
		 * pose
		 *
		 * This function only copy the information used for the tracking procedure
		 * while the deepCopyPrePatch function copy all the information obtained from the frame
		 * NOTE: dataLock not used in this function, every time you call this function, you MUST make sure you have employed the dataLock explicitly
		 * @param dataPatches
		 * @param src
		 */
		void deepCopyPatch(std::list<patch>& dataPatches, patch& src);

		/**
		 * @brief based on the information provided in the curPatch, project the dataPatches onto the image using the curPatch.pose, and the orgPatch as well as its corresponding information is over-written.
		 * the image patch is projected based on the virtual camera
		 * the following information has been processed and saved during this function:
		 * pose not used, since using the cMo is the same
		 * highestConfidenceScale not used
		 * orgPatch used
		 * depth used
		 * mask used
		 *
		 * @param curPatch
		 * @param dataPatches
		 */
		void projPatch(vpHomogeneousMatrix pose, int id, std::list<patch>& dataPatches);

		inline void backProj(
				const vpMatrix& invK, 
				const vpMatrix& invP, 
				float 			depth, 
				float 			ix, 
				float 			iy, 
				vpPoint&        p);

			/**
			 * @brief change a pixel from one frame to another with the Z-buff
			 */
		inline void projPoint(
					const vpMatrix& invK, 
					const vpMatrix& PinvP, 
					const vpMatrix& K, 
					float 			depth, 
					float 			ix, 
					float 			iy, 
					int& 			xc, 
					int& 			yc, 
					float& 			Z);

		// old api
//			/**
//			 * @brief change a pixel from one frame to another with the Z-buff
//			 */
//		inline void projPoint(
//					const vpMatrix& invK, 
//					const vpMatrix& invP, 
//					const vpHomogeneousMatrix& P, 
//					const vpMatrix& K, 
//					float 			depth, 
//					float 			ix, 
//					float 			iy, 
//					int& 			xc, 
//					int& 			yc, 
//					float& 			Z);

		/**
		 * @brief obtain a face patch based on the current image and pose
		 * The following information saved in the patch:
		 * patchRect
		 * faceID
		 * pose
		 * orgPatch
		 * mask
		 * depth
		 * patchScale
		 *
		 * @param faceID
		 *
		 * @return 
		 */
		void obtainPatch(int faceID, patch& p);

		// obsoleted function
		//float calcDepth(
		//		const std::vector<cv::Point>& p, 
		//		const std::vector<float>& depth,
		//		cv::Point cp);

		/**
		 * @brief this function and the following prepCalcDepth function are ultimately optimized for performance.
		 *
		 */
		float calcDepth(
				vpMatrix& A,
				float detx,
				float* coefficient,
				vpMatrix invP,
				vpMatrix invK,
				cv::Point cp
				);

		/**
		 * @brief call this function first before using the calcDepth function
		 *
		 * @param vp
		 * @param A
		 * @param b
		 */
		void prepCalcDepth(vpPoint* vp, vpMatrix& A, vpMatrix& b, float* coefficient);

		void deepCopyPrePatch(patch& src, patch& p);

		/**
		 * @brief the mean value of a batch of points
		 *
		 * @param p
		 *
		 * @return 
		 */
		cv::Point meanPoint(const std::vector<cv::Point>& p);

		/**
		 * @brief the core function of the whole tracking procedure
		 *
		 * @param img 			one scale upped image of the current frame
		 * @param prePatch 		previous frame
		 * @param dataPatches 	related patched in the database
		 */
		void optimizePose(cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches);

		bool isKeyFrame(void);

		void initFaceScaleInfo(void);

		/**
		 * @brief generate the virtual camera to produce different scales of the image
		 *
		 * @param K input camera
		 * @param rate the scale rate
		 *
		 * @return the scaled camera matrix K
		 */
		vpMatrix scaleCam(vpMatrix K, float rate);

		/**
		 * @brief the virtual camera in the largest patch scale is set to the same as the true camera,
		 * 			while the virtual cameras in the lower scale are set by using the scaleCam() function.
		 */
		void initFaceScaleVirtualCam(void);
		void initFaceScaleSize(void);
		/**
		 * @brief the face scale pose is the same for all the scales.
		 */
		void initFaceScalePose(void);
		void initFaceScaleDepth(void);
		/**
		 * @brief the virtual camera her is one scale upped camera
		 *
		 * @return one scale upped camera compared with the true matrix of the webcam
		 */
		vpMatrix getVirtualCam(void);

		/**
		 * @brief get the scale rate between each scale
		 *
		 * @param maxDownScale  	maximum scale rate between the smallest and biggest patches
		 * @param numOfPatchScale 	number of patch scales
		 *
		 * @return 					the scale rate between each scale
		 */
		inline float getRate(float maxDownScale, float numOfPatchScale)
		{
			return pow(maxDownScale, 1.0/(numOfPatchScale-1));
		}

		inline float pointDistance3D(const vpPoint& p1, const vpPoint& p2);

		void findMinMax(std::vector<cv::Point> corners, cv::Point& lu, cv::Point& rb);

		void findStableFeatures(
				std::vector<bool>& 		    		finalStatus, 
				const std::vector<cv::Point2f>& 	corners, 
				const std::vector<cv::Point2f>& 	bFeatures,
				const std::vector<uchar>& 			fStatus, 
				const std::vector<uchar>& 			bStatus,
				const std::vector<float>& 			fErr,
				float 								th,
				float 								rate
				);

		void findClosestPatch(
				vpPoseVector& 		vp, 
				std::vector<int>& 	id, 
				std::vector<int>& 	idx, 
				int 				numOfPatchesUsed,
				int 				faceID);

		inline int findMinCost(float tar, int pos, const cv::Mat& img , int rows, int cols, int ws);

		void findStableFeaturesWithRate(
				const std::vector<float>& 	 	fErr, 
				const std::vector<uchar>& 	 	fStatus, 
				std::vector<bool>& 			 	finalStatus, 
				const std::vector<cv::Point2f>& corners,
				const std::vector<cv::Point2f>& cFeatures,
				float 						 	rate);

		/**
		 * @brief // virtual inherit the vpMbTracker class, so only one cMo is available here // 90% of the codes here are copied from the vpMbEdgeKltTracker class
		 *
		 * @param _nbInfos
		 * @param I
		 * @param pf
		 * @param w_mbt
		 * @param w_klt
		 * @param scale_ 		the mean patch scale chosen for the model, the scale here is for balancing between the edge based tracker and feature point based tracker. However, the technology is not used yet.
		 * @param lvl
		 *
		 * @return residuals
		 */
		double computeVVS(
				unsigned int _nbInfos,
				const vpImage<unsigned char>& I, 
				vpPoseFeatures& pf,
				vpColVector &w_mbt, 
				vpColVector &w_klt, 
				float scale_,
				const unsigned int lvl = 0);

		/**
		 * @brief 				calculate the pose of the current frame
		 *
		 * @param I
		 * @param scale_ 		the mean patch scale chosen for the model, the scale here is for balancing between the edge based tracker and feature point based tracker. However, the technology is not used yet.
		 * @param img 			one scale upped image of the current frame
		 * @param prePatch 		previous frame
		 * @param dataPatches 	patches in the dataset
		 *
		 * @return 				residuals
		 */
		double getPose(const vpImage<unsigned char>& I, float scale_, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches);

		inline double pointDistance2D(const cv::Point& p1, const cv::Point& p2);

		/**
		 * @brief the tracking procedure is actually done here
		 *
		 * @param featuresComputePose
		 * @param img
		 * @param prePatch 		previous frame
		 * @param dataPatches 	patches in the dataset
		 */
		void trackPatch(vpPoseFeatures& featuresComputePose, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches);

		/**
		 * @brief the tracking procedure is actually done here
		 *        this is the version based on the orb detector
		 *
		 * @param featuresComputePose
		 * @param img
		 * @param prePatch 		previous frame
		 * @param dataPatches 	patches in the dataset
		 */
		void trackPatchOrb(vpPoseFeatures& featuresComputePose, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches);

		/**
		 * @brief the tracking procedure is actually done here
		 *        this is the version based on the sift detector
		 *
		 * @param featuresComputePose
		 * @param img
		 * @param prePatch 		previous frame
		 * @param dataPatches 	patches in the dataset
		 */
		void trackPatchSIFT(vpPoseFeatures& featuresComputePose, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches);

		/**
		 * @brief the tracking procedure is actually done here
		 *        this is the version based on the surf detector
		 *
		 * @param featuresComputePose
		 * @param img
		 * @param prePatch 		previous frame
		 * @param dataPatches 	patches in the dataset
		 */
		void trackPatchSURF(vpPoseFeatures& featuresComputePose, cv::Mat& img, dataset_t& prePatch, dataset_t& dataPatches);

		void findCorner(cv::Mat& mask, cv::Point& upLeft, cv::Point& rightbottom);

		inline void genMask(cv::Mat& curMask, cv::Point& upLeft, cv::Point& rightbottom, int erodeSize);

		double CorrNormedWithMask(cv::Mat& src1, cv::Mat& src2, cv::Mat& mask);
		double MADWithMask(cv::Mat& src1, cv::Mat& src2, cv::Mat& mask);
	private:
};
