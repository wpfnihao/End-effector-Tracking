/**
 * @file superResolutionTracker.h
 * @brief this the header file of the superResolutionTracker class
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-08-08
 */


// ViSP
#include <visp/vpMbEdgeTracker.h>
// OpenCV
#include <opencv2/core/core.hpp>
// OpenMP
#include <omp.h>

class superResolutionTracker: public vpMbEdgeTracker
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
			std::map<int, int> confidence;

			int patchScale;
			// the patch extracted from the key frame only have the following fields filled
			/**
			 * @brief only the pixels in the patchRect are saved, and the pixels corresponding to the model face are labeled by the mask.
			 */
			cv::Mat orgPatch, mask;
			cv::Rect patchRect;

			int faceID;
			vpHomogeneousMatrix pose;
		}patch;

		/**
		 * @brief save the info for faces under different scales
		 */
		typedef struct scaleInfo_
		{
			/**
			 * @brief the pose under which the image of the target face is under the desired scale and position.
			 */
			std::map<int, vpHomogeneousMatrix> cMos;
			/**
			 * @brief the desired image size of the face under the very scale
			 */
			std::map<int, cv::Rect> faceSizes;
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

	// private member variables
	protected:
		omp_lock_t buffLock;
		omp_lock_t dataLock;

		/**
		 * @brief dataset here is the structure saving all the patches.
		 * the face ID here should be consistent with the index in the polygon class
		 */
		dataset_t dataset;

		/**
		 * @brief the size and the pose under which can obtain the size of faces is saved in this structure
		 * This structure should be initialized as soon as the cad model of the target has been loaded.
		 */
		std::map<int, scaleInfo> faceScaleInfo;

		/**
		 * @brief the original patches obtained from the image frame are pushed back into the buff, and wait for processing. Each patch here only has one scale (the original patch).
		 */
		face_t buff;

		int numOfPatchScale;
		int buffSize;
		int numOfPatches;
		int numOfconfidence;
		int rows, cols;

		vpCameraParameters cam;

	// private member functions	
	protected:
		void pushDataIntoBuff(patch& patchData);
		bool getDataFromBuff(patch& patchData);	
		void updataDataset(patch& patchData);
		/**
		 * @brief in this function, there is no need to lock the dataset, since the function only read the data in the dataset, and won't make any change.
		 *
		 * @param patchData
		 */
		void processKeyPatch(patch& patchData);

		int findPatchScale(patch& patchData);
	private:
};
