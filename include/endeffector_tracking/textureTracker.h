/**
 * @file textureTracker.h
 * @brief Core of the nonparametricTextureLearningTracker
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-15
 */

#include "endeffector_tracking/cadModel.h"

class textureTracker: public cadModel
{
	public:
		/**
		 * @brief initialize the class
		 */
		void init(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_);

		/**
		 * @brief init the patch coordinates
		 */
		void initCoor();

		/**
		 * @brief retrieve patches based on the current pose
		 *
		 * @param img
		 * @param cMo_
		 */
		void retrievePatch(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_, std::map<int, std::vector<cv::Mat> >& curPatch_ = this->curPatch);

		/**
		 * @brief based on the patches database and the current patches update the pose
		 */
		void track(void);

		/**
		 * @brief measure the fitness of the current patches and the database
		 */
		void measureFit(void);

		/**
		 * @brief based on the fitness of the patches and the contours to decide whether to update the patch database
		 */
		void updatePatches(void);

	protected:
		void initCoorOnFace(std::vector<vpPoint>& features, vpMbtPolygon& pyg, int numOfPtsPerFace);
	private:
		/**
		 * @brief current frame
		 */
		cv::Mat curImg;

		/**
		 * @brief numOfPtsPerFace = numOfPtsPerFace * numOfPtsPerFace;
		 */
		int numOfPtsPerFace;

		/**
		 * @brief number of the patches saved for each face.
		 * the default value is 10.
		 */
		int numOfPatch;

		/**
		 * @brief int face IDs
		 * 		  vector<Mat> saved patches
		 */
		std::map<int, std::vector<cv::Mat> > patches;

		/**
		 * @brief current patches need to be optimized
		 */
		std::map<int, std::vector<cv::Mat> > curPatch;

		/**
		 * @brief int face IDs
		 * 		  vector<vpPoint> the coordinates of the patches
		 */
		std::map<int, std::vector<vpPoint> > patchCoor;
};
