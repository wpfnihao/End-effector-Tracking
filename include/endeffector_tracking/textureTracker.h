/**
 * @file textureTracker.h
 * @brief Core of the nonparametricTextureLearningTracker
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-15
 */

#include "endeffector_tracking/cadModel.h"

// TODO: add an call example here

class textureTracker: public cadModel
{
	public:
		/**
		 * @brief initialize the class
		 */
		void init(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_);


		/**
		 * @brief based on the patches database and the current patches update the pose
		 */
		void track(const cv::Mat& img);


		/**
		 * @brief based on the fitness of the patches and the contours to decide whether to update the patch database
		 */
		void updatePatches(vpHomogeneousMatrix& cMo_);

		/**
		 * @brief get the pose from the upper level class, which usually provided by other trackers
		 *
		 * @param cMo_ the pose
		 */
		inline void getPose(vpHomogeneousMatrix& cMo_)
		{
			this->cMo = cMo_;
		}

		/**
		 * @brief publish the pose calculated by this tracker, which can be used by other trackers to refine the result
		 *
		 * @param cMo_ the pose
		 */
		inline void pubPose(vpHomogeneousMatrix& cMo_)
		{
			cMo_ = this->cMo;
		}
	protected:
		/**
		 * @brief measure the fitness of the current patches and the database
		 */
		bool measureFit(void);

		/**
		 * @brief init the patch coordinates
		 */
		void initCoor(void);

		/**
		 * @brief called by the initCoor function
		 *
		 * @param features
		 * @param pyg
		 * @param numOfPtsPerFace
		 */
		void initCoorOnFace(std::vector<vpPoint>& features, vpMbtPolygon& pyg, int numOfPtsPerFace);

		/**
		 * @brief retrieve patches based on the current pose
		 *
		 * @param img
		 * @param cMo_
		 */
		void retrievePatch(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_, bool isDatabase = false);

		/**
		 * @brief optimize pose based on the nonparametric texture fitness
		 *
		 * @param img
		 */
		void optimizePose(const cv::Mat& img);

		/**
		 * @brief the gradient of the pixel based on the texture database using the mean shift method 
		 *
		 * @param intensity
		 * @param faceID
		 * @param index
		 *
		 * @return 
		 */
		double meanShift(int intensity, int faceID, int index);
		
		/**
		 * @brief image gradient at the pixel (x, y)
		 *
		 * @param img
		 * @param x
		 * @param y
		 *
		 * @return 
		 */
		inline cv::Mat gradientImage(const cv::Mat& img,int x, int y);

		/**
		 * @brief image Jacobian links the pose vector and the pixel location
		 *
		 * @param p
		 *
		 * @return 
		 */
		inline cv::Mat jacobianImage(vpPoint p);

		/**
		 * @brief mean value of a vector of matrixs
		 *
		 * @param m
		 *
		 * @return 
		 */
		cv::Mat meanMat(std::vector<cv::Mat>& m);

		/**
		 * @brief multiply a matrix m with a scalar scale pixel-wise
		 *
		 * @param m
		 * @param scale
		 *
		 * @return 
		 */
		inline cv::Mat scaleMat(cv::Mat m, double scale);

		/**
		 * @brief the kernel density of x under xi with the standard deviation delta
		 *
		 * @param x
		 * @param xi
		 * @param delta
		 *
		 * @return 
		 */
		inline double kernelDensity(double x, double xi, double delta);

	private:
		/**
		 * @brief current frame
		 */
		cv::Mat curImg;

		vpHomogeneousMatrix cMo;
		vpCameraParameters cam;

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
		std::map<int, std::vector<std::vector<unsigned char> > > patches;

		/**
		 * @brief current patches need to be optimized
		 */
		std::map<int, std::vector<unsigned char> > curPatch;

		/**
		 * @brief int face IDs
		 * 		  vector<vpPoint> the coordinates of the patches
		 */
		std::map<int, std::vector<vpPoint> > patchCoor;
};
