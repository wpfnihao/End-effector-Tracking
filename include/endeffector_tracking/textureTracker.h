/**
 * @file textureTracker.h
 * @brief Core of the nonparametricTextureLearningTracker
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-15
 */

#include <visp/vpImageConvert.h>
#include "endeffector_tracking/cadModel.h"

// TODO: add an call example here

class textureTracker: public cadModel
{
	public:
		enum type
		{
			TYPE_TEXTURE,
			TYPE_HYBRID,
			TYPE_EDGE
		};

		textureTracker()
		{}

		/**
		 * @brief initialize the class
		 */
		void init(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_);


		/**
		 * @brief based on the patches database and the current patches update the pose
		 */
		void track(const cv::Mat& img, const cv::Mat& grad);

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
		 *
		 * @param isUpdate
		 *
		 * @return 
		 */
		bool measureFit(bool isUpdate);

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
		void retrievePatch(const cv::Mat& img, vpHomogeneousMatrix& cMo_, vpCameraParameters& cam_, int scale, bool isDatabase = false);

		/**
		 * @brief optimize pose based on the nonparametric texture fitness
		 *
		 * @param img
		 */
		void optimizePose(const cv::Mat& img, int scale, int itr, vpRobust& robust, vpColVector& w, vpColVector& res);

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

		double meanShiftMax(int intensity, int faceID, int index);

		cv::Mat meanShift2(int intensity, int grad, int faceID, int index);

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

		inline cv::Mat gradientImage2(const cv::Mat& img, const cv::Mat& gradient, int x, int y);

		/**
		 * @brief image Jacobian links the pose vector and the pixel location
		 *
		 * @param p
		 *
		 * @return 
		 */
		inline cv::Mat jacobianImage(vpPoint p, int scale);

		inline void stackJacobian(cv::Mat& Jacobian, cv::Mat& GJ, int count);

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

		/**
		 * @brief find the difference between current pixel and the patches database
		 *
		 * @param intensity
		 * @param faceID
		 * @param index
		 *
		 * @return 
		 */
		inline float pixelDiff(int intensity, int faceID, int index);

		/**
		 * @brief test whether the pixel using the index is located at the edge
		 *
		 * @param index
		 *
		 * @return 
		 */
		inline bool isEdge(int index);

		void initLines(void);

		void MovingEdgeBasedTracker(cv::Mat& JacobianMe, cv::Mat& eMe);

		/**
		 * @brief dst here may be the same as the src1
		 *
		 * @param src1
		 * @param src2
		 * @param dst
		 */
		void stackMatrix(cv::Mat& src1, cv::Mat& src2, cv::Mat& dst);
	private:
		/**
		 * @brief current frame
		 */
		cv::Mat curImg;

		/**
		 * @brief the gradient map of the curImg, in the CV_8UC1 format
		 */
		cv::Mat gradient;

		vpHomogeneousMatrix cMo, p_cMo;
		vpCameraParameters cam;

		double gStep, minStep, maxStep;

		/**
		 * @brief numOfPtsPerFace = numOfPtsPerFace * numOfPtsPerFace;
		 */
		int numOfPtsPerFace;

		/**
		 * @brief multi-resolution scales used
		 */
		int scales;

		/**
		 * @brief number of the patches saved for each face.
		 * the default value is 10.
		 */
		int numOfPatch;

		/**
		 * @brief difference between the curPatch and the patches database
		 */
		float curdiff;

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

		// the following member variables are used for the moving edge tracker
		vpMbtDistanceLine lines[12];	
		vpMe mes[12];
		vpImage<unsigned char> vpI;

		float residual;
		bool stopFlag;

		// for robustness
		cv::Mat W;
		int col;
		int nbFeatures;

		type method;
};