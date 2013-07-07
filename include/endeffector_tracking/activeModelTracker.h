/**
 * @file activeModelTracker.h
 * @brief The active model based tracker. The shape prior is obtained from the 
 * 			3d CAD model.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-05
 */

#include <visp/vpMbEdgeTracker.h>
#include <visp/vpImage.h>
#include <sensor_msgs/image_encodings.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpPoseFeatures.h>
#include <image_transport/image_transport.h>

//user defined
#include "endeffector_tracking/cadModel.h"

class activeModelTracker: public cadModel
{
	public:

		/**
		 * @brief call getInitPoints() first before calling this function
		 *
		 * @param cam_
		 * @param cMo_
		 * @param rows_
		 * @param cols_
		 */
		void initialize(const vpCameraParameters& cam_, const vpHomogeneousMatrix& cMo_, int rows_, int cols_);

		void pubRst(cv::Mat& img);

		/**
		 * @brief retrieve the next frame from the webcam for tracking
		 *
		 * @param img
		 */
		inline void retrieveImage(const cv::Mat& img)
		{
			cv::cvtColor(img, curImg, CV_BGR2GRAY);

			processedImg = img.clone();
		}

		/**
		 * @brief the implementation of the active silhouette model
		 */
		void track(void);

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

		void deformSilhouette(vpHomogeneousMatrix& cMo_);

		void plotRst(void);

		void genControlPoints(std::vector<vpPoint>& controlPoints_);

		double lineSlope( cv::Point prePnt, cv::Point nxtPnt, bool& isHorizontal);

		void deformLine(double step, bool isHorizontal, std::vector<vpPoint>& controlPoints_, const cv::Mat& gradient, int rows, int cols, int detectionRange);

		void sobelGradient(const cv::Mat& curImg, cv::Mat& gradient);

		inline void checkIndex(int& cx, int& cy, const int& rows, const int& cols)
		{
			cx = cx < 0 ? 0 : cx;
			cx = cx >= cols ? (cols - 1) : cx;
			cy = cy < 0 ? 0 : cy;
			cy = cy >= rows ? (rows - 1) : cy;
		}	
	private:
		/**
		 * @brief The pose of the tracked object.
		 */
		vpHomogeneousMatrix cMo;		

		int rows, cols;

		vpCameraParameters cam;

		cv::Mat curImg, processedImg;

		/**
		 * @brief map<lineID, controlPoints>
		 * std::vector<vpPoint> 0-1:projected corners; 2-end, controlPoints
		 */
		std::map<int, std::vector<vpPoint> > controlPoints;
};
