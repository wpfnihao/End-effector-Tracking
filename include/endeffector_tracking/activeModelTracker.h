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

class activeModelTracker
{
	public:

		void initialize();

		void pubRst();

		void retrieveImage();

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

	private:
		/**
		 * @brief The pose of the tracked object.
		 */
		vpHomogeneousMatrix cMo;		
};
