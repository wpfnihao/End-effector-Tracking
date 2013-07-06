/**
 * @file cadModel.h
 * @brief This is the cadModel class maintaining the cad model.
 * @author Pengfei Wu - wpfnihao@gmail.com
 * @version 1.0
 * @date 2013-07-06
 */

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//visp headers
//the vpMbTracker class, used to get the initial pose
#include <visp/vpMbEdgeTracker.h>
//the vpPose classes, used to calc pose from features
//the vpHomogeneousMatrix.h is also included in this head file
#include <visp/vpPose.h>
#include <visp/vpImage.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpPoseFeatures.h>

class cadModel
{
	protected:
	/**
	 * @brief  initial points defined in the .init file
	 */
	std::vector<cv::Point3f> initP;

	// the cube
	/**
	 * @brief  cube corners
	 */
	std::vector<cv::Point3f> corners;

	/**
	 * @brief  the polygon form description of the cube
	 * 			VERY IMPORTANT:
	 * 			new operator used in vpMbtPolygon class and operator=() is NOT redefined in the class, which means the vpMbtPolygon instants should be instantiated in the constructor of this class, if not, the data will NOT be properly copied during push_back and the whole program will crash.
	 * 			vpMbtPolygon pyg1, pyg2, pyg3, pyg4, pyg5, pyg6;
	 */
	vpMbtPolygon pyg[6];

	/**
	 * @brief  projected corners
	 */
	std::vector<cv::Point2f> prjCorners;

	protected:
	/**
	 * @brief  project the model into the image using the provided  pose (usually the predicted one or the measured one)
	 *
	 * @param cMo_
	 */
	void projectModel(const vpHomogeneousMatrix& cMo_, const vpCameraParameters& cam);

	/**
	 * @brief  I still don't know how to read the model from the wrl or cad or cao file
	 * However, it is sufficient to get the cube from the init file
	 * if in the future, more complex model is tracked, I should reimplement this function
	 * the 8 corners of the cube is initialized.
	 * the 6 faces of the cube is initialized.
	 * this function will be called before initialize.
	 *
	 * @param initP :the initial points from the *.init file
	 */
	void initModel(void);

	// the member functions
	public:

	void getInitPoints(const std::string& init_file);
};
