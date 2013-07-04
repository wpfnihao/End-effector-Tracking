/* This class is written by wpf @ Singapore
 * 2013/4/22
 *
 * based on the source code of Gurpinder Singh Sandhu (with the permission of the License Agreement)
 * The original License Agreement is listed below:
 */
/*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install,
 copy or use the software.


                          License Agreement

Copyright (C) 2013, Gurpinder Singh Sandhu, all rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * The name of the copyright holders may not be used to endorse or promote products
    derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall Gurpinder Singh Sandhu be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.*/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>

class KernelBasedTracking
{
	public:
		KernelBasedTracking();
		void retrieveImage(const cv::Mat& img);
		void track(void);
		void initTarget(const int *win_);
		cv::Rect getWindow(void);
	private:
		cv::Mat orig_img;
		cv::Mat target_hist,target_candidate_hist,weight;
		cv::Mat kernel_for_now;
		float kernel_sum;
		float pi1;
		int frame_count;
		int model;
		float alpha;

		int no_of_bins;
		int range;
		int bin_width;

		cv::Rect box;

		int kernel_size;

	protected:
		void create_kernel(cv::Mat &kernel);
		cv::Mat kernel_on_patch(cv::Mat &input_image,int *patch_centre,cv::Mat input_kernel);
		cv::Mat create_target_model(cv::Mat &input_image,cv::Mat &input_kernel);
		cv::Mat detect_object(cv::Mat &input_image,cv::Rect &box);
		cv::Mat assign_weight(cv::Mat &input_image,cv::Mat &target_model,cv::Mat &target_candidate,cv::Rect &box);
		float calc_bhattacharya(cv::Mat &target_model,cv::Mat &target_candidate);
		float trackscale(float scale, cv::Rect& box_, cv::Mat& target_candidate);
};
