/*
 * John Hewson, 2018
 * Horizon Segmentation
 */

#include <string>

#include <opencv2/core/core.hpp>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

enum Label {
  GROUND = 0,
  CLEAR_SKY = 1,
  CLOUD = 2,
  BUILDING = 3,
  WATER = 4
};

void processImage(const std::string& image_path, const std::string& out_path, std::unique_ptr<tensorflow::Session>& session);
void readImage(const std::string& file_name, tensorflow::Tensor& out_tensor, cv::Mat& out_image);
void runCRF(const  tensorflow::Tensor& tf_scores, tensorflow::Tensor tf_image, Eigen::Tensor<float, 3, Eigen::RowMajor>& output);
void drawLabel(cv::Mat& image, cv::Mat& labels, cv::Mat& probs, Label label, cv::Scalar color, cv::Mat& dst);
void drawMetric(const std::string& name, float x, float y, cv::Mat& dst);
void drawStatistics(const cv::Mat& src_image, const Eigen::Matrix<Label, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& labels_matrix, cv::Mat& dst);
