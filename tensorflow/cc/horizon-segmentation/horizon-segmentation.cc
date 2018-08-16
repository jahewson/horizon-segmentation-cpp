/*
 * John Hewson, 2018
 * Horizon Segmentation
 */

#include "horizon-segmentation.h"

#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <glob.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <densecrf.h>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: horizon-segmentation <graph.pb> <path/to/*.png>" << std::endl;
    return 1;
  }

  const std::string graph_path(argv[1]);
  const char* in_glob = argv[2];
  
  // load the frozen graph
  tf::GraphDef graph_def;
  TF_CHECK_OK(ReadBinaryProto(tf::Env::Default(), graph_path, &graph_def));

  std::unique_ptr<tf::Session> session(tf::NewSession(tf::SessionOptions()));
  TF_CHECK_OK(session->Create(graph_def));

  // iterate over files with posix glob
  glob_t glob_result;
  glob(in_glob, GLOB_TILDE, nullptr, &glob_result);

  for (unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
    std::cout << glob_result.gl_pathv[i] << std::endl;

    std::string image_path(glob_result.gl_pathv[i]);
    const auto out_path = image_path.substr(image_path.find_last_of("/\\") + 1);
    processImage(image_path, out_path, session);
  }

  TF_CHECK_OK(session->Close());

  return 0;
}

void processImage(const std::string& image_path, const std::string& out_path, std::unique_ptr<tf::Session>& session) {
  // read image from disk
  tf::Tensor tf_input;
  cv::Mat src_image;
  readImage(image_path, tf_input, src_image);

  // DCNN inference
  std::vector<tf::Tensor> tf_outputs;
  TF_CHECK_OK(session->Run({{"ImageTensor:0", tf_input}},
                            {"SemanticPredictions:0"}, {}, &tf_outputs));

  const auto tf_output = tf_outputs[0];
  const int height = tf_output.dim_size(1);
  const int width = tf_output.dim_size(2);
  
  // CRF post-processing
  Tensor<float, 3, RowMajor> crf_probs;
  runCRF(tf_output, tf_input, crf_probs);

  // compute final labels via maximum likelihood
  const Tensor<Label, 2, RowMajor> labels_tensor = crf_probs.argmax(2).cast<Label>();
  const auto labels(Map<const Matrix<Label, Dynamic, Dynamic, RowMajor>>(labels_tensor.data(), height, width));

  // find and draw the horizon
  cv::Mat dst;
  src_image.copyTo(dst);

  const float x_scale = src_image.size().width / static_cast<float>(width);
  const float y_scale = src_image.size().height / static_cast<float>(height);

  cv::Point points[width];
  for (int x = 0; x < width; x++) {
    int y_max = 0;
    for (int y = 0; y < height; y++) {
      auto label = labels(y, x);
      if (label == CLEAR_SKY || label == CLOUD) {
        y_max = y;
      }
    }
    points[x] = cvPoint(x * x_scale, y_max * y_scale);
  }

  const cv::Point *points_ptr = points;
  const int num_points[] = { width };
  cv::polylines(dst, &points_ptr, num_points, 1, false, cvScalar(0, 255, 0), 2, CV_AA);

  // draw the cloud segments
  cv::Mat cv_labels_8(height, width, CV_8UC1);
  cv::Mat probs_cloud(height, width, CV_32F);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      cv_labels_8.at<uchar>(y, x) = labels(y, x); // long -> byte
      probs_cloud.at<float>(y, x, 0) = crf_probs(y, x, CLOUD); // alpha channel
    }
  }
  drawLabel(dst, cv_labels_8, probs_cloud, CLOUD, cvScalar(0, 255, 0), dst);

  // compute some simple statistics from the image and segmentation
  drawStatistics(src_image, labels, dst);

  // save the annotated image to disk
  cv::imwrite(out_path, dst);
}

void drawStatistics(const cv::Mat& src_image, const Matrix<Label, Dynamic, Dynamic, RowMajor>& labels_matrix, cv::Mat& dst) {
  const auto labels = labels_matrix.array();
  
  // compute some basic segmentation ratios
  const float sky_count = (labels == CLEAR_SKY || labels == CLOUD).count();
  const float cloud_ratio = (labels == CLOUD).count() / sky_count;
  const float strict_ground_count = (labels == GROUND || labels == BUILDING).count();
  const float building_ratio = (labels == BUILDING).count() / strict_ground_count;
  const float ground_count = (labels == GROUND || labels == BUILDING || labels == WATER).count();
  const float water_ratio = (labels == WATER).count() / ground_count;

  // classify day/night
  cv::Mat image_gray;
  src_image.convertTo(image_gray, CV_32F, 1.0 / 255);
  const float gray_mean = cv::mean(image_gray)[0];
  const bool is_night = gray_mean < 0.10;
  const std::string night_class = is_night ? "night" : "day";

  // classify clouds using National Weather Service ratios
  const std::string cloud_class =
    cloud_ratio >= 1.0 / 8.0 ? (is_night ? "mostly clear" : "mostly sunny") :
    cloud_ratio >= 3.0 / 5.0 ? "partly cloudy" :
    cloud_ratio >= 3.0 / 4.0 ? "mostly cloudy" :
    cloud_ratio >= 7.0 / 8.0 ? "cloudy": "clear";
  
  // classify buildings
  const std::string building_class =
    building_ratio >= 0.75 ? "dense urban" :
    building_ratio >= 0.50 ? "urban" :
    building_ratio >= 0.25 ? "some urban" : "rural";

  // classify water
  const std::string water_class =
    water_ratio >= 0.75 ? "over water" :
    water_ratio >= 0.50 ? "near water" :
    water_ratio >= 0.10 ? "some water" :
    water_ratio >  0.0  ? "negligible" : "none";
  
  // draw the background rectangle into a buffer
  float x = 15, y = 15;
  const int line_height = 20;

  cv::Mat buffer;
  dst.copyTo(buffer);
  cv::rectangle(buffer, cvRect(x, y, 200, line_height * 4.5), cvScalar(0, 0, 0), CV_FILLED);

  // draw the buffer with alpha
  const double alpha = 0.3;
  cv::addWeighted(buffer, alpha, dst, (1 - alpha), 0, dst);
  x += 10;

  // draw the label text
  drawMetric("weather: " + cloud_class, x, y += line_height, dst);
  drawMetric("landscape: " + building_class, x, y += line_height, dst);
  drawMetric("water: " + water_class, x, y += line_height, dst);
  drawMetric("time: " + night_class, x, y += line_height, dst);
}

void drawMetric(const std::string& name, float x, float y, cv::Mat& dst) {
  cv::putText(dst, name, cvPoint(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 255, 0), 1, CV_AA);
}

void readImage(const std::string& file_name, tf::Tensor& out_tensor, cv::Mat& out_image) {
  // load file from disk (we assume this never fails)
  out_image = cv::imread(file_name, CV_LOAD_IMAGE_COLOR);
  const auto size = out_image.size();

  // make an RGB copy for TensorFlow
  cv::Mat image_rgb;
  cv::cvtColor(out_image, image_rgb, CV_BGR2RGB);

  // downscale, preserving aspect ratio
  const float input_size = 513; // hardcoded in the DCNN
  const float ratio = input_size / fmax(size.width, size.height);
  const int width = round(size.width * ratio);
  const int height = round(size.height * ratio);

  // output is a cv Mat backed by a tf tensor
  out_tensor = tf::Tensor(tf::DT_UINT8, tf::TensorShape({1, height, width, 3}));
  cv::Mat cv_out_tensor(height, width, CV_8UC3, out_tensor.flat<uchar>().data());

  cv::resize(image_rgb, cv_out_tensor, cv_out_tensor.size());
}

void runCRF(const tf::Tensor& tf_scores, tf::Tensor tf_image, Tensor<float, 3, RowMajor>& output) {
  const int height = tf_scores.dim_size(1);
  const int width = tf_scores.dim_size(2);
  const int num_channels = tf_scores.dim_size(3);

  // CRF parameters
  const int max_iterations = 5;

  // gaussian
  const float g_potts = 3;
  const float g_std = 3;  

  // bilateral
  const float bl_potts = 5;
  const float bl_xy_std = 70;
  const float bl_rgb_std = 5;

  // scaled input
  cv::Mat src_image(height, width, CV_8UC3, tf_image.flat<uchar>().data());

  // map (h, w, ch) scores tensor to a (h*w, ch) matrix
  const auto tensor = tf_scores.tensor<float, 4>();
  const MatrixXf scores(Map<const MatrixXf>(tensor.data(), num_channels, height * width));

  // negate for CRF
  const MatrixXf unary = -scores;

  DenseCRF2D crf(width, height, num_channels);
  
  // unary potential: row-major (h*w, ch) matrix
  crf.setUnaryEnergy(unary);
  
  // color-independent (xy)
  crf.addPairwiseGaussian(g_std, g_std, new PottsCompatibility(g_potts)); // freed by call

  // color-dependent term (xyrgb)
  crf.addPairwiseBilateral(bl_xy_std, bl_xy_std,
                           bl_rgb_std, bl_rgb_std, bl_rgb_std,
                           src_image.data, new PottsCompatibility(bl_potts)); // freed by call

  // mean field inference
  MatrixXf crf_probs = crf.inference(max_iterations);

  // map the (h*w, ch) matrix back to a (h, w, ch) tensor
  output = Tensor<float, 3, RowMajor>(height, width, num_channels);
  std::copy_n(crf_probs.data(), height * width * num_channels, output.data());
}

void drawLabel(cv::Mat& image, cv::Mat& labels, cv::Mat& probs, Label label, cv::Scalar color, cv::Mat& dst) {
  // upsize labels to match source image
  cv::Mat labels_scaled;
  cv::resize(labels, labels_scaled, image.size(), cv::INTER_NEAREST);

  cv::Mat probs_scaled;
  cv::resize(probs, probs_scaled, image.size());

  // mask out the channel of interest
  cv::Mat mask;
  cv::inRange(labels_scaled, label, label, mask);

  cv::Mat overlay_color = cv::Mat::zeros(image.size(), CV_8UC3);
  overlay_color.setTo(color, mask);

  // use the proability as the alpha, for smooth edges
  cv::Mat alpha;
  probs_scaled.copyTo(alpha, mask);
  alpha = alpha * 0.33;

  cv::Mat alpha_bgr;
  cv::cvtColor(alpha, alpha_bgr, CV_GRAY2BGR);
  
  // alpha blend
  cv::Mat background, foreground, composed;
  overlay_color.convertTo(foreground, CV_32FC3);
  image.convertTo(background, CV_32FC3);
  
  cv::multiply(alpha_bgr, foreground, foreground);
  cv::multiply(cv::Scalar::all(1.0) - alpha_bgr, background, background);
  cv::add(foreground, background, composed); 

  composed.convertTo(dst, CV_8UC3);
}
