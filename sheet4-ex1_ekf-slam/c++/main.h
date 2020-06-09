#ifndef EKF_SLAM_MAIN_H
#define EKF_SLAM_MAIN_H

#include <cmath>
#include <Eigen/Dense>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#include <fstream>
#include <iostream>
#include <vector>

struct Pose {
  int time;
  double x0;
  double y0;
  double theta0;
};

struct OdometryData {
  int time;
  double rot1;
  double trans;
  double rot2;
};

struct SensorData {
  int time;
  int id;
  double range;
  double bearing;
};

struct Landmark {
  int id;
  double x;
  double y;
  Eigen::Matrix3d C;
  int idx_prm_x;
  int idx_prm_y;
  bool initialized{false};
};

int main();

std::vector<std::vector<std::string>> ReadDataFromFile(const std::string &path_to_data);

std::tuple<std::vector<OdometryData>, std::vector<SensorData>> ExtractOdometryAndSensorData(
    const std::vector<std::vector<std::string>> &data);

OdometryData LineToOdometryData(const std::vector<std::string> &line, const int &time);

SensorData LineToSensorData(const std::vector<std::string> &line, const int &time);

std::vector<Landmark> InitializeLandmarkCollection(const std::vector<SensorData> &sen_collection);

autodiff::VectorXdual Initializex(const size_t &no_prm);

Eigen::MatrixXd InitializeCx(const size_t &no_prm);

void AddPoseToCollection(const autodiff::VectorXdual &x,
                         const int &time,
                         std::vector<Pose> pose_collection);

void PredictionStep(const OdometryData &odo, autodiff::VectorXdual &x, Eigen::MatrixXd &Cx);

Eigen::MatrixXd GetR(const long &no_prm);

autodiff::VectorXdual g(const autodiff::VectorXdual &x, const OdometryData &odo);

void CorrectionStep(const std::vector<SensorData> &sen_this_step,
                    std::vector<Landmark> &lm_collection,
                    autodiff::VectorXdual &x,
                    Eigen::MatrixXd &Cx);

std::vector<SensorData> GetCurrentSensorData(const std::vector<SensorData> &sen_collection,
                                             const int &time);

bool NewLandmark(const SensorData &sen, const std::vector<Landmark> &lm_collection);

void AddFirstEstimateOfLandmarkTox(const SensorData &sen,
                                   std::vector<Landmark> &lm_collection,
                                   autodiff::VectorXdual &x);

std::tuple<int, int> GetIdxPrmOfLandmark(const SensorData &sen, const std::vector<Landmark> &lm);

autodiff::VectorXdual h(const autodiff::VectorXdual &x, const int &idx_prm_x, const int &idx_prm_y);

Eigen::MatrixXd GetQ(const unsigned long &no_obs);

void ReportRobotPose(const autodiff::VectorXdual &x, const int &time);

autodiff::dual NormalizeAngle(autodiff::dual angle);

autodiff::VectorXdual ResetDualVector(autodiff::VectorXdual x);

#endif  // EKF_SLAM_MAIN_H
