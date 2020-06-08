#ifndef EKF_SLAM_MAIN_H
#define EKF_SLAM_MAIN_H

#include <math.h>
#include <Eigen/Dense>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>
#include <fstream>
#include <iostream>
#include <vector>

using namespace autodiff;
using namespace Eigen;

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
  Matrix3d C;
  int idx_prm_x;
  int idx_prm_y;
  bool initialized{false};
};

int main();

std::vector<std::vector<std::string>> ReadDataFromFile(
    const std::string &path_to_data);

std::tuple<std::vector<OdometryData>, std::vector<SensorData>>
ExtractOdometryAndSensorData(const std::vector<std::vector<std::string>> &data);

OdometryData LineToOdometryData(const std::vector<std::string> &line,
                                const int &time);

SensorData LineToSensorData(const std::vector<std::string> &line,
                            const int &time);

std::vector<Landmark> InitializeLandmarkCollection(
    const std::vector<SensorData> &sen_collection);

VectorXdual Initializex(const size_t &no_prm);

MatrixXd InitializeCx(const size_t &no_prm);

void AddPoseToCollection(const VectorXdual &x,
                         const int &time,
                         std::vector<Pose> pose_collection);

void PredictionStep(const OdometryData &odo, VectorXdual &x, MatrixXd &Cx);

MatrixXd GetR(const long &no_prm);

VectorXdual g(const VectorXdual &x, const OdometryData &odo);

void CorrectionStep(const std::vector<SensorData> &sen_this_step,
                    std::vector<Landmark> &lm_collection,
                    VectorXdual &x,
                    MatrixXd &Cx);

std::vector<SensorData> GetCurrentSensorData(
    const std::vector<SensorData> &sen_collection, const int &time);

bool NewLandmark(const SensorData &sen,
                 const std::vector<Landmark> &lm_collection);

void AddFirstEstimateOfLandmarkTox(const SensorData &sen,
                                   std::vector<Landmark> &lm_collection,
                                   VectorXdual &x);

std::tuple<int, int> GetIdxPrmOfLandmark(const SensorData &sen,
                                         const std::vector<Landmark> &lm);

VectorXdual h(const VectorXdual &x, const int &idx_prm_x, const int &idx_prm_y);

MatrixXd GetQ(const unsigned long &no_obs);

void ReportRobotPose(const VectorXdual &x, const int &time);

dual NormalizeAngle(dual angle);

VectorXdual ResetDualVector(VectorXdual x);

#endif  // EKF_SLAM_MAIN_H
