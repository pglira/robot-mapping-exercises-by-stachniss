#include "main.h"

int main() {
  const std::string kPathToSensorData{"sensor_data.dat"};

  auto data{ReadDataFromFile(kPathToSensorData)};
  auto [odo_collection, sen_collection]{ExtractOdometryAndSensorData(data)};

  std::vector<Landmark> lm_collection{InitializeLandmarkCollection(sen_collection)};
  std::vector<Pose> pose_collection{};

  size_t no_prm{3 + lm_collection.size() * 2};  // x0, y0, theta0, x_lm1, y_lm1, x_lm2, y_lm2, ...
  auto x{Initializex(no_prm)};
  auto Cx{InitializeCx(no_prm)};

  AddPoseToCollection(x, 0, pose_collection);
  ReportRobotPose(x.head(3), 0);

  for (auto &odo : odo_collection) {
    PredictionStep(odo, x, Cx);
    CorrectionStep(GetCurrentSensorData(sen_collection, odo.time), lm_collection, x, Cx);
    AddPoseToCollection(x, odo.time, pose_collection);
    ReportRobotPose(x.head(3), odo.time);
  }

  return 0;
}

std::vector<std::vector<std::string>> ReadDataFromFile(const std::string &path_to_data) {
  std::vector<std::vector<std::string>> parsed_data;

  std::ifstream data(path_to_data);
  if (data.is_open()) {
    std::string line;
    while (getline(data, line)) {
      std::stringstream line_stream(line);
      std::string cell;  // single value
      std::vector<std::string> parsed_row;
      while (getline(line_stream, cell, ' ')) {
        parsed_row.push_back(cell);
      }
      parsed_data.push_back(parsed_row);
    }
  } else {
    std::cerr << "Error opening file!" << std::endl;
    exit(-1);
  }

  return parsed_data;
}

std::tuple<std::vector<OdometryData>, std::vector<SensorData>> ExtractOdometryAndSensorData(
    const std::vector<std::vector<std::string>> &data) {
  std::vector<OdometryData> odo{};
  std::vector<SensorData> sen{};
  int obs_time{0};
  for (const auto &line : data) {
    if (line[0] == "ODOMETRY") {
      obs_time = obs_time + 1;
      odo.push_back(LineToOdometryData(line, obs_time));
    } else {
      sen.push_back(LineToSensorData(line, obs_time));
    }
  }
  return {odo, sen};
}

OdometryData LineToOdometryData(const std::vector<std::string> &line, const int &time) {
  OdometryData odo{};
  odo.time = time;
  odo.rot1 = std::stod(line[1]);
  odo.trans = std::stod(line[2]);
  odo.rot2 = std::stod(line[3]);
  return odo;
}

SensorData LineToSensorData(const std::vector<std::string> &line, const int &time) {
  SensorData sen{};
  sen.time = time;
  sen.id = std::stoi(line[1]);
  sen.range = std::stod(line[2]);
  sen.bearing = std::stod(line[3]);
  return sen;
}

std::vector<Landmark> InitializeLandmarkCollection(const std::vector<SensorData> &sen_collection) {
  // Get max id
  int max_id{0};
  for (const auto &sen : sen_collection) {
    if (sen.id > max_id) {
      max_id = sen.id;
    }
  }

  // Create collection with landmarks with ids from 1 to max_id
  std::vector<Landmark> lm_collection{};
  for (int i = 1; i <= max_id; i++) {
    lm_collection.emplace_back();
    lm_collection.back().id = i;
    lm_collection.back().idx_prm_x = 2 * i + 1;
    lm_collection.back().idx_prm_y = 2 * i + 2;
  }

  return lm_collection;
}

VectorXdual Initializex(const size_t &no_prm) { return VectorXdual(no_prm); }

MatrixXd InitializeCx(const size_t &no_prm) {
  const double kInitialLandmarkVariance = 1000;
  MatrixXd Cx{MatrixXd::Zero(no_prm, no_prm)};
  for (int i = 3; i < no_prm; i++) {
    Cx(i, i) = kInitialLandmarkVariance;
  }
  return Cx;
}

void AddPoseToCollection(const VectorXdual &x, const int &time, std::vector<Pose> pose_collection) {
  Pose pose{};
  pose.time = time;
  pose.x0 = x[0].val;
  pose.y0 = x[1].val;
  pose.theta0 = x[2].val;
  pose_collection.push_back(pose);
}

void PredictionStep(const OdometryData &odo, VectorXdual &x, MatrixXd &Cx) {
  MatrixXd R{GetR(x.size())};

  x = g(x, odo);
  x = ResetDualVector(x);
  MatrixXd G = jacobian(g, wrt(x), at(x, odo));

  Cx = G * Cx * G.transpose() + R;
}

MatrixXd GetR(const long &no_prm) {
  const double kMotionNoise{0.1};
  MatrixXd R{MatrixXd::Zero(no_prm, no_prm)};
  R(0, 0) = kMotionNoise;
  R(1, 1) = kMotionNoise;
  R(2, 2) = kMotionNoise / 10;
  return R;
}

VectorXdual g(const VectorXdual &x, const OdometryData &odo) {
  VectorXdual fx(x.rows());

  fx[0] = x[0] + odo.trans * cos(x[2] + odo.rot1);
  fx[1] = x[1] + odo.trans * sin(x[2] + odo.rot1);
  fx[2] = NormalizeAngle(x[2] + odo.rot1 + odo.rot2);

  for (int i = 3; i < x.rows(); i++) {
    fx[i] = x[i];
  }

  return fx;
}

void CorrectionStep(const std::vector<SensorData> &sen_this_step,
                    std::vector<Landmark> &lm_collection,
                    VectorXdual &x,
                    MatrixXd &Cx) {
  auto no_prm{x.size()};
  auto no_obs{sen_this_step.size() * 2};

  MatrixXd H(no_obs, no_prm);
  VectorXdual dz(no_obs);

  for (int i = 0; i < sen_this_step.size(); i++) {
    if (NewLandmark(sen_this_step[i], lm_collection)) {
      AddFirstEstimateOfLandmarkTox(sen_this_step[i], lm_collection, x);
    }

    auto [idx_prm_x, idx_prm_y]{GetIdxPrmOfLandmark(sen_this_step[i], lm_collection)};

    VectorXdual z(2);
    z << sen_this_step[i].range, sen_this_step[i].bearing;

    x = ResetDualVector(x);
    VectorXdual z_exp{};
    H.middleRows(2 * i, 2) = jacobian(h, wrt(x), at(x, idx_prm_x, idx_prm_y), z_exp);

    dz.middleRows(2 * i, 2) = z - z_exp;
    dz[2 * i + 1] = NormalizeAngle(dz[2 * i + 1]);
  }

  auto Q{GetQ(no_obs)};
  auto K{Cx * H.transpose() * (H * Cx * H.transpose() + Q).inverse()};

  x = x + K * dz;
  x[2] = NormalizeAngle(x[2]);

  Cx = (MatrixXd::Identity(no_prm, no_prm) - K * H) * Cx;
}

std::vector<SensorData> GetCurrentSensorData(const std::vector<SensorData> &sen_collection,
                                             const int &time) {
  std::vector<SensorData> sen_current{};
  for (const auto &sen : sen_collection) {
    if (sen.time == time) {
      sen_current.push_back(sen);
    }
  }
  return sen_current;
}

bool NewLandmark(const SensorData &sen, const std::vector<Landmark> &lm_collection) {
  for (const auto &lm : lm_collection) {
    if (lm.id == sen.id) {
      return !lm.initialized;
    }
  }
}

void AddFirstEstimateOfLandmarkTox(const SensorData &sen,
                                   std::vector<Landmark> &lm_collection,
                                   VectorXdual &x) {
  double x_lm{x[0].val + sen.range * cos(sen.bearing + x[2].val)};
  double y_lm{x[1].val + sen.range * sin(sen.bearing + x[2].val)};
  x[lm_collection[sen.id - 1].idx_prm_x] = x_lm;
  x[lm_collection[sen.id - 1].idx_prm_y] = y_lm;
  lm_collection[sen.id - 1].initialized = true;
}

std::tuple<int, int> GetIdxPrmOfLandmark(const SensorData &sen,
                                         const std::vector<Landmark> &lm_collection) {
  for (const auto &lm : lm_collection) {
    if (lm.id == sen.id) {
      return {lm.idx_prm_x, lm.idx_prm_y};
    }
  }
}

VectorXdual h(const VectorXdual &x, const int &idx_prm_x, const int &idx_prm_y) {
  dual dx{x[idx_prm_x] - x[0]};
  dual dy{x[idx_prm_y] - x[1]};

  VectorXdual fh(2);
  fh[0] = sqrt(dx * dx + dy * dy);
  fh[1] = NormalizeAngle(atan2(dy, dx) - x[2]);

  return fh;
}

MatrixXd GetQ(const unsigned long &no_obs) {
  const double kSensorNoise{0.01};
  MatrixXd Q{MatrixXd::Identity(no_obs, no_obs) * kSensorNoise};
  return Q;
}

void ReportRobotPose(const VectorXdual &x, const int &time) {
  std::cout << "Robot pose at time " << time << " = " << x.transpose() << std::endl;
}

dual NormalizeAngle(dual angle) {
  while (angle > M_PI) {
    angle = angle - 2 * M_PI;
  }
  while (angle < -M_PI) {
    angle = angle + 2 * M_PI;
  }
  return angle;
}

VectorXdual ResetDualVector(VectorXdual x) {
  for (int i = 0; i < x.size(); i++) {
    x[i].grad = 0;
  }
  return x;
}
