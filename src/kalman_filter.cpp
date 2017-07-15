#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd x_polar = GetPolarizedPrediction();

  VectorXd y = z - x_polar;
  MatrixXd Ht = H_.transpose(); //already initialized as jacobian when Radar
  MatrixXd Pht = P_ * Ht;
  MatrixXd S = (H_ * Pht) + R_;
  MatrixXd K = Pht * S.inverse();

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - (K * H_)) * P_;

}

VectorXd KalmanFilter::GetPolarizedPrediction() {
    VectorXd measurement_vector(3);
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float norm = sqrt((px * px) + (py * py));

    /* Check for divide by 0s */
    if (norm > 0.00001f)
    {
      measurement_vector(0) = norm;
      measurement_vector(1) = atan2(py, px);
      measurement_vector(2) = ((px * vx) + (py * vy)) / norm;
    }
    else
    {
      measurement_vector = (H_ * x_); //revert to  jacobian instead if zero divison
    }

    return measurement_vector;
}
