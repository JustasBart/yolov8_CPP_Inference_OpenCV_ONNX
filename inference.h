#ifndef INFERENCE_H
#define INFERENCE_H

// Qt includes
#include <QRandomGenerator>
#include <QObject>
#include <QDebug>
#include <QColor>
#include <QFile>

// Cpp native
#include <fstream>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    int class_id{0};
    QString className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const QString &onnxModelPath, const cv::Size2f &modelInputShape, const QString &classesTxtFile, const bool &runWithCuda = true);
    QVector<Detection> runInference(const cv::Mat &input);

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    QString modelPath;
    QString classesPath;
    bool cudaEnabled;

    QVector<QString> classes{};
    cv::Size2f modelShape{};

    float modelConfidenseThreshold  {0.3};
    float modelScoreThreshold       {0.4};
    float modelNMSThreshold         {0.4};

    cv::dnn::Net net;
};

#endif // INFERENCE_H
