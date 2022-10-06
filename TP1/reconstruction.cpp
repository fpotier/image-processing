#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

const std::string project_dir(PROJECT_DIRECTORY);

struct fragment
{
    fragment(int _id, int _x, int _y, double _angle, cv::Mat _img)
        : id(_id), x(_x), y(_y), angle(_angle)
    {
        cv::Point2f center((_img.cols - 1) / 2.0, (_img.rows - 1) / 2.0);
        cv::Mat rotation = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(_img, img, rotation, _img.size());
    }
    int id;
    int x;
    int y;
    double angle;
    cv::Mat img;
};

int main(int argc, char** argv)
{
    cv::Mat image_original = cv::imread("../Michelangelo_ThecreationofAdam_1707x775.jpg", cv::IMREAD_UNCHANGED);
    if (!image_original.data) // Check for invalid input
    {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }
    cv::Mat imageIn;
    cv::cvtColor(image_original, imageIn, cv::COLOR_BGR2BGRA, 4);
    cv::Mat veil(imageIn.rows, imageIn.cols, CV_8UC4, cv::Scalar(200, 200, 200, 255));
    cv::addWeighted(imageIn, 1.0, veil, 0.5, 0, imageIn);

    std::vector<fragment> fragments;
    std::ifstream fragment_list(project_dir + "/fragments.txt");
    int i, frag_x, frag_y;
    double frag_angle;
    while (fragment_list >> i && fragment_list >> frag_x && fragment_list >> frag_y && fragment_list >> frag_angle)
    {
        std::ostringstream ss;
        ss << project_dir << "/frag_eroded/frag_eroded_" << i << ".png";
        fragment frag(i, frag_x, frag_y, frag_angle, cv::imread(ss.str(), cv::IMREAD_UNCHANGED));
        fragments.push_back(frag);
    }

    for (fragment const& frag : fragments)
    {
        int frag_roi_x = 0, frag_roi_y = 0;
        int roi_x = frag.x - frag.img.cols / 2;
        int roi_y = frag.y - frag.img.rows / 2;
        int roi_w = std::min(frag.img.cols, imageIn.cols - roi_x);
        int roi_h = std::min(frag.img.rows, imageIn.rows - roi_y);
        if (roi_x < 0)
        {
            roi_w += roi_x;
            frag_roi_x = -roi_x;
            roi_x = 0;
        }
        if (roi_y < 0)
        {
            roi_h += roi_y;
            frag_roi_y = -roi_y;
            roi_y = 0;
        }
        cv::Mat roi = imageIn(cv::Rect(roi_x, roi_y, roi_w, roi_h));
        cv::Mat frag_roi = frag.img(cv::Rect(frag_roi_x, frag_roi_y, roi_w, roi_h));
        cv::Mat mask;
        cv::extractChannel(frag_roi, mask, 3);
        mask.forEach<uint8_t>([] (uint8_t& p, const int* pos) {
            p = p > 128;
        });
        frag_roi.copyTo(roi, mask);
    }

    cv::imshow("Display window", imageIn);
    cv::waitKey(0);

    return 0;
}
