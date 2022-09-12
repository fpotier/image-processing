#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv)
{
    cv::Mat imageIn = cv::imread("../Michelangelo_ThecreationofAdam_1707x775.jpg", cv::IMREAD_UNCHANGED);
    if (!imageIn.data) // Check for invalid input
    {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }
    cv::Mat background(imageIn.rows, imageIn.cols, CV_8UC3, cv::Scalar(200, 200, 200));
    std::cout << "Original image: cols=" << imageIn.cols << " rows=" << imageIn.rows << '\n';
    std::cout << "Backgroud image: cols=" << background.cols << " rows=" << background.rows << '\n';
    cv::addWeighted(imageIn, 1.0, background, 0.5, 0, imageIn);
    cv::imshow("Display window", imageIn);
    cv::waitKey(0);

    return 0;
}
