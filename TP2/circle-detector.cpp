#include <opencv2/opencv.hpp>
#include <string>

const std::string project_dir(PROJECT_DIRECTORY);

int main()
{
    cv::Mat original_img = cv::imread(project_dir + "/images/four.png", cv::IMREAD_GRAYSCALE);

    // We apply a Gaussian filter to reduce noise
    cv::GaussianBlur(original_img, original_img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    cv::imshow("TEST", original_img);
    while (cv::waitKey() != 'q')
        ;

    return 0;
}
