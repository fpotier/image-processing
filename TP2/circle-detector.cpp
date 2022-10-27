#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/types.hpp>
#include <string>

const std::string project_dir(PROJECT_DIRECTORY);

int nb_row = 0;
int nb_col = 0;
int radius_min = 5;
int radius_max = 0;
int nb_radius = 0;

int compute_accumulator_index(int r, int c, int rad);
bool is_local_max(std::vector<double> const& accumulator, int r, int c, int rad);

struct accumulator_point
{
    accumulator_point(int r_, int c_, int radius_ , int score_)
        : r(r_), c(c_), radius(radius_), score(score_)
    {}

    int r, c, radius;
    int score;
};

void draw_circle(cv::Mat const& image, accumulator_point const& accu_point);

int main()
{
    cv::Mat original_img = cv::imread(project_dir + "/images/four.png", cv::IMREAD_UNCHANGED);

    cv::Mat grayscale_img;
    if (original_img.channels() == 1)
    {
        grayscale_img = original_img;
        cv::cvtColor(original_img, original_img, cv::COLOR_GRAY2BGRA);
    }
    else
    {
        cv::cvtColor(original_img, grayscale_img, cv::COLOR_BGRA2GRAY);
    }

    // We apply a Gaussian filter to reduce noise
    cv::GaussianBlur(grayscale_img, grayscale_img, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    cv::Mat gradient_x, gradient_y;
    cv::Sobel(grayscale_img, gradient_x, CV_16S, 1, 0);
    cv::Sobel(grayscale_img, gradient_y, CV_16S, 0, 1);

    cv::Mat abs_gradient_x, abs_gradient_y;
    cv::convertScaleAbs(gradient_x, abs_gradient_x);
    cv::convertScaleAbs(gradient_y, abs_gradient_y);

    cv::Mat gradient;
    cv::addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0, gradient);

    double threshold_coef = 0.5;
    double gradient_max;
    cv::minMaxLoc(gradient, nullptr, &gradient_max);
    std::cout << "Gradient maximum value: " << gradient_max << '\n';
    double threshold_gradient = gradient_max * threshold_coef;
    /*
    gradient.forEach<uint8_t>([threshold_gradient](uint8_t& pixel, const int position[2]) {
        if (pixel >= threshold_gradient)
        {
            pixel = 255;
            std::cout << "(" << position[0] << ", " << position[1] << ")\n";
        }
    });
    */
    nb_row = gradient.rows;
    nb_col = gradient.cols;
    radius_max = std::min(gradient.cols / 2, gradient.rows / 2);
    nb_radius = radius_max - radius_min;
    std::vector<double> accumulator(gradient.cols * gradient.rows * nb_radius, 0);
    std::cout << "vec size " << gradient.cols * gradient.rows * nb_radius << '\n';
    for (size_t i = 0; i < gradient.rows; i++)
    {
        for (size_t j = 0; j < gradient.cols; j++)
        {
            if (gradient.at<uint8_t>(i, j) < threshold_gradient)
                continue;

            for (size_t r = 0; r < gradient.rows; r++)
            {
                for (size_t c = 0; c < gradient.cols; c++)
                {
                    int delta_x = i - r;
                    int delta_y = j - c;
                    double radius = std::sqrt(std::pow(delta_x, 2) + std::pow(delta_y, 2));
                    if (radius >= radius_min && radius < radius_max)
                        accumulator[compute_accumulator_index(r, c, radius)] += gradient.at<uint8_t>(i, j) / radius;
                }
            }
        }
    }

    std::vector<accumulator_point> local_maximums;
    for (int r = 0; r < gradient.rows; r++)
    {
        for (int c = 0; c < gradient.cols; c++)
        {
            for (int rad = radius_min; rad < radius_max; rad++)
            {
                if (is_local_max(accumulator, r, c, rad))
                    local_maximums.emplace_back(r, c, rad, accumulator[compute_accumulator_index(r, c, rad)]);
            }
        }
    }

    std::cout << "Number of circles: " << local_maximums.size() << '\n';
    std::sort(local_maximums.begin(), local_maximums.end(), [](accumulator_point const& lhs, accumulator_point const& rhs) {
        return lhs.score > rhs.score;
    });
    std::cout << "First circle: center(" << local_maximums[0].r << ", " << local_maximums[0].c << ") radius=" << local_maximums[0].radius << '\n';
    for (int i = 0; i < 5; i++)
        draw_circle(original_img, local_maximums[i]);

    cv::imshow("TEST", original_img);
    while (cv::waitKey() != 'q')
        ;

    return 0;
}

int compute_accumulator_index(int r, int c, int rad)
{
    return r * nb_col * nb_radius + c * nb_radius + rad - radius_min;
}

bool is_local_max(std::vector<double> const& accumulator, int r, int c, int rad)
{
    int local_value = accumulator[compute_accumulator_index(r, c, rad)];
    bool is_max = local_value > 0;
    for (int i = -1; i <= 1 && is_max; i++)
    {
        if (r + i < 0 || r + i >= nb_row)
            continue;
        for (int j = -1; j <= 1 && is_max; j++)
        {
            if ( c + j < 0 || c + j >= nb_col)
                continue;
            for (int k = -1; k <= 1 && is_max; k++)
            {
                if ((i == 0 && j == 0 && k == 0) || rad + k < radius_min || rad + k >= radius_max)
                    continue;
                is_max = local_value > accumulator[compute_accumulator_index(r + i, c + j, rad + k)];
            }
        }
    }

    return is_max;
}

void draw_circle(cv::Mat const& image, accumulator_point const& accu_point)
{
    cv::circle(image,
        cv::Point(accu_point.c, accu_point.r),
        accu_point.radius,
        cv::viz::Color::red());
}
