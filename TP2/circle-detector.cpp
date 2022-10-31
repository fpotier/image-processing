#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/types.hpp>
#include <string>
#include <vector>

const std::string project_dir(PROJECT_DIRECTORY);

struct accumulator_point
{
    accumulator_point(int _row, int _column, int _radius , int _score)
        : row(_row), column(_column), radius(_radius), score(_score)
    {}

    int row, column, radius;
    int score;
};

class accumulator
{
public:
    accumulator(int rows, int columns, int radius_min, int radius_max)
        : m_vec(rows * columns * radius_max - radius_max, 0),
        m_rows(rows), m_columns(columns), m_radiuses(radius_max - radius_min),
        m_radius_min(radius_min), m_radius_max(radius_max)
    {}

    double& at(int row, int column, int radius)
    {
        return m_vec[row * m_rows * m_radiuses + column * m_radiuses + radius - m_radius_min];
    }

    double at(int row, int column, int radius) const
    {
        return m_vec[row * m_rows * m_radiuses + column * m_radiuses + radius - m_radius_min];
    }

    bool is_local_max(int row, int column, int radius, int neighbor_distance = 3)
    {
        double local_value = at(row, column, radius);
        bool is_max = local_value > 0;
        for (int i = -neighbor_distance; i <= neighbor_distance && is_max; i++)
        {
            if (row + i < 0 || row + i >= m_rows)
                continue;
            for (int j = -neighbor_distance; j <= neighbor_distance && is_max; j++)
            {
                if (column + j < 0 || column + j >= m_columns)
                    continue;
                for (int k = -neighbor_distance; k <= neighbor_distance && is_max; k++)
                {
                    if ((i == 0 && j == 0 && k == 0) || radius + k < m_radius_min || radius + k >= m_radius_max)
                        continue;
                    is_max = local_value > at(row + i, column + j, radius + k);
                }
            }
        }

        return is_max;
    }

private:
    std::vector<double> m_vec;
    int m_rows, m_columns, m_radiuses;
    int m_radius_min, m_radius_max;
};

void naive_circle_detection(cv::Mat& original_image);
void optimized_circle_detection(cv::Mat& original_image);
void intermediate_circle_detection(cv::Mat& original_image, std::vector<accumulator_point>& local_maximums, int radius_min, int radius_max, int inverted_scale_factor);
void draw_circle(cv::Mat const& image, accumulator_point const& accu_point, cv::viz::Color color);

int main(int argc, char** argv)
{
    cv::Mat original_image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (original_image.empty())
    {
        std::cerr << "Could not load the image\n";
        exit(1);
    }
    auto start = std::chrono::system_clock::now();
    //naive_circle_detection(original_image);
    optimized_circle_detection(original_image);
    auto end = std::chrono::system_clock::now();
    std::cout << "Total compute time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    cv::imshow("TEST", original_image);
    while (cv::waitKey() != 'q')
        ;

    return 0;
}

void draw_circle(cv::Mat const& image, accumulator_point const& accu_point, cv::viz::Color color=cv::viz::Color::red())
{
    cv::circle(image,
        cv::Point(accu_point.column, accu_point.row),
        accu_point.radius,
        color);
}

void naive_circle_detection(cv::Mat& original_image)
{
    cv::Mat grayscale_image;
    if (original_image.channels() == 1)
    {
        grayscale_image = original_image;
        cv::cvtColor(original_image, original_image, cv::COLOR_GRAY2BGRA);
    }
    else
    {
        cv::cvtColor(original_image, grayscale_image, cv::COLOR_BGRA2GRAY);
    }

    // We apply a Gaussian filter to reduce noise
    cv::GaussianBlur(grayscale_image, grayscale_image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    cv::Mat gradient_x, gradient_y;
    cv::Sobel(grayscale_image, gradient_x, CV_16S, 1, 0);
    cv::Sobel(grayscale_image, gradient_y, CV_16S, 0, 1);

    cv::Mat abs_gradient_x, abs_gradient_y;
    cv::convertScaleAbs(gradient_x, abs_gradient_x);
    cv::convertScaleAbs(gradient_y, abs_gradient_y);

    cv::Mat gradient;
    cv::addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0, gradient);

    double threshold_coef = 0.5;
    double gradient_max;
    cv::minMaxLoc(gradient, nullptr, &gradient_max);
    double threshold_gradient = gradient_max * threshold_coef;

    int radius_max = std::min(gradient.cols / 2, gradient.rows / 2);
    int radius_min = radius_max * 0.1;

    std::vector<accumulator_point> local_maximums;
    intermediate_circle_detection(grayscale_image, local_maximums, radius_min, radius_max, 1);

    std::cout << "Number of circles: " << local_maximums.size() << '\n';
    std::sort(local_maximums.begin(), local_maximums.end(), [](accumulator_point const& lhs, accumulator_point const& rhs) {
        return lhs.score > rhs.score;
    });

    double max_score = local_maximums[0].score;
    int nb_circle = 0;
    for (accumulator_point const& circle : local_maximums)
    {
        if (circle.score < max_score * 0.6)
            break;
        draw_circle(original_image, circle);
        nb_circle++;
    }
    std::cout << "Number of circles drawn: " << nb_circle << '\n';
}

void optimized_circle_detection(cv::Mat& original_image)
{
    cv::Mat grayscale_image;
    if (original_image.channels() == 1)
    {
        grayscale_image = original_image;
        cv::cvtColor(original_image, original_image, cv::COLOR_GRAY2BGRA);
    }
    else
    {
        cv::cvtColor(original_image, grayscale_image, cv::COLOR_BGRA2GRAY);
    }

    // We apply a Gaussian filter to reduce noise
    cv::GaussianBlur(grayscale_image, grayscale_image, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    cv::Mat gradient_x, gradient_y;
    cv::Sobel(grayscale_image, gradient_x, CV_16S, 1, 0);
    cv::Sobel(grayscale_image, gradient_y, CV_16S, 0, 1);

    cv::Mat abs_gradient_x, abs_gradient_y;
    cv::convertScaleAbs(gradient_x, abs_gradient_x);
    cv::convertScaleAbs(gradient_y, abs_gradient_y);

    cv::Mat gradient;
    cv::addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0, gradient);

    double threshold_coef = 0.5;
    double gradient_max;
    cv::minMaxLoc(gradient, nullptr, &gradient_max);
    double threshold_gradient = gradient_max * threshold_coef;

    std::vector<accumulator_point> local_maximums;

    int radius_max = std::min(gradient.cols / 2, gradient.rows / 2);
    int radius_min = radius_max * 0.1;

    constexpr int nb_downscale = 3;
    intermediate_circle_detection(grayscale_image, local_maximums, radius_min, radius_max / (nb_downscale + 1), 1);
    cv::Mat downscaled_image;
    cv::resize(grayscale_image, downscaled_image, cv::Size(grayscale_image.cols / 2, grayscale_image.rows / 2));
    for (int i = 1; i <= nb_downscale; i++)
    {
        cv::resize(downscaled_image, downscaled_image, cv::Size(grayscale_image.cols / std::pow(2, i), grayscale_image.rows / std::pow(2, i)));
        intermediate_circle_detection(grayscale_image, local_maximums, i * radius_max / (nb_downscale + 1), (i + 1) * radius_max / (nb_downscale + 1), std::pow(2, i));
    }

    std::cout << "Number of circles: " << local_maximums.size() << '\n';
    std::sort(local_maximums.begin(), local_maximums.end(), [](accumulator_point const& lhs, accumulator_point const& rhs) {
        return lhs.score > rhs.score;
    });

    double max_score = local_maximums[0].score;
    int nb_circle = 0;
    for (accumulator_point const& circle : local_maximums)
    {
        if (circle.score < max_score * 0.7)
            break;
        draw_circle(original_image, circle);
        nb_circle++;
    }
    std::cout << "Number of circles drawn: " << nb_circle << '\n';
}

void intermediate_circle_detection(cv::Mat& grayscale_image, std::vector<accumulator_point>& local_maximums, int radius_min, int radius_max, int inverted_scale_factor)
{
    radius_min /= inverted_scale_factor;
    radius_max /= inverted_scale_factor;

    cv::Mat gradient_x, gradient_y;
    cv::Sobel(grayscale_image, gradient_x, CV_16S, 1, 0);
    cv::Sobel(grayscale_image, gradient_y, CV_16S, 0, 1);

    cv::Mat abs_gradient_x, abs_gradient_y;
    cv::convertScaleAbs(gradient_x, abs_gradient_x);
    cv::convertScaleAbs(gradient_y, abs_gradient_y);

    cv::Mat gradient;
    cv::addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0, gradient);

    double threshold_coef = 0.5;
    double gradient_max;
    cv::minMaxLoc(gradient, nullptr, &gradient_max);
    double threshold_gradient = gradient_max * threshold_coef;

    accumulator acc(gradient.rows, gradient.cols, radius_min, radius_max);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(1);
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < gradient.rows; i++)
    {
        for (int j = 0; j < gradient.cols; j++)
        {
            if (gradient.at<uint8_t>(i, j) < threshold_gradient)
                continue;

            int r_min = i - radius_max;
            if (r_min < 0)
                r_min = 0;
            int r_max = i + radius_max;
            if (r_max > gradient.rows)
                r_max = gradient.rows;
            for (int r = r_min; r  < r_max; r++)
            {
                int c_min = j - radius_max;
                if (c_min < 0)
                    c_min = 0;
                int c_max = j + radius_max;
                if (c_max > gradient.cols)
                    c_max = gradient.cols;
                for (int c = c_min; c < c_max; c++)
                {
                    int delta_x = i - r;
                    int delta_y = j - c;
                    double radius = std::sqrt(std::pow(delta_x, 2) + std::pow(delta_y, 2));
                    if (radius >= radius_min && radius < radius_max)
                        acc.at(r, c, radius) += gradient.at<uint8_t>(i, j) / radius;
                }
            }
        }
        std::cout << "\r[Downscaling: " << inverted_scale_factor << "] Processing: " << double(i * 100) / gradient.rows << "%" << std::flush;
    }
    auto end = std::chrono::system_clock::now();
    std::cout << "\33[2K\r[Downscaling: " << inverted_scale_factor << "] Processing: Done."
        << " Vote compute time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    for (int r = 0; r < gradient.rows; r++)
    {
        for (int c = 0; c < gradient.cols; c++)
        {
            for (int rad = radius_min; rad < radius_max; rad++)
            {
                if (acc.is_local_max(r, c, rad))
                    local_maximums.emplace_back(r * inverted_scale_factor, c * inverted_scale_factor, rad * inverted_scale_factor, acc.at(r, c, rad));
            }
        }
    }
}
