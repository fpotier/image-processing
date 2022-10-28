#include <algorithm>
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

    bool is_local_max(int row, int column, int radius)
    {
        double local_value = at(row, column, radius);
        bool is_max = local_value > 0;
        for (int i = -1; i <= 1 && is_max; i++)
        {
            if (row + i < 0 || row + i >= m_rows)
                continue;
            for (int j = -1; j <= 1 && is_max; j++)
            {
                if (column + j < 0 || column + j >= m_columns)
                    continue;
                for (int k = -1; k <= 1 && is_max; k++)
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
void draw_circle(cv::Mat const& image, accumulator_point const& accu_point);

int main()
{
    cv::Mat original_image = cv::imread(project_dir + "/images/coins2.jpg", cv::IMREAD_UNCHANGED);
    if (original_image.empty())
    {
        std::cerr << "Could not load the image\n";
        exit(1);
    }
    naive_circle_detection(original_image);
    cv::imshow("TEST", original_image);
    while (cv::waitKey() != 'q')
        ;

    return 0;
}

void draw_circle(cv::Mat const& image, accumulator_point const& accu_point)
{
    cv::circle(image,
        cv::Point(accu_point.column, accu_point.row),
        accu_point.radius,
        cv::viz::Color::red());
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
    std::cout << "Gradient maximum value: " << gradient_max << '\n';
    double threshold_gradient = gradient_max * threshold_coef;

    int radius_max = std::min(gradient.cols / 2, gradient.rows / 2);
    int radius_min = radius_max * 0.1;

    accumulator acc(gradient.rows, gradient.cols, radius_min, radius_max);
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
                        acc.at(r, c, radius) += gradient.at<uint8_t>(i, j) / radius;
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
                if (acc.is_local_max(r, c, rad))
                    local_maximums.emplace_back(r, c, rad, acc.at(r, c, rad));
            }
        }
    }

    std::cout << "Number of circles: " << local_maximums.size() << '\n';
    std::sort(local_maximums.begin(), local_maximums.end(), [](accumulator_point const& lhs, accumulator_point const& rhs) {
        return lhs.score > rhs.score;
    });
    std::cout << "First circle: center(" << local_maximums[0].row << ", " << local_maximums[0].column << ") radius=" << local_maximums[0].radius << '\n';
    for (int i = 0; i < 4; i++)
        draw_circle(original_image, local_maximums[i]);
}
