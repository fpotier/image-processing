#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

static constexpr int NB_FRAGMENTS = 328;

const std::string usage = "Usage: check solution_file [delta_x] [delta_y] [delta_alpha]\n";
int delta_x = 1;
int delta_y = 1;
double delta_alpha = 1;

const std::string project_dir(PROJECT_DIRECTORY);

struct fragment_position
{
    int x;
    int y;
    double alpha;
};

struct fragment_image
{
    fragment_image(std::string const& path)
    {
        img = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (!img.data)
            throw std::invalid_argument("Invalid fragment path: " + path);
        cv::Mat alpha_channel;
        cv::extractChannel(img, alpha_channel, 3);
        visible_pixels = std::accumulate(alpha_channel.begin<uint8_t>(), alpha_channel.end<uint8_t>(), 0,
            [](int accumulator, uint8_t val) {
                return accumulator + (val > 128);
            }
        );
    }
    cv::Mat img;
    int visible_pixels;
};

bool has_correct_position(fragment_position const& ref_fragment, fragment_position const& user_fragment)
{
    return std::abs(ref_fragment.x - user_fragment.x) <= delta_x
        && std::abs(ref_fragment.y - user_fragment.y) <= delta_y
        && std::abs(ref_fragment.alpha - user_fragment.alpha) <= delta_alpha;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << usage;
        return 1;
    }
    if (argc >= 3)
        delta_x = std::atoi(argv[2]);
    if (argc >= 4)
        delta_y = std::atoi(argv[3]);
    if (argc == 5)
        delta_alpha = std::atof(argv[4]);

    std::vector<fragment_image> fragments;
    fragments.reserve(NB_FRAGMENTS);
    for (size_t i = 0; i < NB_FRAGMENTS; i++)
    {
        std::ostringstream ss;
        ss << project_dir << "/frag_eroded/frag_eroded_" << i << ".png";
        fragments.emplace_back(ss.str());
    }

    int frag_index, frag_x, frag_y;
    double frag_alpha;

    std::array<std::optional<fragment_position>, NB_FRAGMENTS> reference_solution;
    {
        std::ifstream reference_file(project_dir + "/fragments.txt");
        while (reference_file >> frag_index >> frag_x >> frag_y >> frag_alpha)
            reference_solution[frag_index] = fragment_position { frag_x, frag_y, frag_alpha };
    }

    std::array<std::optional<fragment_position>, NB_FRAGMENTS> user_solution;
    {
        std::ifstream user_file(argv[1]);
        while (user_file >> frag_index >> frag_x >> frag_y >> frag_alpha)
            user_solution[frag_index] = fragment_position { frag_x, frag_y, frag_alpha };
    }

    int reference_score = 0;
    int user_score = 0;
    for (size_t i = 0; i < NB_FRAGMENTS; i++)
    {
        std::optional<fragment_position> const& ref_fragment = reference_solution[i];
        std::optional<fragment_position> const& user_fragment = user_solution[i];

        if (ref_fragment.has_value())
        {
            reference_score += fragments[i].visible_pixels;
            if (user_fragment.has_value() && (has_correct_position(ref_fragment.value(), user_fragment.value())))
                user_score += fragments[i].visible_pixels;
        }
        else if (user_fragment.has_value()) // User's solution contains a fragment that is not part of the fresco
            user_score -= fragments[i].visible_pixels;
    }

    double accuracy = static_cast<double>(user_score) / static_cast<double>(reference_score);
    std::cout << "User score: " << accuracy * 100 << "% (" << user_score << "/" << reference_score << ")\n";

    return 0;
}
