# Report
Potier - Boireau
## Loading original image and display it in background
To form the background, we load the original image as is with:
```cpp
cv::Mat image_original = cv::imread("../Michelangelo_ThecreationofAdam_1707x775.jpg", cv::IMREAD_UNCHANGED);
cv::Mat imageIn;
cv::cvtColor(image_original, imageIn, cv::COLOR_BGR2BGRA, 4);
```
Then we create a "veil" with a light color and blend it with the image:
```cpp
cv::Mat veil(imageIn.rows, imageIn.cols, CV_8UC4, cv::Scalar(200, 200, 200, 255));
cv::addWeighted(imageIn, 1.0, veil, 0.5, 0, imageIn); // alpha_orig = 1.0 and alpha_veil = 0.5
```
![veiled_orig_image](veiled_orig_image.jpg)
## Loading fragments
To load the fragments, we read the fragment list to fetch their coordinates, associated with the corresponding image.
```cpp
std::vector<fragment> fragments;
std::ifstream fragment_list("../fragments.txt");
int i, frag_x, frag_y;
double frag_angle;
while (fragment_list >> i && fragment_list >> frag_x && fragment_list >> frag_y && fragment_list >> frag_angle)
{
    std::ostringstream ss;
    ss << "../frag_eroded/frag_eroded_" << i << ".png";
    fragment frag(i, frag_x, frag_y, frag_angle, cv::imread(ss.str(), cv::IMREAD_UNCHANGED));
    fragments.push_back(frag);
}

```

The fragment struct holds data of a fragment and its constructor is responsible for applying the rotation specified in the fragment list to the fragment image.
```cpp
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
```
## Adding fragments to image

## Channels of original image and fragments' do not match
- adding alpha channel to original image

![fragments_with_borders](fragments_with_borders.png)

## Not copying transparent borders of fragments
- building a mask to filter the pixel with low alpha component and passing it to copyTo()

![reconstruction](reconstruction.jpg)