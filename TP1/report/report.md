# Report
Potier - Boireau

## Exercise 1:

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
In openCV, when you want to paste an image onto another, with the first being smaller than the second one, you have to select a region of interest (ROI) in the destination image.
```cpp
int roi_x = frag.x - frag.img.cols / 2; // The position of the fragment is given relative to its center
int roi_y = frag.y - frag.img.rows / 2;
int roi_w = frag.img.cols;
int roi_h = frag.img.rows;

cv::Mat roi = imageIn(cv::Rect(roi_x, roi_y, roi_w, roi_h));
frag_roi.copyTo(roi, mask);
```
## Image format
When copying an image onto another one, they must be in the same format. Unfortunately, the fragments are in BGRA format while the original image is in BGR (without alpha channel).
To make them match, we add an alpha channel to the original image:
```cpp
cv::Mat imageIn;
cv::cvtColor(image_original, imageIn, cv::COLOR_BGR2BGRA, 4);
```
With that done, we are able to paste most of the fragments onto the image. However, some of them (those on the edges) have a part out of the image so they are not pasted and produce an exception (that we just catch for now). 
At this point, here is the output image:
![fragments_with_borders](fragments_with_borders.png)

## Fragments on the edges
We don't want to copy the part of the fragment out of the image so we clip it from the ROI:
```cpp
int roi_w = std::min(frag.img.cols, imageIn.cols - roi_x); // removes the part at the right
int roi_h = std::min(frag.img.rows, imageIn.rows - roi_y);
if (roi_x < 0)
{
    roi_w += roi_x;  // x is negative so this removes the part that is at the left of the image
    frag_roi_x = -roi_x;
    roi_x = 0;
}
if (roi_y < 0)
{
    roi_h += roi_y;
    frag_roi_y = -roi_y;
    roi_y = 0;
}
```
## Not copying transparent borders of fragments
In order not to draw the transparent pixels of the fragments (and avoid these black borders), we build a mask, based on the alpha channel of the fragment, that will filter out the pixels with alpha less than 128 (arbitrary value that seems to fit).
```cpp
cv::Mat mask;
cv::extractChannel(frag_roi, mask, 3);
mask.forEach<uint8_t>([] (uint8_t& p, const int* pos) {
    p = p > 128;
});
frag_roi.copyTo(roi, mask);

```

The final output image is:
![reconstruction](reconstruction.jpg)
___

## Exercise 2:

In this exercise the goal was to compare a user-provided solution with the reference solution file. Each of these file contain a list of fragment id and position (coordinates of the center and angle use to rotate it), each entry can be represented by this `struct`:
```cpp
struct fragment_position
{
    int x;
    int y;
    double alpha;
};
```
We chose to represent a solution with:
```cpp
std::array<std::optional<fragment_position>, NB_FRAGMENTS> solution;
```
The index in the array corresponds to the fragment's identifier. This is possible because we know the total number of fragments.
Since a solution may not include a fragment we decided to use `std::optional`.
This way, an empty entry in the array means that the fragment is not used in the solution.

This is how we read the reference solution:
```cpp
std::array<std::optional<fragment_position>, NB_FRAGMENTS> reference_solution;
{
    std::ifstream reference_file(project_dir + "/fragments.txt");
    while (reference_file >> frag_index >> frag_x >> frag_y >> frag_alpha)
        reference_solution[frag_index] = fragment_position { frag_x, frag_y, frag_alpha };
}
```
And we do the same with the user-provided solution.

Now we can compare the two solutions by iterating over the arrays
```cpp
if (ref_fragment.has_value())
{
    if (user_fragment.has_value() && (has_correct_position(ref_fragment.value(), user_fragment.value())))
        // The fragment position is correct
}
else if (user_fragment.has_value())
    // User's solution contains a fragment that is not part of the fresco
```

With the verification function:
```cpp
bool has_correct_position(fragment_position const& ref_fragment, fragment_position const& user_fragment)
{
    return std::abs(ref_fragment.x - user_fragment.x) <= delta_x
        && std::abs(ref_fragment.y - user_fragment.y) <= delta_y
        && std::abs(ref_fragment.alpha - user_fragment.alpha) <= delta_alpha;
}
```
To compute the final score, we need to know each fragment's surface (we consider that the surface of a fragment is the number of opaque pixels).
To determine this, we use this data structure:
```cpp
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
```
A pixel is considered opaque if the value of its alpha channel is greater than a certain threshold (here 128).

The only issue we faced during this exercise was related to C++. We tried to use the same code as in the first exercise to determine opaque pixels:
```cpp
cv::Mat alpha_channel;
cv::extractChannel(img, alpha_channel, 3);
alpha_channel.forEach<uint8_t>([this] (uint8_t& p, const int* pos) {
    visible_pixels += p > 128;
});
```
We captured `this` to access `fragment_image::visible_pixels` in the constructor. The issue is that at this point the object is not properly initialized yet so the `this` pointer isn't valid. Using it at this point is undefined behavior. This caused random values for visible pixels between two executions with the same parameters.
To overcome this issue we used a temporary value instead of directly referencing the `visible_pixels` field.

