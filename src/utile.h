#ifndef __UTIL__H
#define __UTIL__H

// Utility method for checking if a file exists on disk
inline bool doesFileExist(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Pose estimation key points
    std::vector<float> kps{};
};


// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
struct YoloV8Config {
    // The precision to be used for inference
    Precision precision = Precision::FP16;
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Max number of detected objects to return
    int topK = 100;
    // Segmentation config options
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Pose estimation options
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // Class thresholds (default are COCO classes)
    std::vector<std::string> classNames = {
        "person"};
    // std::vector<std::string> classNames = {
    //     "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    //     "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    //     "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    //     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    //     "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    //     "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    //     "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    //     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    //     "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    //     "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    //     "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    //     "teddy bear",     "hair drier", "toothbrush"};
};
#endif