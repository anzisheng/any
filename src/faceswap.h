# ifndef FACESWAP
# define FACESWAP
#include <array>
#include "engine.h"
#include "utile.h"
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

class SwapFace
{
public:
	SwapFace(std::string modelpath, const YoloV8Config &config);
	cv::Mat process(cv::Mat target_img, const std::vector<float> source_face_embedding, const std::vector<cv::Point2f> target_landmark_5);
    cv::Mat process(cv::cuda::GpuMat& gpuImg, const std::vector<float> source_face_embedding, const std::vector<cv::Point2f> target_landmark_5);
private:
	std::unique_ptr<Engine<float>> m_trtEngine_swap = nullptr;
	//Mat preprocess(cv::Mat target_img, const std::vector<cv::Point2f> face_landmark_5, const std::vector<float> source_face_embedding, cv::Mat& affine_matrix, cv::Mat& box_mask);
    void preprocess(cv::Mat target_img, const std::vector<cv::Point2f> face_landmark_5, const std::vector<float> source_face_embedding, cv::Mat& affine_matrix, cv::Mat& box_mask);
    std::vector<float> input_image;
	int input_height;
	int input_width;
    //std::vector<cv::Point2f> normed_template;

    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

	//std::vector<float> input_image;
	std::vector<float> input_embedding;
	
	const int len_feature = 512;
    float* model_matrix;
	std::vector<cv::Point2f> normed_template;
	const float FACE_MASK_BLUR = 0.3;
	const int FACE_MASK_PADDING[4] = {0, 0, 0, 0};
	const float INSWAPPER_128_MODEL_MEAN[3] = {0.0, 0.0, 0.0};
	const float INSWAPPER_128_MODEL_STD[3] = {1.0, 1.0, 1.0};






};
#endif