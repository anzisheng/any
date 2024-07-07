# ifndef DETECT_FACE68LANDMARKS
# define DETECT_FACE68LANDMARKS
#include "engine.h"
#include "utile.h"
class Face68Landmarks
{
public:
    //Face68Landmarks(std::string modelpath);
public:    
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    Face68Landmarks(const std::string &onnxModelPath, const YoloV8Config &config);

private:
    std::unique_ptr<Engine<float>> m_trtEngine = nullptr;

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;


};



#endif