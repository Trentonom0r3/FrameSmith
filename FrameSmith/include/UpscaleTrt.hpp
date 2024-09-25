#include <TrtBase.hpp>

class UpscaleTrt : public TRTBase {
public:
	//using TRTBase::TRTBase;
	explicit UpscaleTrt(std::string upscaleMethod, int upscaleFactor, int width, int height, bool half,
				bool benchmark, std::string outputPath, int fps);
	~UpscaleTrt();

	void run(at::Tensor input) override;
};
