#include "TrtBase.hpp"

class RifeTrt : public TRTBase {
public:
	//using TRTBase::TRTBase;
	explicit RifeTrt(std::string modelName, int interpFactor, int width, int height, bool half,
		bool benchmark, std::string outputPath, int fps);
	~RifeTrt();

	void run(at::Tensor input) override;
private:
	torch::Tensor I0, I1, rgb_tensor;
	std::vector<at::Tensor> timestep_tensors;
	bool firstRun;
	bool useI0AsSource;
};
