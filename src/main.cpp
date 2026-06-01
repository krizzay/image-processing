#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <CL/cl.h>
#include <memory>
#include <unordered_map>
#include <functional>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

cl_context context;
cl_command_queue commandQueue;
cl_device_id device;

std::unordered_map<std::string, cl_program> programs;
std::unordered_map<std::string, cl_kernel> kernels;

std::vector<cl_mem> memories;

size_t globalWorkSize = 256;
size_t localWorkSize = 32;

void cleanup();

#define CHECK_RES(res, txt) \
		if(res != CL_SUCCESS){ \
				std::cerr << txt << "\n Failed with error (" << res << ")\n"; \
				cleanup(); \
				return 1; \
		} \

static std::string readFile(const char* fileName){                                       
    std::fstream f;
    f.open( fileName, std::ios_base::in );
    if(f.is_open() != true){
        std::cerr << "file aint open :<\n";
        return "oops!";
    }

    std::string res;
    while( !f.eof() ) {
        char c;
        f.get( c );
        res += c;
    }
    
    f.close();

    return std::move(res);
}

// currently not in use
size_t getLocalWorkSize(size_t maxLocalSize, size_t globalSize){
    size_t result = -1;

    for(int i = 1; i <= sqrt(globalSize); i++){
        if (globalSize % i == 0){

            if(globalSize/i == i){
                if(i >= result && i <= maxLocalSize && i < (int)sqrt(globalSize)){
                    result = i;
                }
            }
            else{
                if(i >= result && i <= maxLocalSize && i < (int)sqrt(globalSize)){
                    result = i;
                }
                if(globalSize/i >= result && globalSize/i <= maxLocalSize){
                    result = globalSize/i;
                }
            }
        }
    }

    if(result == -1){
        std::cerr << "couldnt find a factor for " << globalSize << " something strage is ";
    }

    return result;
}               

// this function expects a <kernelName>.cl in the src/kernels dir
bool compileKernel(std::string kernelName){

    // create program and kernel
	std::string f = "../src/kernels/" + kernelName;
    std::string s = readFile(f.c_str());
    const char* programSource = s.c_str();
    size_t length = 0;
    cl_int programResult;
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, &length,
		   								&programResult);
    if (programResult != CL_SUCCESS){
        std::cerr << "Failed to make program!\n Failed with error (" << programResult << ")\n";
        return false;
    }

    cl_int programBuildResult = clBuildProgram( program, 1, &device, "-cl-std=CL3.0\0", nullptr, nullptr);
	std::cout << "building " << kernelName << std::endl;
	char log[1024];
	size_t logLength;
	cl_int programBuildInfoResult = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 1024, log, &logLength);

	if(logLength >= 1024){
		char newlog[logLength];
		cl_int programBuildInfoResult = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logLength, newlog, &logLength);

		std::cout << "log len - " << logLength << std::endl;
		std::cout <<  "log:\n" << newlog << std::endl;
	}else{
		std::cout << "log len - " << logLength << std::endl;
		std::cout <<  "log:\n" << log << std::endl;
	}
	
	if (programBuildInfoResult != CL_SUCCESS){
		std::cerr << "Failed to build program!\n Failed with error (" << programBuildInfoResult << ")\n";
		return false;
	}

    cl_int kernelResult;              // this string must mach entry function name
	cl_kernel kernel = clCreateKernel( program, "start", &kernelResult);
    if (programResult != CL_SUCCESS){
        std::cerr << "Failed to make kernel!\n Failed with error (" << programResult << ")\n";
        return false;
    }

	programs[kernelName] = program;
	kernels[kernelName] = kernel;

	return true;
}

bool setup(std::vector<std::string> kernelNames){
    cl_int platformResult = CL_SUCCESS;
    cl_uint numPlatforms = 0;
    cl_platform_id platforms[64];

    platformResult = clGetPlatformIDs( 64, platforms, &numPlatforms );

    if (platformResult != CL_SUCCESS) {
        std::cerr << "Couldnt get platform IDs!\n Failed with error(" << platformResult << ")\n";
        return false;
    }

    for(int i = 0; i < numPlatforms; i++){
		char version[128];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, NULL);
		std::cout << "Platform version: " << version << std::endl;

        cl_device_id devices[64];
        unsigned int deviceCount;
        cl_int deviceResult = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount);
		std::cout << deviceCount << " devices found\n";

        if( deviceResult == CL_SUCCESS){
            for (int j = 0; j < deviceCount; j++){
                unsigned long numComputeUnits;
				bool imageSupport;
                size_t computeUnitsLen;
                cl_int deviceInfoResult = clGetDeviceInfo( devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &numComputeUnits, &computeUnitsLen);
                cl_int deviceInfoResult2 = clGetDeviceInfo( devices[j], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, &computeUnitsLen);
                std::cout << numComputeUnits << " compute units on the device\n";
				std::cout << (imageSupport == CL_TRUE ? "yes" : "no") << " imageSupport on device\n";

                
                if(deviceInfoResult == CL_SUCCESS){
                    // currently picks first device, rank by mem size ig?
                    std::cout << "device found!!\n";
                    device = devices[j];
                    break;
                }
            }

            char version[128];
            clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
            printf("OpenCL version: %s\n", version);
            
        }
    }

    cl_int contextResult;
    context = clCreateContext( nullptr, 1, &device, nullptr, nullptr, &contextResult);
    if (contextResult != CL_SUCCESS){
        std::cerr << "Failed to make context!\n Failed with error (" << contextResult << ")\n";
        return false;
    }

    cl_int commandQueueResult;
    cl_queue_properties props[] = { 0 };
    commandQueue = clCreateCommandQueueWithProperties(context, device, props, &commandQueueResult);
    if (commandQueueResult != CL_SUCCESS){
        std::cerr << "Failed to make command queue!\n Failed with error (" << commandQueueResult << ")\n";
        return false;
    }

	for(std::string name : kernelNames){
		if (compileKernel(name) == false) {
			std::cout << "Failed to compile kernel" << std::endl;
			return false;
		}
	}

    return true;
}

void cleanup(){
	for(cl_mem mem : memories){
		clReleaseMemObject(mem);
	}

	for (auto iter = kernels.begin(); iter != kernels.end(); ++iter){
		clReleaseKernel(iter->second);
	}
	for (auto iter = programs.begin(); iter != programs.end(); ++iter){
		clReleaseProgram(iter->second);
	}

    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
    clReleaseDevice(device); // added in 1.2
}

const char* getChannelOrderName(cl_channel_order order) {
    switch(order) {
        case CL_R: return "CL_R";
        case CL_A: return "CL_A";
        case CL_RG: return "CL_RG";
        case CL_RA: return "CL_RA";
        case CL_RGB: return "CL_RGB";
        case CL_RGBA: return "CL_RGBA";
        case CL_BGRA: return "CL_BGRA";
        case CL_ARGB: return "CL_ARGB";
        case CL_INTENSITY: return "CL_INTENSITY";
        case CL_LUMINANCE: return "CL_LUMINANCE";
        default: return "UNKNOWN_CHANNEL_ORDER";
    }
}

const char* getChannelTypeName(cl_channel_type type) {
    switch(type) {
        case CL_SNORM_INT8: return "CL_SNORM_INT8";
        case CL_SNORM_INT16: return "CL_SNORM_INT16";
        case CL_UNORM_INT8: return "CL_UNORM_INT8";
        case CL_UNORM_INT16: return "CL_UNORM_INT16";
        case CL_SIGNED_INT8: return "CL_SIGNED_INT8";
        case CL_SIGNED_INT16: return "CL_SIGNED_INT16";
        case CL_SIGNED_INT32: return "CL_SIGNED_INT32";
        case CL_UNSIGNED_INT8: return "CL_UNSIGNED_INT8";
        case CL_UNSIGNED_INT16: return "CL_UNSIGNED_INT16";
        case CL_UNSIGNED_INT32: return "CL_UNSIGNED_INT32";
        case CL_HALF_FLOAT: return "CL_HALF_FLOAT";
        case CL_FLOAT: return "CL_FLOAT";
        default: return "UNKNOWN_CHANNEL_TYPE";
    }
}

// not needed but still interesting
/*template <typename Func, typename ... Args>
bool func_api(Func fn, Args&&... args){

	fn(std::forward<Args>(args)...);
}
*/

bool testFun(int x, bool y){

	std::cout << "test - " << x << " : " << y << std::endl;
	return y;
}

enum StageType { KERNEL_STAGE, FUNCTION_STAGE };

struct Stage {
public:
	StageType m_type;

	virtual ~Stage() {}
protected:
	// protected constructor makes it kinda abstract class
	Stage(StageType type) : m_type {type} {}
};

struct KernelStage : public Stage {
public:
	std::string m_kernelName;

	std::vector<size_t> m_paramSizes;
	std::vector<void *> m_ptrs;

	KernelStage(StageType type, std::string name, std::vector<size_t> params, std::vector<void*> ptrs)
		: Stage(type),
		  m_kernelName {name},
		  m_paramSizes {params},
	  	  m_ptrs {ptrs}
	{ }	

};


struct FunctionStage : public Stage {
public:
	std::function<bool()> m_fn;
	
	FunctionStage(StageType type, std::function<bool()> fn) : Stage(type), m_fn {fn} {}

};



int main(int argc, char* argv[]){

	std::vector<std::shared_ptr<Stage>> stages;
    // arg parsing
	
	// init all params at start and fill them out in arg parsing
	int a = 5;
	bool b = true;

	// if sobel
	if (true) {
		int BlurKernelSize = 5;
		std::vector<size_t> sizes = { sizeof(BlurKernelSize) };
		std::vector<void*> ptrs = { &BlurKernelSize };

		std::shared_ptr<Stage> ptr ( new KernelStage(KERNEL_STAGE, std::string("sobel.cl"), sizes, ptrs));  
		stages.push_back(std::move(ptr));
	}

	// if function testFun
	if (true) {
		std::function<bool()> fn = [a, b](){ return testFun(a, b); };
		std::shared_ptr<Stage> ptr ( new FunctionStage(FUNCTION_STAGE, fn));
		stages.push_back(std::move(ptr));
	}

		
	// setup
	
	// TODO: either properly init kernelNames here or make setup accept stages
	std::vector<std::string> kernelNames = { std::string("sobel.cl") };
	if (setup(kernelNames) != true){
		std::cerr << "Failed to set up opencl\n";
		cleanup();
		return 1;
	}

	// display all image formats
	bool displayImageFormats = false;
	if (displayImageFormats) {
		unsigned int numFormats = 0;
		clGetSupportedImageFormats( context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 0, nullptr, &numFormats);

		cl_image_format formats[numFormats];
		CHECK_RES( clGetSupportedImageFormats( context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, numFormats, formats, &numFormats),
					"Failed to get supported image formats!");

		for(cl_image_format f : formats){

			std::cout << "channel order : " << getChannelOrderName(f.image_channel_order) <<
						",   channel type : " << getChannelTypeName(f.image_channel_data_type) <<
						std::endl;
		}
	}

	//TODO: handle images of arbitrary size correctly
	// set up images
	int x,y,n;
	std::unique_ptr<unsigned char> data (stbi_load("../images/cover.png", &x, &y, &n, 0));
	std::unique_ptr<unsigned char> outData(new unsigned char[x * y * n]);

	int dims[] = {x, y, n};
	std::cout << "x - " << x << ", y - " << y << ", n - " << n << std::endl;
	
	cl_image_format imageFormat;
	imageFormat.image_channel_order = CL_RGBA;
	imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

	cl_image_desc imageDescriptor;
	imageDescriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
	imageDescriptor.image_width = x;
	imageDescriptor.image_height = y;
	imageDescriptor.image_depth = 0;
	imageDescriptor.image_row_pitch = 0; //0 lets opencl calculate it
	imageDescriptor.image_slice_pitch = 0;
	imageDescriptor.num_mip_levels = 0; // not using mip mapping
	imageDescriptor.num_samples = 0; // must be 0 according to docs, whats the point of it then?
	imageDescriptor.mem_object = nullptr;
									 
	cl_int imageRes = CL_TRUE;
	cl_mem inImgBuf = clCreateImage( context,
				   					 CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
									 &imageFormat,
									 &imageDescriptor,
									 data.get(),
									 &imageRes);
	CHECK_RES(imageRes, "Failed to create in image!");
	memories.push_back(inImgBuf);

	cl_mem outImgBuf = clCreateImage( context,
				   					 CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
									 &imageFormat,
									 &imageDescriptor,
									 outData.get(),
									 &imageRes);
	CHECK_RES(imageRes, "Failed to create out image!");
	memories.push_back(outImgBuf);
	
	// work group stuff
	size_t maxLocalWorkSize;
	CHECK_RES( clGetKernelWorkGroupInfo(kernels[kernelNames[0]], device, CL_KERNEL_WORK_GROUP_SIZE,
						   				sizeof(size_t), &maxLocalWorkSize, nullptr),
				"Failed to query work group info from device!");
	

	size_t maxLocalWorkSizeDims[3];
	clGetDeviceInfo(device,
					CL_DEVICE_MAX_WORK_ITEM_SIZES,
					sizeof(maxLocalWorkSizeDims),
					maxLocalWorkSizeDims,
					nullptr);

	//TODO: figure something better out for local work group sizes
	size_t workDims[] = {x, y};
	size_t groupSize[] = {8, 8};

	std::cout << "max work group size - " << maxLocalWorkSize  
			  << "\n max work local work sizee dims - " << dims[0] << ","
			  << dims[1] << "," << dims[2] << std::endl;
	std::cout << "group size :: x - " << groupSize[0] << ", y - " << groupSize[1] << std::endl;

	for (auto& stage : stages){

		// execute each kernel
		if(stage->m_type == KERNEL_STAGE){
			std::shared_ptr<KernelStage> krnlStagePtr = std::static_pointer_cast<KernelStage>(stage);

			std::string krnlName = krnlStagePtr->m_kernelName;

			// assume the first 2 args of each kernel is the in and out images
			CHECK_RES( clSetKernelArg(kernels[krnlName], 0, sizeof(cl_mem), &inImgBuf),
						"Failed to set kernel arg (inimg)!");
			CHECK_RES( clSetKernelArg(kernels[krnlName], 1, sizeof(cl_mem), &outImgBuf),
						"Failed to set kernel arg (outimg)!");

			for(int i = 0; i < krnlStagePtr->m_paramSizes.size(); i++){

				cl_int res = clSetKernelArg(kernels[krnlName], (i+2), krnlStagePtr->m_paramSizes[i], krnlStagePtr->m_ptrs[i]);
				if (res != CL_SUCCESS){
					std::cerr << "Failed to set param " << (i+2) << " for kernel " << krnlName 
								<< std::endl << "Failed with error " << res << std::endl;
					cleanup();
					return 1;
				}
			}

			CHECK_RES( clEnqueueNDRangeKernel(commandQueue, kernels[krnlName], 2, 0, workDims,
											  groupSize, 0, nullptr, nullptr),
						"Failed to enqueeueeue kernel!");
		}
		else if(stage->m_type == FUNCTION_STAGE){
			
			std::shared_ptr<FunctionStage> fn_s = std::static_pointer_cast<FunctionStage>(stage);
			fn_s->m_fn();
		}
	}

	// output reading
	size_t origin[] = {0, 0, 0};
	size_t readRegion[] = {x, y, 1};

	CHECK_RES( clEnqueueReadImage( commandQueue, outImgBuf, CL_TRUE, origin,
						   			readRegion, x * sizeof(unsigned char) * 4, 0, outData.get(),
								   	0, nullptr, nullptr),
				"Failed to enqueue read iamge! out");

	// reads in input image for debug
	CHECK_RES( clEnqueueReadImage( commandQueue, inImgBuf, CL_TRUE, origin,
						   			readRegion, x * sizeof(unsigned char) * 4, 0, data.get(),
								   	0, nullptr, nullptr),
				"Failed to enqueue read iamge! in");

	clFinish(commandQueue);

	const char *outName = "out.png";
	stbi_write_png(outName, x, y, n, outData.get(), n * x); 

	cleanup();
}
