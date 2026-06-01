
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
bool compileKernel(std::string kernelName, cl_context *context, cl_device_id *device,
					std::unordered_map<std::string, cl_program> *programs,
					std::unordered_map<std::string, cl_kernel> *kernels) {

    // create program and kernel
	std::string f = "../src/kernels/" + kernelName;
    std::string s = readFile(f.c_str());
    const char* programSource = s.c_str();
    size_t length = 0;
    cl_int programResult;
    cl_program program = clCreateProgramWithSource(*context, 1, &programSource, &length,
		   								&programResult);
    if (programResult != CL_SUCCESS){
        std::cerr << "Failed to make program!\n Failed with error (" << programResult << ")\n";
        return false;
    }

    cl_int programBuildResult = clBuildProgram( program, 1, device, "-cl-std=CL3.0\0", nullptr, nullptr);
	std::cout << "building " << kernelName << std::endl;
	char log[1024];
	size_t logLength;
	cl_int programBuildInfoResult = clGetProgramBuildInfo(program, *device, CL_PROGRAM_BUILD_LOG, 1024, log, &logLength);

	if(logLength >= 1024){
		char newlog[logLength];
		cl_int programBuildInfoResult = clGetProgramBuildInfo(program, *device, CL_PROGRAM_BUILD_LOG, logLength, newlog, &logLength);

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

	(*programs)[kernelName] = program;
	(*kernels)[kernelName] = kernel;

	return true;
}

bool setup(std::vector<std::string> kernelNames, cl_context *context,
	   		cl_command_queue *commandQueue, cl_device_id *device,
			std::unordered_map<std::string, cl_program> *programs,
			std::unordered_map<std::string, cl_kernel> *kernels) {
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
                    *device = devices[j];
                    break;
                }
            }

            char version[128];
            clGetDeviceInfo(*device, CL_DEVICE_VERSION, sizeof(version), version, NULL);
            printf("OpenCL version: %s\n", version);
            
        }
    }

    cl_int contextResult;
    *context = clCreateContext( nullptr, 1, device, nullptr, nullptr, &contextResult);
    if (contextResult != CL_SUCCESS){
        std::cerr << "Failed to make context!\n Failed with error (" << contextResult << ")\n";
        return false;
    }

    cl_int commandQueueResult;
    cl_queue_properties props[] = { 0 };
    *commandQueue = clCreateCommandQueueWithProperties(*context, *device, props, &commandQueueResult);
    if (commandQueueResult != CL_SUCCESS){
        std::cerr << "Failed to make command queue!\n Failed with error (" << commandQueueResult << ")\n";
        return false;
    }

	for(std::string name : kernelNames){
		if (compileKernel(name, context, device, programs, kernels) == false) {
			std::cout << "Failed to compile kernel" << std::endl;
			return false;
		}
	}

    return true;
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

bool displayImageFormats(cl_context *context){

		unsigned int numFormats = 0;
		clGetSupportedImageFormats( *context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 0, nullptr, &numFormats);

		cl_image_format formats[numFormats];
		cl_int res = clGetSupportedImageFormats( *context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, numFormats, formats, &numFormats);

		if(res != CL_SUCCESS){
			std::cerr << "Failed to get supported image formats!" << std::endl 
						<< "Failed with error( "<< res << ")" << std::endl;
			return false;
		}

		for(cl_image_format f : formats){
			std::cout << "channel order : " << getChannelOrderName(f.image_channel_order) <<
						",   channel type : " << getChannelTypeName(f.image_channel_data_type) <<
						std::endl;
		}
}
