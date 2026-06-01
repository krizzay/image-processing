#pragma once

static std::string readFile(const char* fileName); 

size_t getLocalWorkSize(size_t maxLocalSize, size_t globalSize);

bool compileKernel(std::string kernelName);

bool setup(std::vector<std::string> kernelNames, cl_context *context,
	   		cl_command_queue *commandQueue, cl_device_id *device,
			std::unordered_map<std::string, cl_program> *programs,
			std::unordered_map<std::string, cl_kernel> *kernels);

const char* getChannelOrderName(cl_channel_order order);

const char* getChannelTypeName(cl_channel_type type);

bool displayImageFormats(cl_context *context);

