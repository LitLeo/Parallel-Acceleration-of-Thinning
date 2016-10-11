#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
using namespace std;

typedef unsigned int uint;

// Allocates an array with random uint entries.
void RandomInit(uint *data, int n, uint lower_limit, uint upper_limit);

// warmup 函数，用于计时时 warmup GPU，实际是一个 vector 相加
void warmup();

// Error Code string definitions here
typedef struct
{
    char const *error_string;
    int  error_id;
} s_CudaErrorStr;

/**
 * Error codes
 */
typedef enum cudaError_enum
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    CUDA_SUCCESS                              = 0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    CUDA_ERROR_INVALID_VALUE                  = 1,

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    CUDA_ERROR_OUT_OF_MEMORY                  = 2,

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    CUDA_ERROR_NOT_INITIALIZED                = 3,

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    CUDA_ERROR_DEINITIALIZED                  = 4,

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode.
    */
    CUDA_ERROR_PROFILER_DISABLED           = 5,
    /**
     * This indicates profiling has not been initialized for this context.
     * Call cuProfilerInitialize() to resolve this.
    */
    CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,
    /**
     * This indicates profiler has already been started and probably
     * cuProfilerStart() is incorrectly called.
    */
    CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,
    /**
     * This indicates profiler has already been stopped and probably
     * cuProfilerStop() is incorrectly called.
    */
    CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    CUDA_ERROR_NO_DEVICE                      = 100,

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    CUDA_ERROR_INVALID_DEVICE                 = 101,


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    CUDA_ERROR_INVALID_IMAGE                  = 200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    CUDA_ERROR_INVALID_CONTEXT                = 201,

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,

    /**
     * This indicates that a map or register operation has failed.
     */
    CUDA_ERROR_MAP_FAILED                     = 205,

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    CUDA_ERROR_UNMAP_FAILED                   = 206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    CUDA_ERROR_ARRAY_IS_MAPPED                = 207,

    /**
     * This indicates that the resource is already mapped.
     */
    CUDA_ERROR_ALREADY_MAPPED                 = 208,

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,

    /**
     * This indicates that a resource has already been acquired.
     */
    CUDA_ERROR_ALREADY_ACQUIRED               = 210,

    /**
     * This indicates that a resource is not mapped.
     */
    CUDA_ERROR_NOT_MAPPED                     = 211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    CUDA_ERROR_ECC_UNCORRECTABLE              = 214,

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

    /**
     * This indicates that the device kernel source is invalid.
     */
    CUDA_ERROR_INVALID_SOURCE                 = 300,

    /**
     * This indicates that the file specified was not found.
     */
    CUDA_ERROR_FILE_NOT_FOUND                 = 301,

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

    /**
     * This indicates that initialization of a shared object failed.
     */
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

    /**
     * This indicates that an OS call failed.
     */
    CUDA_ERROR_OPERATING_SYSTEM               = 304,


    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    CUDA_ERROR_INVALID_HANDLE                 = 400,


    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    CUDA_ERROR_NOT_FOUND                      = 500,


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    CUDA_ERROR_NOT_READY                      = 600,


    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    CUDA_ERROR_LAUNCH_FAILED                  = 700,

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,

    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

    /**
     * This error indicates that a call to ::cuMemPeerRegister is trying to
     * register memory from a context which has not had peer access
     * enabled yet via ::cuCtxEnablePeerAccess(), or that
     * ::cuCtxDisablePeerAccess() is trying to disable peer access
     * which has not been enabled yet.
     */
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED    = 705,

    /**
     * This error indicates that a call to ::cuMemPeerRegister is trying to
     * register already-registered memory.
     */
    CUDA_ERROR_PEER_MEMORY_ALREADY_REGISTERED = 706,

    /**
     * This error indicates that a call to ::cuMemPeerUnregister is trying to
     * unregister memory that has not been registered.
     */
    CUDA_ERROR_PEER_MEMORY_NOT_REGISTERED     = 707,

    /**
     * This error indicates that ::cuCtxCreate was called with the flag
     * ::CU_CTX_PRIMARY on a device which already has initialized its
     * primary context.
     */
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    CUDA_ERROR_ASSERT                         = 710,

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cuCtxEnablePeerAccess().
     */
    CUDA_ERROR_TOO_MANY_PEERS                 = 711,

    /**
     * This error indicates that the memory range passed to ::cuMemHostRegister()
     * has already been registered.
     */
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

    /**
     * This error indicates that the pointer passed to ::cuMemHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    CUDA_ERROR_UNKNOWN                        = 999
} CUresult;


/**
 * Error codes
 */
static s_CudaErrorStr sCudaDrvErrorString[] =
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    { "CUDA_SUCCESS", 0 },

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    { "CUDA_ERROR_INVALID_VALUE", 1 },

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    { "CUDA_ERROR_OUT_OF_MEMORY", 2 },

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    { "CUDA_ERROR_NOT_INITIALIZED", 3 },

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    { "CUDA_ERROR_DEINITIALIZED", 4 },

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode.
    */
    { "CUDA_ERROR_PROFILER_DISABLED", 5 },
    /**
     * This indicates profiling has not been initialized for this context.
     * Call cuProfilerInitialize() to resolve this.
    */
    { "CUDA_ERROR_PROFILER_NOT_INITIALIZED", 6 },
    /**
     * This indicates profiler has already been started and probably
     * cuProfilerStart() is incorrectly called.
    */
    { "CUDA_ERROR_PROFILER_ALREADY_STARTED", 7 },
    /**
     * This indicates profiler has already been stopped and probably
     * cuProfilerStop() is incorrectly called.
    */
    { "CUDA_ERROR_PROFILER_ALREADY_STOPPED", 8 },
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    { "CUDA_ERROR_NO_DEVICE (no CUDA-capable devices were detected)", 100 },

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    { "CUDA_ERROR_INVALID_DEVICE (device specified is not a valid CUDA device)", 101 },


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    { "CUDA_ERROR_INVALID_IMAGE", 200 },

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    { "CUDA_ERROR_INVALID_CONTEXT", 201 },

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    { "CUDA_ERROR_CONTEXT_ALREADY_CURRENT", 202 },

    /**
     * This indicates that a map or register operation has failed.
     */
    { "CUDA_ERROR_MAP_FAILED", 205 },

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    { "CUDA_ERROR_UNMAP_FAILED", 206 },

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    { "CUDA_ERROR_ARRAY_IS_MAPPED", 207 },

    /**
     * This indicates that the resource is already mapped.
     */
    { "CUDA_ERROR_ALREADY_MAPPED", 208 },

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    { "CUDA_ERROR_NO_BINARY_FOR_GPU", 209 },

    /**
     * This indicates that a resource has already been acquired.
     */
    { "CUDA_ERROR_ALREADY_ACQUIRED", 210 },

    /**
     * This indicates that a resource is not mapped.
     */
    { "CUDA_ERROR_NOT_MAPPED", 211 },

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    { "CUDA_ERROR_NOT_MAPPED_AS_ARRAY", 212 },

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    { "CUDA_ERROR_NOT_MAPPED_AS_POINTER", 213 },

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    { "CUDA_ERROR_ECC_UNCORRECTABLE", 214 },

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    { "CUDA_ERROR_UNSUPPORTED_LIMIT", 215 },

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    { "CUDA_ERROR_CONTEXT_ALREADY_IN_USE", 216 },

    /**
     * This indicates that the device kernel source is invalid.
     */
    { "CUDA_ERROR_INVALID_SOURCE", 300 },

    /**
     * This indicates that the file specified was not found.
     */
    { "CUDA_ERROR_FILE_NOT_FOUND", 301 },

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    { "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", 302 },

    /**
     * This indicates that initialization of a shared object failed.
     */
    { "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED", 303 },

    /**
     * This indicates that an OS call failed.
     */
    { "CUDA_ERROR_OPERATING_SYSTEM", 304 },


    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    { "CUDA_ERROR_INVALID_HANDLE", 400 },


    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names }, and surface names.
     */
    { "CUDA_ERROR_NOT_FOUND", 500 },


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    { "CUDA_ERROR_NOT_READY", 600 },


    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used }, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    { "CUDA_ERROR_LAUNCH_FAILED", 700 },

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    { "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", 701 },

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    { "CUDA_ERROR_LAUNCH_TIMEOUT", 702 },

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    { "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", 703 },

    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    { "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED", 704 },

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is
     * trying to disable peer access which has not been enabled yet
     * via ::cuCtxEnablePeerAccess().
     */
    { "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED", 705 },

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    { "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE", 708 },

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy }, or is a primary context which
     * has not yet been initialized.
     */
    { "CUDA_ERROR_CONTEXT_IS_DESTROYED", 709 },

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    { "CUDA_ERROR_ASSERT", 710 },

    /**
     * This indicates that an unknown internal error has occurred.
     */
    { "CUDA_ERROR_UNKNOWN", 999 },
    { NULL, -1 }
};

// This is just a linear search through the array, since the error_id's are not
// always ocurring consecutively
const char *getCudaDrvErrorString(CUresult error_id);

// void __checkCudaErrors(CUresult err, const char *file, const int line);
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __check_cuda_errors (err, __FILE__, __LINE__)
// These are the inline versions for all of the SDK helper functions
void __check_cuda_errors(cudaError err, const char *file, const int line);


// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError (msg, __FILE__, __LINE__)
void __getLastCudaError(const char *errorMessage, const char *file, const int line);

//冒泡排序
void BubbleSort(uint* array,int start,int end);
// 快速排序
void quicksort(uint* arr, int st, int end);

#endif // COMMON_H_
