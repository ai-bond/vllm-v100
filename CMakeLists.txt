cmake_minimum_required(VERSION 3.26)

project(vllm_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "" FORCE)
set(CAFFE2_USE_NVRTC OFF CACHE BOOL "" FORCE)
set(USE_CUFILE OFF CACHE BOOL "" FORCE)
set(USE_KINETO OFF CACHE BOOL "" FORCE)

set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")

option(VLLM_VOLTA_FP8_EMULATION "Enable FP8 emulation on Volta via FP16 fallback" ON)
option(VLLM_VOLTA_FP4_EMULATION "Enable FP4 emulation on Volta via INT8/FP16 fallback" ON)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")
message(STATUS "Volta FP8 emulation: ${VLLM_VOLTA_FP8_EMULATION}")
message(STATUS "Volta FP4 emulation: ${VLLM_VOLTA_FP4_EMULATION}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

set(ignoreMe "${VLLM_PYTHON_PATH}")

install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13")

set(TORCH_SUPPORTED_VERSION_CUDA "2.10.0")

if (VLLM_PYTHON_EXECUTABLE)
    find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
    message(FATAL_ERROR
        "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
        " before running cmake configure.")
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

find_package(Torch REQUIRED)

if (NOT CUDA_FOUND)
    message(FATAL_ERROR "Can't find CUDA installation.")
endif()

set(VLLM_GPU_LANG "CUDA")

if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
        "expected for CUDA build, saw ${Torch_VERSION} instead.")
endif()

if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
    CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
    CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0;12.1")
else()
    set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

clear_cuda_arches(CUDA_ARCH_FLAGS)
extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")

cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")

get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
    list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
    list(APPEND VLLM_GPU_FLAGS "-Wno-deprecated-gpu-targets")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include")
    if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
    endif()
endif()

if(VLLM_VOLTA_FP8_EMULATION)
    list(APPEND VLLM_GPU_FLAGS "-DVLLM_VOLTA_FP8_EMULATION=1")
endif()

if(VLLM_VOLTA_FP4_EMULATION)
    list(APPEND VLLM_GPU_FLAGS "-DVLLM_VOLTA_FP4_EMULATION=1")
endif()

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR})
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

set(VLLM_CUMEM_EXT_SRC "csrc/cumem_allocator.cpp")
set_gencode_flags_for_srcs(
    SRCS "${VLLM_CUMEM_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

message(STATUS "Enabling cumem allocator extension.")
list(APPEND CUMEM_LIBS CUDA::cuda_driver)

define_extension_target(
    cumem_allocator
    DESTINATION vllm
    LANGUAGE CXX
    SOURCES ${VLLM_CUMEM_EXT_SRC}
    LIBRARIES ${CUMEM_LIBS}
    USE_SABI 3.8
    WITH_SOABI)

set(VLLM_EXT_SRC
    "csrc/mamba/mamba_ssm/selective_scan_fwd.cu"
    "csrc/cache_kernels.cu"
    "csrc/cache_kernels_fused.cu"
    "csrc/attention/paged_attention_v1.cu"
    "csrc/attention/paged_attention_v2.cu"
    "csrc/attention/merge_attn_states.cu"
    "csrc/attention/vertical_slash_index.cu"
    "csrc/pos_encoding_kernels.cu"
    "csrc/activation_kernels.cu"
    "csrc/layernorm_kernels.cu"
    "csrc/fused_qknorm_rope_kernel.cu"
    "csrc/layernorm_quant_kernels.cu"
    "csrc/sampler.cu"
    "csrc/topk.cu"
    "csrc/cuda_view.cu"
    "csrc/quantization/gptq/q_gemm.cu"
    "csrc/quantization/w8a8/int8/scaled_quant.cu"
    "csrc/quantization/w8a8/fp8/common.cu"
    "csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
    "csrc/quantization/gguf/gguf_kernel.cu"
    "csrc/quantization/activation_kernels.cu"
    "csrc/cuda_utils_kernels.cu"
    "csrc/custom_all_reduce.cu"
    "csrc/torch_bindings.cpp"
    "csrc/minimax_reduce_rms_kernel.cu")

SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
set(CUTLASS_REVISION "v4.2.1")

if (DEFINED ENV{VLLM_CUTLASS_SRC_DIR})
    set(VLLM_CUTLASS_SRC_DIR $ENV{VLLM_CUTLASS_SRC_DIR})
endif()

if(VLLM_CUTLASS_SRC_DIR)
    if(NOT IS_ABSOLUTE VLLM_CUTLASS_SRC_DIR)
        get_filename_component(VLLM_CUTLASS_SRC_DIR "${VLLM_CUTLASS_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The VLLM_CUTLASS_SRC_DIR is set, using ${VLLM_CUTLASS_SRC_DIR} for compilation")
    FetchContent_Declare(cutlass SOURCE_DIR ${VLLM_CUTLASS_SRC_DIR})
else()
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/nvidia/cutlass.git
        GIT_TAG ${CUTLASS_REVISION}
        GIT_PROGRESS TRUE
        GIT_SHALLOW TRUE
    )
endif()

FetchContent_MakeAvailable(cutlass)

list(APPEND VLLM_EXT_SRC
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"
    "csrc/quantization/fp4/nvfp4_quant_entry.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"
    "csrc/cutlass_extensions/common.cpp")

set_gencode_flags_for_srcs(
    SRCS "${VLLM_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

# Marlin SM 7.5 - remove for feature
cuda_archs_loose_intersection(MARLIN_ARCHS "7.0+PTX;8.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_SM75_ARCHS "7.0;7.5" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "7.0+PTX;8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0;12.1" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "7.0;7.5;8.0+PTX" "${CUDA_ARCHS}")

if (MARLIN_OTHER_ARCHS)
    set(MARLIN_GEN_SCRIPT
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/marlin/generate_kernels.py)
    file(MD5 ${MARLIN_GEN_SCRIPT} MARLIN_GEN_SCRIPT_HASH)
    list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
    set(MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")
    message(STATUS "Marlin generation script hash: ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")

    if (NOT DEFINED CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
        OR NOT $CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=$ENV{PYTHONPATH}
                ${Python_EXECUTABLE} ${MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
            RESULT_VARIABLE marlin_generation_result
            OUTPUT_VARIABLE marlin_generation_result
            OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
            ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
        )
        if (NOT marlin_generation_result EQUAL 0)
            message(FATAL_ERROR "Marlin generation failed."
                " Result: \"${marlin_generation_result}\""
                "\nCheck the log for details: "
                "${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log")
        else()
            set(MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
                CACHE STRING "Last run Marlin generate script hash and arch" FORCE)
        endif()
    endif()

    if (MARLIN_ARCHS)
        file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "csrc/quantization/marlin/sm80_kernel_*_float16.cu")
        set_gencode_flags_for_srcs(SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}" CUDA_ARCHS "${MARLIN_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
            set_source_files_properties(${MARLIN_TEMPLATE_KERNEL_SRC}
                PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})

        file(GLOB MARLIN_TEMPLATE_BF16_KERNEL_SRC "csrc/quantization/marlin/sm80_kernel_*_bfloat16.cu")
        set_gencode_flags_for_srcs(SRCS "${MARLIN_TEMPLATE_BF16_KERNEL_SRC}" CUDA_ARCHS "${MARLIN_BF16_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
            set_source_files_properties(${MARLIN_TEMPLATE_BF16_KERNEL_SRC}
                PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_BF16_KERNEL_SRC})
    endif()

    if (MARLIN_SM75_ARCHS)
        file(GLOB MARLIN_TEMPLATE_SM75_KERNEL_SRC "csrc/quantization/marlin/sm75_kernel_*.cu")
        set_gencode_flags_for_srcs(SRCS "${MARLIN_TEMPLATE_SM75_KERNEL_SRC}" CUDA_ARCHS "${MARLIN_SM75_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
            set_source_files_properties(${MARLIN_TEMPLATE_SM75_KERNEL_SRC}
                PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_SM75_KERNEL_SRC})
    endif()

    if (MARLIN_FP8_ARCHS)
        file(GLOB MARLIN_TEMPLATE_FP8_KERNEL_SRC "csrc/quantization/marlin/sm89_kernel_*.cu")
        set_gencode_flags_for_srcs(SRCS "${MARLIN_TEMPLATE_FP8_KERNEL_SRC}" CUDA_ARCHS "${MARLIN_FP8_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
            set_source_files_properties(${MARLIN_TEMPLATE_FP8_KERNEL_SRC}
                PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_FP8_KERNEL_SRC})
    endif()

    set(MARLIN_SRCS
        "csrc/quantization/marlin/marlin.cu"
        "csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu"
        "csrc/quantization/marlin/gptq_marlin_repack.cu"
        "csrc/quantization/marlin/awq_marlin_repack.cu")
    set_gencode_flags_for_srcs(SRCS "${MARLIN_SRCS}" CUDA_ARCHS "${MARLIN_OTHER_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_SRCS}
            PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_EXT_SRC "${MARLIN_SRCS}")
    message(STATUS "Building Marlin kernels for archs: ${MARLIN_OTHER_ARCHS}")
endif()

# AllSpark (SM 8.0+) - remove for feature
cuda_archs_loose_intersection(ALLSPARK_ARCHS "8.0;8.6;8.7;8.9" "${CUDA_ARCHS}")
if (ALLSPARK_ARCHS)
    set(ALLSPARK_SRCS
        "csrc/quantization/gptq_allspark/allspark_repack.cu"
        "csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu")
    set_gencode_flags_for_srcs(SRCS "${ALLSPARK_SRCS}" CUDA_ARCHS "${ALLSPARK_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${ALLSPARK_SRCS}")
endif()

# Scaled MM 3x (SM 9.0 Hopper) - remove for feature
set(SCALED_MM_3X_ARCHS)
cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
    set(SRCS
        "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm90.cu"
        "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu"
        "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_int8.cu"
        "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_azp_sm90_int8.cu"
        "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM90=1")
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
endif()

# Scaled MM 3x (SM 12.0 Blackwell) - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS
        "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu"
        "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu"
        "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM120=1")
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
endif()

# Scaled MM 3x (SM 10.0/11.0 Blackwell) - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS
        "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm100.cu"
        "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu"
        "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM100=1")
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
endif()

# Scaled MM 2x (SM 7.0+ Volta/Turing/Ampere/Ada)
cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
    "7.0;7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
if (SCALED_MM_2X_ARCHS)
    set(SRCS "csrc/quantization/w8a8/cutlass/scaled_mm_c2x.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${SCALED_MM_2X_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")
endif()

# NVFP4 SM 12.0 Blackwell - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FP4_ARCHS "12.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(FP4_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
    set(SRCS
        "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
        "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
        "csrc/quantization/fp4/nvfp4_experts_quant.cu"
        "csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu"
        "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${FP4_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM120=1")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM120=1")
endif()

# NVFP4 SM 10.0/11.0 Blackwell - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FP4_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(FP4_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
    set(SRCS
        "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
        "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
        "csrc/quantization/fp4/nvfp4_experts_quant.cu"
        "csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
        "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${FP4_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM100=1")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
endif()

# CUTLASS MLA (SM 10.0+ Blackwell) - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(MLA_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(MLA_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND MLA_ARCHS)
    set(SRCS "csrc/attention/mla/sm100_cutlass_mla_kernel.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${MLA_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MLA=1")
    set_source_files_properties(${SRCS}
        PROPERTIES INCLUDE_DIRECTORIES "${CUTLASS_DIR}/examples/77_blackwell_fmha;${CUTLASS_DIR}/examples/common")
endif()

# CUTLASS MoE SM 9.0 - remove for feature
cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
    set(SRCS "csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm90.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
endif()

# CUTLASS MoE SM 10.0 - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS "csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
endif()

# ES MXFP8 grouped (SM 10.0+) - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(ES_MXFP8_GROUPED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(ES_MXFP8_GROUPED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND ES_MXFP8_GROUPED_MM_ARCHS)
    set(SRCS
        "csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm.cu"
        "csrc/moe/mxfp8_moe/mxfp8_experts_quant.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${ES_MXFP8_GROUPED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_ES_MXFP8_GROUPED_MM_SM100=1")
endif()

# DSV3 fused A GEMM (SM 9.0+) - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0f;11.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND DSV3_FUSED_A_GEMM_ARCHS)
    set(DSV3_FUSED_A_GEMM_SRC "csrc/dsv3_fused_a_gemm.cu")
    set_gencode_flags_for_srcs(SRCS "${DSV3_FUSED_A_GEMM_SRC}" CUDA_ARCHS "${DSV3_FUSED_A_GEMM_ARCHS}")
    list(APPEND VLLM_EXT_SRC ${DSV3_FUSED_A_GEMM_SRC})
endif()

# moe_data.cu (SM 9.0+ / 10.0+ / 12.0+) - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
    set(SRCS "csrc/quantization/w8a8/cutlass/moe/moe_data.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${CUTLASS_MOE_DATA_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
endif()

# Machete (SM 9.0a Hopper) - remove for feature
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()

cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND MACHETE_ARCHS)
    set(MACHETE_GEN_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/machete/generate.py)
    file(MD5 ${MACHETE_GEN_SCRIPT} MACHETE_GEN_SCRIPT_HASH)

    if (NOT DEFINED CACHE{MACHETE_GEN_SCRIPT_HASH}
        OR NOT $CACHE{MACHETE_GEN_SCRIPT_HASH} STREQUAL ${MACHETE_GEN_SCRIPT_HASH})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_extensions/:${CUTLASS_DIR}/python/:${VLLM_PYTHON_PATH}:$ENV{PYTHONPATH}
                ${Python_EXECUTABLE} ${MACHETE_GEN_SCRIPT}
            RESULT_VARIABLE machete_generation_result
            OUTPUT_VARIABLE machete_generation_output
            OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
            ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
        )
        if (NOT machete_generation_result EQUAL 0)
            message(FATAL_ERROR "Machete generation failed."
                " Result: \"${machete_generation_result}\""
                "\nCheck the log for details: "
                "${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log")
        else()
            set(MACHETE_GEN_SCRIPT_HASH ${MACHETE_GEN_SCRIPT_HASH}
                CACHE STRING "Last run machete generate script hash" FORCE)
        endif()
    endif()

    file(GLOB MACHETE_GEN_SOURCES "csrc/quantization/machete/generated/*.cu")
    list(APPEND VLLM_EXT_SRC ${MACHETE_GEN_SOURCES})
    set_gencode_flags_for_srcs(SRCS "${MACHETE_GEN_SOURCES}" CUDA_ARCHS "${MACHETE_ARCHS}")
    list(APPEND VLLM_EXT_SRC csrc/quantization/machete/machete_pytorch.cu)
endif()

# W4A8 (SM 9.0a) - remove for feature
cuda_archs_loose_intersection(W4A8_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND W4A8_ARCHS)
    set(SRCS
        "csrc/quantization/cutlass_w4a8/w4a8_mm_entry.cu"
        "csrc/quantization/cutlass_w4a8/w4a8_grouped_mm_entry.cu"
        "csrc/quantization/cutlass_w4a8/w4a8_utils.cu")
    set_gencode_flags_for_srcs(SRCS "${SRCS}" CUDA_ARCHS "${W4A8_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
endif()

message(STATUS "Enabling C extension.")
define_extension_target(
    _C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
    USE_SABI 3
    WITH_SOABI)

target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

set(VLLM_STABLE_EXT_SRC
    "csrc/libtorch_stable/torch_bindings.cpp"
    "csrc/libtorch_stable/permute_cols.cu"
    "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
    "csrc/libtorch_stable/quantization/w8a8/int8/per_token_group_quant.cu")

set_gencode_flags_for_srcs(
    SRCS "${VLLM_STABLE_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

message(STATUS "Enabling C_stable extension.")
define_extension_target(
    _C_stable_libtorch
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_STABLE_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)

target_compile_definitions(_C_stable_libtorch PRIVATE
    TORCH_TARGET_VERSION=0x020A000000000000ULL)
target_compile_definitions(_C_stable_libtorch PRIVATE USE_CUDA)

set(VLLM_MOE_EXT_SRC
    "csrc/moe/torch_bindings.cpp"
    "csrc/moe/moe_align_sum_kernels.cu"
    "csrc/moe/topk_softmax_kernels.cu"
    "csrc/moe/moe_wna16.cu"
    "csrc/moe/grouped_topk_kernels.cu"
    "csrc/moe/gpt_oss_router_gemm.cu"
    "csrc/moe/router_gemm.cu")

set(MOE_PERMUTE_SRC
    "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
    "csrc/moe/moe_permute_unpermute_op.cu")
list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")

set_gencode_flags_for_srcs(
    SRCS "${VLLM_MOE_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "7.0+PTX;8.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_SM75_ARCHS "7.0;7.5" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0;12.1" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.0;7.5;8.0+PTX" "${CUDA_ARCHS}")

if (MARLIN_MOE_OTHER_ARCHS)
    set(MOE_MARLIN_GEN_SCRIPT
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py)
    file(MD5 ${MOE_MARLIN_GEN_SCRIPT} MOE_MARLIN_GEN_SCRIPT_HASH)
    list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
    set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MOE_MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")

    if (NOT DEFINED CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
        OR NOT $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=$ENV{PYTHONPATH}
                ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
            RESULT_VARIABLE moe_marlin_generation_result
            OUTPUT_VARIABLE moe_marlin_generation_output
            OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
            ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
        )
        if (NOT moe_marlin_generation_result EQUAL 0)
            message(FATAL_ERROR "Marlin MOE generation failed."
                " Result: \"${moe_marlin_generation_result}\""
                "\nCheck the log for details: "
                "${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log")
        else()
            set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
                CACHE STRING "Last run Marlin MOE generate script hash" FORCE)
        endif()
    endif()

    if (MARLIN_MOE_ARCHS)
        file(GLOB MARLIN_MOE_SRC "csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu")
        set_gencode_flags_for_srcs(SRCS "${MARLIN_MOE_SRC}" CUDA_ARCHS "${MARLIN_MOE_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
            set_source_files_properties(${MARLIN_MOE_SRC}
                PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SRC})
    endif()

    if (MARLIN_MOE_SM75_ARCHS)
        file(GLOB MARLIN_MOE_SM75_SRC "csrc/moe/marlin_moe_wna16/sm75_kernel_*.cu")
        set_gencode_flags_for_srcs(SRCS "${MARLIN_MOE_SM75_SRC}" CUDA_ARCHS "${MARLIN_MOE_SM75_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
            set_source_files_properties(${MARLIN_MOE_SM75_SRC}
                PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SM75_SRC})
    endif()

    if (MARLIN_MOE_FP8_ARCHS)
        file(GLOB MARLIN_MOE_FP8_SRC "csrc/moe/marlin_moe_wna16/sm89_kernel_*.cu")
        set_gencode_flags_for_srcs(SRCS "${MARLIN_MOE_FP8_SRC}" CUDA_ARCHS "${MARLIN_MOE_FP8_ARCHS}")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
            set_source_files_properties(${MARLIN_MOE_FP8_SRC}
                PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
        endif()
        list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_FP8_SRC})
    endif()

    set(MARLIN_MOE_OTHER_SRC "csrc/moe/marlin_moe_wna16/ops.cu")
    set_gencode_flags_for_srcs(SRCS "${MARLIN_MOE_OTHER_SRC}" CUDA_ARCHS "${MARLIN_MOE_OTHER_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_MOE_OTHER_SRC}
            PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_MOE_EXT_SRC "${MARLIN_MOE_OTHER_SRC}")
endif()

# DSV3 router GEMM (SM 9.0+) - remove for a futere
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(DSV3_ROUTER_GEMM_ARCHS "9.0a;10.0f;11.0f" "${CUDA_ARCHS}")
else()
    cuda_archs_loose_intersection(DSV3_ROUTER_GEMM_ARCHS "9.0a;10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
endif()
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND DSV3_ROUTER_GEMM_ARCHS)
    set(DSV3_ROUTER_GEMM_SRC
        "csrc/moe/dsv3_router_gemm_entry.cu"
        "csrc/moe/dsv3_router_gemm_float_out.cu"
        "csrc/moe/dsv3_router_gemm_bf16_out.cu")
    set_gencode_flags_for_srcs(SRCS "${DSV3_ROUTER_GEMM_SRC}" CUDA_ARCHS "${DSV3_ROUTER_GEMM_ARCHS}")
    list(APPEND VLLM_MOE_EXT_SRC "${DSV3_ROUTER_GEMM_SRC}")
endif()

message(STATUS "Enabling moe extension.")
define_extension_target(
    _moe_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_MOE_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
    INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
    USE_SABI 3
    WITH_SOABI)

include(cmake/external_projects/triton_kernels.cmake)
