/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef _CUDNN_BACKEND_H_
#define _CUDNN_BACKEND_H_

/*
 * The content in this header file is under development to be included in cudnn.h in the future
 * Production code should have all include of this header file remove.
 */

#include "cudnn_ops_infer.h"
#include "cudnn_cnn_infer.h"

/* NOTE: definition in extern "C" to be copied later to public header */
#if defined(__cplusplus)
extern "C" {
#endif

typedef void *cudnnBackendDescriptor_t;

typedef enum {
    CUDNN_POINTWISE_ADD  = 0,
    CUDNN_POINTWISE_MUL  = 1,
    CUDNN_POINTWISE_MIN  = 2,
    CUDNN_POINTWISE_MAX  = 3,
    CUDNN_POINTWISE_SQRT = 4,

    CUDNN_POINTWISE_RELU_FWD    = 100,
    CUDNN_POINTWISE_TANH_FWD    = 101,
    CUDNN_POINTWISE_SIGMOID_FWD = 102,
    CUDNN_POINTWISE_ELU_FWD     = 103,
} cudnnPointwiseMode_t;

typedef enum {
    CUDNN_GENSTATS_SUM_SQSUM = 0,
} cudnnGenStatsMode_t;

typedef enum {
    CUDNN_ATTR_POINTWISE_MODE            = 0,
    CUDNN_ATTR_POINTWISE_MATH_PREC       = 1,
    CUDNN_ATTR_POINTWISE_NAN_PROPAGATION = 2,
    CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP = 3,
    CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP = 4,

    CUDNN_ATTR_CONVOLUTION_COMP_TYPE      = 100,
    CUDNN_ATTR_CONVOLUTION_CONV_MODE      = 101,
    CUDNN_ATTR_CONVOLUTION_DILATIONS      = 102,
    CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = 103,
    CUDNN_ATTR_CONVOLUTION_POST_PADDINGS  = 104,
    CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS   = 105,
    CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS   = 106,

    CUDNN_ATTR_ENGINEHEUR_MODE            = 200,
    CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201,
    CUDNN_ATTR_ENGINEHEUR_RESULTS         = 202,

    CUDNN_ATTR_ENGINECFG_ENGINE            = 300,
    CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO = 301,
    CUDNN_ATTR_ENGINECFG_KNOB_CHOICES      = 302,

    CUDNN_ATTR_EXECUTION_PLAN_HANDLE                     = 400,
    CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG              = 401,
    CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE             = 402,
    CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = 403,
    CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = 404,

    CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID            = 500,
    CUDNN_ATTR_INTERMEDIATE_INFO_SIZE                 = 501,
    CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS  = 502,
    CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = 503,

    CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE  = 600,
    CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE = 601,

    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA        = 700,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA         = 701,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC    = 702,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W            = 703,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X            = 704,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y            = 705,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA       = 706,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA        = 707,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC   = 708,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W           = 709,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX          = 710,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY          = 711,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA     = 712,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA      = 713,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = 714,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW        = 715,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X         = 716,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY        = 717,

    CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = 750,
    CUDNN_ATTR_OPERATION_POINTWISE_XDESC         = 751,
    CUDNN_ATTR_OPERATION_POINTWISE_BDESC         = 752,
    CUDNN_ATTR_OPERATION_POINTWISE_YDESC         = 753,
    CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1        = 754,
    CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2        = 755,

    CUDNN_ATTR_OPERATION_GENSTATS_MODE      = 770,
    CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = 771,
    CUDNN_ATTR_OPERATION_GENSTATS_XDESC     = 772,
    CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC   = 773,
    CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = 774,

    CUDNN_ATTR_OPERATIONGRAPH_HANDLE              = 800,
    CUDNN_ATTR_OPERATIONGRAPH_OPS                 = 801,
    CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 802,

    CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT       = 900,
    CUDNN_ATTR_TENSOR_DATA_TYPE            = 901,
    CUDNN_ATTR_TENSOR_DIMENSIONS           = 902,
    CUDNN_ATTR_TENSOR_STRIDES              = 903,
    CUDNN_ATTR_TENSOR_VECTOR_COUNT         = 904,
    CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION = 905,
    CUDNN_ATTR_TENSOR_UNIQUE_ID            = 906,
    CUDNN_ATTR_TENSOR_IS_VIRTUAL           = 907,

    CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS    = 1000,
    CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001,
    CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES = 1002,
    CUDNN_ATTR_VARIANT_PACK_WORKSPACE     = 1003,

    CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID = 1100,
    CUDNN_ATTR_LAYOUT_INFO_TYPES      = 1101,

    CUDNN_ATTR_KNOB_INFO_TYPE          = 1200,
    CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE = 1201,
    CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE = 1202,
    CUDNN_ATTR_KNOB_INFO_STRIDE        = 1203,

    CUDNN_ATTR_ENGINE_OPERATION_GRAPH = 1300,
    CUDNN_ATTR_ENGINE_GLOBAL_INDEX    = 1301,
    CUDNN_ATTR_ENGINE_KNOB_INFO       = 1302,
    CUDNN_ATTR_ENGINE_NUMERICAL_NOTE  = 1303,
    CUDNN_ATTR_ENGINE_LAYOUT_INFO     = 1304,
} cudnnBackendAttributeName_t;

typedef enum {
    CUDNN_TYPE_HANDLE = 0,
    CUDNN_TYPE_DATA_TYPE,
    CUDNN_TYPE_BOOLEAN,
    CUDNN_TYPE_INT64,
    CUDNN_TYPE_FLOAT,
    CUDNN_TYPE_DOUBLE,
    CUDNN_TYPE_VOID_PTR,
    CUDNN_TYPE_CONVOLUTION_MODE,
    CUDNN_TYPE_HEUR_MODE,
    CUDNN_TYPE_KNOB_TYPE,
    CUDNN_TYPE_NAN_PROPOGATION,
    CUDNN_TYPE_NUMERICAL_NOTE,
    CUDNN_TYPE_LAYOUT_TYPE,
    CUDNN_TYPE_ATTRIB_NAME,
    CUDNN_TYPE_POINTWISE_MODE,
    CUDNN_TYPE_BACKEND_DESCRIPTOR,
    CUDNN_TYPE_GENSTATS_MODE
} cudnnBackendAttributeType_t;

typedef enum {
    CUDNN_BACKEND_POINTWISE_DESCRIPTOR = 0,
    CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
    CUDNN_BACKEND_ENGINE_DESCRIPTOR,
    CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
    CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
    CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
    CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR,
    CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR,
    CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
    CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
    CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
    CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR,
} cudnnBackendDescriptorType_t;

typedef enum {
    CUDNN_NUMERICAL_NOTE_TENSOR_CORE = 0,
    CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS,
    CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION,
    CUDNN_NUMERICAL_NOTE_FFT,
    CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC,
    CUDNN_NUMERICAL_NOTE_WINOGRAD,
    CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
} cudnnBackendNumericalNote_t;

typedef enum {
    CUDNN_KNOB_TYPE_SPLIT_K          = 0,
    CUDNN_KNOB_TYPE_SWIZZLE          = 1,
    CUDNN_KNOB_TYPE_TILE_SIZE        = 2,
    CUDNN_KNOB_TYPE_USE_TEX          = 3,
    CUDNN_KNOB_TYPE_EDGE             = 4,
    CUDNN_KNOB_TYPE_KBLOCK           = 5,
    CUDNN_KNOB_TYPE_LDGA             = 6,
    CUDNN_KNOB_TYPE_LDGB             = 7,
    CUDNN_KNOB_TYPE_CHUNK_K          = 8,
    CUDNN_KNOB_TYPE_SPLIT_H          = 9,
    CUDNN_KNOB_TYPE_WINO_TILE        = 10,
    CUDNN_KNOB_TYPE_MULTIPLY         = 11,
    CUDNN_KNOB_TYPE_SPLIT_K_BUF      = 12,
    CUDNN_KNOB_TYPE_TILEK            = 13,
    CUDNN_KNOB_TYPE_STAGES           = 14,
    CUDNN_KNOB_TYPE_REDUCTION_MODE   = 15,
    CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE = 16,
    CUDNN_KNOB_TYPE_SPLIT_K_SLC      = 17,
    CUDNN_KNOB_TYPE_IDX_MODE         = 18,
    CUDNN_KNOB_TYPE_SLICED           = 19,
    CUDNN_KNOB_TYPE_SPLIT_RS         = 20,
    CUDNN_KNOB_TYPE_SINGLEBUFFER     = 21,
    CUDNN_KNOB_TYPE_LDGC             = 22,
    CUDNN_KNOB_TYPE_SPECFILT         = 23,

    CUDNN_KNOB_TYPE_COUNTS = 24,
} cudnnBackendKnobType_t;

typedef enum {
    CUDNN_LAYOUT_TYPE_PREFERRED_NCHW   = 0,
    CUDNN_LAYOUT_TYPE_PREFERRED_NHWC   = 1,
    CUDNN_LAYOUT_TYPE_PREFERRED_PAD4CK = 2,
    CUDNN_LAYOUT_TYPE_PREFERRED_PAD8CK = 3,
    CUDNN_LAYOUT_TYPE_COUNT            = 4,
} cudnnBackendLayoutType_t;

typedef enum { CUDNN_HEUR_MODE_INSTANT = 0, CUDNN_HEUR_MODES_COUNT } cudnnBackendHeurMode_t;

cudnnStatus_t CUDNNWINAPI
cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType, cudnnBackendDescriptor_t *descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendInitialize(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor);

cudnnStatus_t CUDNNWINAPI
cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t elementCount,
                         const void *arrayOfElements);

cudnnStatus_t CUDNNWINAPI
cudnnBackendGetAttribute(cudnnBackendDescriptor_t const descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t requestedElementCount,
                         int64_t *elementCount,
                         void *arrayOfElements);

cudnnStatus_t CUDNNWINAPI
cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t variantPack);

#if defined(__cplusplus)
}
#endif

#endif /* _CUDNN_BACKEND_H_ */
