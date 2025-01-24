/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvcluster/nvcluster.h"
#include "clusterizer.hpp"

#include <stddef.h>

nvcluster::Result nvclusterCreateContext(const nvcluster::ContextCreateInfo* createInfo, nvcluster::Context* context)
{
  if(createInfo == nullptr || context == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_ARGUMENT;
  }

  *context = new nvcluster::Context_t{createInfo->version};

  return nvcluster::Result::SUCCESS;
}


nvcluster::Result nvclusterDestroyContext(nvcluster::Context context)
{
  if(context == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_CONTEXT;
  }

  delete context;

  return nvcluster::Result::SUCCESS;
}


nvcluster::Result nvclusterGetRequirements(nvcluster::Context                           context,
                                           const nvcluster::ClusterGetRequirementsInfo* info,
                                           nvcluster::Requirements*                     requirements)
{
  if(context == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_CONTEXT;
  }

  if(requirements == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_ARGUMENT;
  }

  if(info->input->config.minClusterSize == 0 || info->input->config.maxClusterSize == 0)
  {
    // FIXME: return error code
    *requirements = {~0u, ~0u};
    return nvcluster::Result::ERROR_INVALID_CONFIG;
  }
  const size_t n           = info->input->spatialElements->elementCount;
  const size_t Ca          = info->input->config.minClusterSize;
  const size_t P           = info->input->config.preSplitThreshold;
  const size_t P_underfill = P == 0 ? 0 : (n + P - 1) / P;
  const size_t maxClusters = (n + Ca - 1) / Ca + P_underfill;


  requirements->maxClusteredElementCount = uint32_t(n);
  requirements->maxClusterCount          = uint32_t(maxClusters);

  return nvcluster::Result::SUCCESS;
}
nvcluster::Result nvclusterCreate(nvcluster::Context context, const nvcluster::ClusterCreateInfo* info, nvcluster::Output* clusters)
{

  if(context == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_CONTEXT;
  }


  if(info == nullptr || info->input == nullptr || info->input->spatialElements == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_CREATE_INFO;
  }

  nvcluster::Result result = nvcluster_internal::clusterize(*info->input, *clusters);
  return result;
}


nvcluster::Result nvclusterGetRequirementsSegmented(nvcluster::Context                                    context,
                                                    const nvcluster::ClusterGetRequirementsSegmentedInfo* info,
                                                    nvcluster::Requirements*                              requirements)
{

  if(context == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_CONTEXT;
  }

  if(requirements == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_ARGUMENT;
  }

  requirements->maxClusterCount          = 0u;
  requirements->maxClusteredElementCount = 0u;

  for(uint32_t itemSegmentIndex = 0; itemSegmentIndex < info->elementSegmentCount; itemSegmentIndex++)
  {
    const nvcluster::Range& range = info->elementSegments[itemSegmentIndex];

    nvcluster::ClusterGetRequirementsInfo segmentInfo{};
    nvcluster::Input                      segmentInput;
    nvcluster::SpatialElements            segmentBounds;
    segmentInput.config          = info->input->config;
    segmentBounds.elementCount   = range.count;
    segmentInput.spatialElements = &segmentBounds;
    segmentInfo.input            = &segmentInput;

    nvcluster::Requirements segmentResult;
    nvcluster::Result       res = nvclusterGetRequirements(context, &segmentInfo, &segmentResult);
    if(res != nvcluster::Result::SUCCESS)
    {
      return res;
    }
    requirements->maxClusterCount += segmentResult.maxClusterCount;
    requirements->maxClusteredElementCount += segmentResult.maxClusteredElementCount;
  }
  return nvcluster::Result::SUCCESS;
}
nvcluster::Result nvclustersCreateSegmented(nvcluster::Context                           context,
                                            const nvcluster::ClusterCreateSegmentedInfo* info,
                                            nvcluster::Output*                           clusters,
                                            nvcluster::Range*                            clusterSegments)
{

  if(context == nullptr)
  {
    return nvcluster::Result::ERROR_INVALID_CONTEXT;
  }


  nvcluster::Requirements sizes = {0, 0};

  for(uint32_t segmentIndex = 0; segmentIndex < info->elementSegmentCount; segmentIndex++)
  {
    const nvcluster::Range&    range = info->elementSegments[segmentIndex];
    nvcluster::Input           segmentInput{};
    nvcluster::SpatialElements segmentBounds{};
    segmentBounds.boundingBoxes  = info->input->spatialElements->boundingBoxes + range.offset;
    segmentBounds.centroids      = info->input->spatialElements->centroids + 3 * range.offset;
    segmentBounds.elementCount   = range.count;
    segmentInput.spatialElements = &segmentBounds;
    segmentInput.config          = info->input->config;
    segmentInput.graph           = info->input->graph;

    if(sizes.maxClusteredElementCount + range.count > clusters->clusteredElementIndexCount)
    {
      return nvcluster::Result::ERROR_INTERNAL;
    }

    nvcluster::Output segmentedOutput{.clusterRanges = clusters->clusterRanges + sizes.maxClusterCount,
                                      .clusteredElementIndices = clusters->clusteredElementIndices + sizes.maxClusteredElementCount,
                                      .clusterCount               = clusters->clusterCount - sizes.maxClusterCount,
                                      .clusteredElementIndexCount = range.count};


    if(segmentInput.spatialElements == nullptr)
    {
      return nvcluster::Result::ERROR_INVALID_BOUNDS;
    }


    nvcluster::Result result = nvcluster_internal::clusterize(segmentInput, segmentedOutput);
    if(result != nvcluster::Result::SUCCESS)
    {
      return result;
    }

    if(sizes.maxClusterCount + segmentedOutput.clusterCount > clusters->clusterCount)
    {
      return nvcluster::Result::ERROR_INTERNAL;
    }

    // Translate local ranges offsets and item indices to global
    for(uint32_t rangeIndex = 0; rangeIndex < segmentedOutput.clusterCount; rangeIndex++)
    {
      nvcluster::Range& clusterRange = segmentedOutput.clusterRanges[rangeIndex];
      clusterRange.offset += sizes.maxClusteredElementCount;
    }

    for(uint32_t itemIndex = 0; itemIndex < segmentedOutput.clusteredElementIndexCount; itemIndex++)
    {
      segmentedOutput.clusteredElementIndices[itemIndex] += range.offset;
    }
    // Emit the segment of clustered items
    clusterSegments[segmentIndex] = nvcluster::Range{sizes.maxClusterCount, segmentedOutput.clusterCount};
    sizes.maxClusterCount += segmentedOutput.clusterCount;
    sizes.maxClusteredElementCount += segmentedOutput.clusteredElementIndexCount;
  }


  /*********************************************/

  clusters->clusteredElementIndexCount = sizes.maxClusteredElementCount;
  clusters->clusterCount               = sizes.maxClusterCount;

  return nvcluster::Result::SUCCESS;
}