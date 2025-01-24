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

#pragma once
#include <vector>

#include <nvcluster/nvcluster.h>


namespace nvcluster {

// Shortcut and storage for flat clustering output
struct ClusterStorage
{
  std::vector<Range>    clusterRanges;
  std::vector<uint32_t> clusterItems;
};


inline nvcluster::Result generateClusters(nvcluster::Context context, const Input& input, ClusterStorage& clusterStorage)
{

  ClusterGetRequirementsInfo info{.input = &input};


  Requirements      reqs;
  nvcluster::Result result = nvclusterGetRequirements(context, &info, &reqs);

  if(result != nvcluster::Result::SUCCESS)
  {
    return result;
  }

  clusterStorage.clusterRanges.resize(reqs.maxClusterCount);
  clusterStorage.clusterItems.resize(reqs.maxClusteredElementCount);

  ClusterCreateInfo createInfo;
  createInfo.input = &input;

  Output clusters;
  clusters.clusteredElementIndices    = clusterStorage.clusterItems.data();
  clusters.clusterRanges              = clusterStorage.clusterRanges.data();
  clusters.clusterCount               = reqs.maxClusterCount;
  clusters.clusteredElementIndexCount = reqs.maxClusteredElementCount;
  result                              = nvclusterCreate(context, &createInfo, &clusters);

  if(result == nvcluster::Result::SUCCESS)
  {
    clusterStorage.clusterRanges.resize(clusters.clusterCount);
    clusterStorage.clusterItems.resize(clusters.clusteredElementIndexCount);
  }
  return result;
}

inline void clearClusters(ClusterStorage& clusterStorage)
{
  clusterStorage.clusterRanges.clear();
  clusterStorage.clusterItems.clear();
}


struct SegmentedClusterStorage
{
  std::vector<Range>    clusterRangeSegments;
  std::vector<Range>    clusterRanges;
  std::vector<uint32_t> clusterItems;
};

inline nvcluster::Result generateSegmentedClusters(nvcluster::Context       context,
                                                   const Input&             input,
                                                   const Range*             itemSegments,
                                                   uint32_t                 itemSegmentCount,
                                                   SegmentedClusterStorage& segmentedClusterStorage)
{
  segmentedClusterStorage.clusterRangeSegments.resize(itemSegmentCount);

  ClusterGetRequirementsSegmentedInfo info;
  info.input               = &input;
  info.elementSegmentCount = itemSegmentCount;
  info.elementSegments     = itemSegments;

  Requirements      reqs;
  nvcluster::Result result = nvclusterGetRequirementsSegmented(context, &info, &reqs);
  if(result != nvcluster::Result::SUCCESS)
  {
    return result;
  }
  segmentedClusterStorage.clusterRanges.resize(reqs.maxClusterCount);
  segmentedClusterStorage.clusterItems.resize(reqs.maxClusteredElementCount);


  ClusterCreateSegmentedInfo createInfo;
  createInfo.input               = &input;
  createInfo.elementSegmentCount = itemSegmentCount;
  createInfo.elementSegments     = itemSegments;

  Output clusters;
  clusters.clusteredElementIndices    = segmentedClusterStorage.clusterItems.data();
  clusters.clusterRanges              = segmentedClusterStorage.clusterRanges.data();
  clusters.clusterCount               = reqs.maxClusterCount;
  clusters.clusteredElementIndexCount = reqs.maxClusteredElementCount;

  result = nvclustersCreateSegmented(context, &createInfo, &clusters, segmentedClusterStorage.clusterRangeSegments.data());

  if(result == nvcluster::Result::SUCCESS)
  {

    segmentedClusterStorage.clusterRanges.resize(clusters.clusterCount);
    segmentedClusterStorage.clusterItems.resize(clusters.clusteredElementIndexCount);
  }
  return result;
}

inline void clearSegmentedClusters(SegmentedClusterStorage& segmentedClusterStorage)
{
  segmentedClusterStorage.clusterRangeSegments.clear();
  segmentedClusterStorage.clusterRanges.clear();
  segmentedClusterStorage.clusterItems.clear();
}


}  // namespace nvcluster
