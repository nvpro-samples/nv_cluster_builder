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

#include <cfloat>
#include <cstdint>

#define NVCLUSTER_VERSION 1

namespace nvcluster {

// Axis aligned bounding box
struct AABB
{
  float bboxMin[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
  float bboxMax[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
};

// An index/cursor based subrange
struct Range
{
  uint32_t offset = 0u;
  uint32_t count  = 0u;
};


// Spatial definition of elements to cluster
struct SpatialElements
{
  // Bounding boxes of elements to cluster
  const AABB* boundingBoxes = nullptr;

  // Center positions (xyz) of elements to cluster
  const float* centroids = nullptr;

  // Number of elements
  uint32_t elementCount = 0u;
};

// Definition of the connectivity of the items to cluster
// While the data structure defines unidirectional connections the underlying
// clustering requires all connections to be explicitly bidirectional (if node A has a connection to node B, then node B has
// a connection to node A)
struct Graph
{
  // Each node is defined by its connections to other nodes, stored at node.offset in connectionTargets. Each node has node.count connections
  const Range* nodes = nullptr;
  // Total number of nodes in the graph
  uint32_t nodeCount = 0u;


  // Connection targets for the nodes, referenced by nodes
  const uint32_t* connectionTargets = nullptr;
  // Weight of each connection
  const float* connectionWeights = nullptr;
  // Total number of connections in the graph
  uint32_t connectionCount = 0u;
};

// Clustering knobs, including cost balancing
struct Config
{
  // Minimum number of elements contained in a cluster
  uint32_t minClusterSize = 1u;
  // Maximum number of elements contained in a cluster
  uint32_t maxClusterSize = ~0u;
  // Cost penalty for under-filling clusters
  float costUnderfill = 1.0f;
  // Cost penalty for overlapping bounding boxes
  float costOverlap = 0.01f;
  // If nonzero the set of input elements will first be split along its median
  // until each subset contains at most preSplitThreshold elements prior to actual
  // clustering. This is an optimization intended to speed up clustering of large
  // sets of elements
  uint32_t preSplitThreshold = 0u;
};

// Definition of the input data
struct Input
{
  // Clustering configuration
  Config config;
  // Set of elements to cluster, required
  const SpatialElements* spatialElements = nullptr;
  // Optional graph defining the weighted connectivity between elements, used to optimize a cost function
  // when clustering
  const Graph* graph = nullptr;
};


// Structure to request the memory requirements to cluster the input data
struct ClusterGetRequirementsInfo
{
  // Input elements to cluster
  const nvcluster::Input* input = nullptr;
};

// Memory requirements for clustering
struct Requirements
{
  // Maximum number of generated clusters
  uint32_t maxClusterCount = 0u;
  // Maximum total number of elements referenced by the clusters
  uint32_t maxClusteredElementCount = 0u;
};

// Structure to request clustering the input set of elements
struct ClusterCreateInfo
{
  // Input elements to cluster
  const nvcluster::Input* input = nullptr;
};

// Structure to request the memory requirements to individually cluster the provided segments of the input data
struct ClusterGetRequirementsSegmentedInfo
{
  // Input elements to cluster
  const nvcluster::Input* input = nullptr;
  // Each segment is defined by a range within the array of elements defined in input
  const nvcluster::Range* elementSegments = nullptr;
  // Number of segments
  uint32_t elementSegmentCount = 0u;
};

// Structure to request individual clustering of the provided segments of the input data
struct ClusterCreateSegmentedInfo
{
  const nvcluster::Input* input = nullptr;

  // Each segment is defined by a range within the array of elements defined in input
  const nvcluster::Range* elementSegments = nullptr;
  // Number of segments
  uint32_t elementSegmentCount = 0u;
};

// Output of the clustering where the clusterRanges define a partition of the input elements
struct Output
{
  // Clusters defined by ranges of element indices, where each cluster starts at range.offset in clusteredElementIndices and contains range.count elements
  Range* clusterRanges = nullptr;
  // Indices of the elements, referenced by clusterRanges
  uint32_t* clusteredElementIndices = nullptr;
  // Total number of clusters generated by the clustering (may be less than maxClusterCount)
  uint32_t clusterCount = 0u;
  // Total number of cluster element indices (FIXME why, shouldn't this be the same as the input?)
  uint32_t clusteredElementIndexCount = 0u;
};


struct Context_t;
typedef Context_t* Context;

struct ContextCreateInfo
{
  uint32_t version = NVCLUSTER_VERSION;
};

enum Result
{
  SUCCESS,
  ERROR_INVALID_CREATE_INFO,
  ERROR_INTERNAL,  // Should not have to use this, find more explicit errors where this one is used
  ERROR_INVALID_CONFIG,
  ERROR_INVALID_BOUNDS,
  ERROR_INVALID_GRAPH,
  ERROR_WEIGHT_OVERFLOW,
  ERROR_INVALID_ARGUMENT,
  ERROR_INVALID_CONTEXT
};

}  // namespace nvcluster

extern "C" {


nvcluster::Result nvclusterCreateContext(const nvcluster::ContextCreateInfo* info, nvcluster::Context* context);
nvcluster::Result nvclusterDestroyContext(nvcluster::Context context);

// Usage:
// 1. call nvclusterGetRequirements(...) / nvclusterGetRequirementsSegmented(...) to get conservative sizes
// 2. allocate Output data
// 3. call nvclusterCreate(...) / nvclusterCreateSegmented(...)
// 4. resize down to what was written
// Alternatively use nvcluster::ClusterStorage / SegmentedClusterStorage, which encapsulates the above
nvcluster::Result nvclusterGetRequirements(nvcluster::Context                           context,
                                           const nvcluster::ClusterGetRequirementsInfo* info,
                                           nvcluster::Requirements*                     requirements);
nvcluster::Result nvclusterCreate(nvcluster::Context context, const nvcluster::ClusterCreateInfo* info, nvcluster::Output* clusters);


nvcluster::Result nvclusterGetRequirementsSegmented(nvcluster::Context                                    context,
                                                    const nvcluster::ClusterGetRequirementsSegmentedInfo* info,
                                                    nvcluster::Requirements*                              requirements);
nvcluster::Result nvclustersCreateSegmented(nvcluster::Context                           context,
                                            const nvcluster::ClusterCreateSegmentedInfo* info,
                                            nvcluster::Output*                           clusters,
                                            nvcluster::Range*                            clusterSegments);
}