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

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <nvcluster/nvcluster.h>
#include <nvcluster/nvcluster_storage.hpp>
#include <stddef.h>
#include <unordered_map>

// GLSL-like type definitions
using uvec3 = std::array<uint32_t, 3>;
using vec3  = std::array<float, 3>;

vec3 make_vec3(const float a[3])
{
  return vec3{a[0], a[1], a[2]};
}

// Adds two vec3s.
vec3 add(const vec3& a, const vec3& b)
{
  return vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

// Multiplies a vec3 by a constant value.
vec3 mul(const vec3& v, float a)
{
  return vec3{v[0] * a, v[1] * a, v[2] * a};
}

// Normalizes a vec3.
vec3 normalize(const vec3& v)
{
  const float lengthSquared = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const float factor        = (lengthSquared == 0.0f) ? 1.0f : (1.0f / std::sqrt(lengthSquared));
  return mul(v, factor);
}

// Define a hash for vec3, so that we can use it in std::unordered_map.
template <>
struct std::hash<vec3>
{
  std::size_t operator()(const vec3& v) const noexcept
  {
    // This doesn't need to be a good hash; it just needs to exist.
    const std::hash<float> hasher{};
    return hasher(v[0]) + 3 * hasher(v[1]) + 5 * hasher(v[2]);
  }
};

// Computes the axis-aligned bounding box of a triangle with the given indices.
const nvcluster::AABB aabb(uvec3 triangle, const vec3* positions)
{
  nvcluster::AABB result{};
  vec3            vertex = positions[triangle[0]];
  memcpy(result.bboxMin, vertex.data(), sizeof(vertex));
  memcpy(result.bboxMax, vertex.data(), sizeof(vertex));
  for(size_t i = 1; i < 3; i++)
  {
    vertex = positions[triangle[i]];
    for(size_t component = 0; component < 3; component++)
    {
      result.bboxMin[component] = std::min(result.bboxMin[component], vertex[component]);
      result.bboxMax[component] = std::min(result.bboxMax[component], vertex[component]);
    }
  }
  return result;
}

// Returns the center of a bounding box.
const vec3 centroid(const nvcluster::AABB& aabb)
{
  return mul(add(make_vec3(aabb.bboxMin), make_vec3(aabb.bboxMax)), 0.5f);
}

// Sort items within each cluster by their index and then the clusters based on
// the first item. This makes verifying clustering results easier.
void sortClusters(nvcluster::ClusterStorage& clustering)
{
  for(const nvcluster::Range& cluster : clustering.clusterRanges)
  {
    std::sort(clustering.clusterItems.begin() + cluster.offset,                   // Start of elements to sort
              clustering.clusterItems.begin() + cluster.offset + cluster.count);  // Exclusive end of elements to sort
  }

  std::sort(clustering.clusterRanges.begin(), clustering.clusterRanges.end(),
            [&clustering](nvcluster::Range& clusterA, nvcluster::Range& clusterB) {
              return clustering.clusterItems[clusterA.offset] < clustering.clusterItems[clusterB.offset];
            });
}

// Sorts clusters within segments, but not the segments themselves
void sortClusters(nvcluster::SegmentedClusterStorage& clustering)
{
  for(const nvcluster::Range& cluster : clustering.clusterRanges)
  {
    std::sort(clustering.clusterItems.begin() + cluster.offset,                   // Start of elements to sort
              clustering.clusterItems.begin() + cluster.offset + cluster.count);  // Exclusive end of elements to sort
  }

  for(auto& segment : clustering.clusterRangeSegments)
  {
    std::sort(clustering.clusterRanges.begin() + segment.offset,                  // Start
              clustering.clusterRanges.begin() + segment.offset + segment.count,  // Exclusive end
              [&clustering](nvcluster::Range& clusterA, nvcluster::Range& clusterB) {
                return clustering.clusterItems[clusterA.offset] < clustering.clusterItems[clusterB.offset];
              });
  }
}

// This nvcluster::Context cleans itself up so that we don't leak memory if
// a test exits early.
struct ScopedContext
{
  nvcluster::Context context = nullptr;

  nvcluster::Result init()
  {
    assert(!context);
    nvcluster::ContextCreateInfo info{};
    return nvclusterCreateContext(&info, &context);
  }

  ~ScopedContext() { std::ignore = nvclusterDestroyContext(context); }
};

// This is the test that appears in the README, so any changes to this test
// also should likely be reflected in the README file.
TEST(Clusters, Simple2x2)
{
  // Test items
  // 0 and 2 are close and should be in a cluster
  // 1 and 3 are close and should be in a cluster
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},
      {{0, 100, 0}, {1, 101, 1}},
      {{1, 0, 0}, {2, 1, 1}},
      {{1, 100, 0}, {2, 101, 1}},
  };

  // Generate centroids
  std::vector<vec3> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = centroid(boundingBoxes[i]);
  }

  // Input structs
  nvcluster::SpatialElements spatialElements{.boundingBoxes = boundingBoxes.data(),
                                             .centroids     = reinterpret_cast<const float*>(centroids.data()),
                                             .elementCount  = static_cast<uint32_t>(boundingBoxes.size())};
  nvcluster::Input           input{
                .config =
          {
                        .minClusterSize    = 2,
                        .maxClusterSize    = 2,
                        .costUnderfill     = 0.0f,
                        .costOverlap       = 0.0f,
                        .preSplitThreshold = 0,
          },
                .spatialElements = &spatialElements,
                .graph           = nullptr,
  };

  // Clustering
  ScopedContext context;
  ASSERT_EQ(context.init(), nvcluster::Result::SUCCESS);
  nvcluster::ClusterStorage clustering;
  nvcluster::Result         result = nvcluster::generateClusters(context.context, input, clustering);
  ASSERT_EQ(result, nvcluster::Result::SUCCESS);

  sortClusters(clustering);

  // Verify
  ASSERT_EQ(clustering.clusterRanges.size(), 2);
  ASSERT_EQ(clustering.clusterItems.size(), 4);
  const nvcluster::Range cluster0 = clustering.clusterRanges[0];
  ASSERT_EQ(cluster0.count, 2);
  EXPECT_EQ(clustering.clusterItems[cluster0.offset], 0);
  EXPECT_EQ(clustering.clusterItems[cluster0.offset + 1], 2);
  const nvcluster::Range cluster1 = clustering.clusterRanges[1];
  ASSERT_EQ(cluster1.count, 2);
  EXPECT_EQ(clustering.clusterItems[cluster1.offset], 1);
  EXPECT_EQ(clustering.clusterItems[cluster1.offset + 1], 3);

  for(size_t clusterIndex = 0; clusterIndex < clustering.clusterRanges.size(); ++clusterIndex)
  {
    const nvcluster::Range& range = clustering.clusterRanges[clusterIndex];
    for(uint32_t clusterItemIndex = 0; clusterItemIndex < range.count; ++clusterItemIndex)
    {
      uint32_t clusterItem = clustering.clusterItems[range.offset + clusterItemIndex];
      std::ignore          = clusterItem;
    }
  }
};

/*
 * Tests that weights affect the clusterizer's result.
 *
 * In the following diagram, v0 ... v3 are bounding boxes,
 * the edges are connections, and the `w` labels are weights:
 *
 *  v0 <- w1 -> v2
 *   ^           |
 *   |           |
 *   |           |
 * w1000       w1000
 *   |           |
 *   |           |
 *   v           v
 *  v1 <- w1 -> v3
 */
TEST(Clusters, Simple2x2Weights)
{
  // Test items
  // 0 and 2 are close and would normally be in a cluster
  // 1 and 3 are close and would normally be in a cluster
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}},
      {{0, 100, 0}, {1, 101, 1}},
      {{1, 0, 0}, {2, 1, 1}},
      {{1, 100, 0}, {2, 101, 1}},
  };

  // Adjacency/connections to override normal spatial clustering and instead
  // make clusters of {0, 1}, {2, 3}.
  std::vector<nvcluster::Range> graphNodes{{0, 2}, {2, 2}, {4, 2}, {6, 2}};  // 2 connections each
  std::vector<uint32_t>         connectionTargets{
      1, 2,  // item 0 connections
      0, 3,  // item 1 connections
      0, 3,  // item 2 connections
      1, 2,  // item 3 connections
  };
  std::vector<float> connectionWeights{
      1000.0f, 1.0f,     // weight from 0 to 1 and 2 respectively
      1000.0f, 1.0f,     // weight from 1 to 0 and 3 respectively
      1.0f,    1000.0f,  // weight from 2 to 0 and 3 respectively
      1.0f,    1000.0f,  // weight from 3 to 1 and 2 respectively
  };

  // Generate centroids
  std::vector<vec3> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = centroid(boundingBoxes[i]);
  }

  // Input structs
  nvcluster::SpatialElements spatialElements{.boundingBoxes = boundingBoxes.data(),
                                             .centroids     = reinterpret_cast<const float*>(centroids.data()),
                                             .elementCount  = static_cast<uint32_t>(boundingBoxes.size())};
  nvcluster::Graph           graph{
                .nodes             = graphNodes.data(),
                .nodeCount         = static_cast<uint32_t>(graphNodes.size()),
                .connectionTargets = connectionTargets.data(),
                .connectionWeights = connectionWeights.data(),
                .connectionCount   = static_cast<uint32_t>(connectionTargets.size()),
  };
  nvcluster::Input input{.config =
                             {
                                 .minClusterSize    = 2,
                                 .maxClusterSize    = 2,
                                 .costUnderfill     = 0.0f,
                                 .costOverlap       = 0.0f,
                                 .preSplitThreshold = 0,
                             },
                         .spatialElements = &spatialElements,
                         .graph           = &graph};

  // Clustering
  ScopedContext context;
  ASSERT_EQ(context.init(), nvcluster::Result::SUCCESS);
  nvcluster::ClusterStorage clustering;
  nvcluster::Result         result = nvcluster::generateClusters(context.context, input, clustering);
  ASSERT_EQ(result, nvcluster::Result::SUCCESS);

  sortClusters(clustering);

  // Verify
  ASSERT_EQ(clustering.clusterRanges.size(), 2);
  ASSERT_EQ(clustering.clusterItems.size(), 4);
  const nvcluster::Range cluster0 = clustering.clusterRanges[0];
  ASSERT_EQ(cluster0.count, 2);
  EXPECT_EQ(clustering.clusterItems[cluster0.offset], 0);
  EXPECT_EQ(clustering.clusterItems[cluster0.offset + 1], 1);
  const nvcluster::Range cluster1 = clustering.clusterRanges[1];
  ASSERT_EQ(cluster1.count, 2);
  EXPECT_EQ(clustering.clusterItems[cluster1.offset], 2);
  EXPECT_EQ(clustering.clusterItems[cluster1.offset + 1], 3);
};

TEST(Clusters, Segmented2x2)
{
  // Test items
  // 0 and 2 are close and should be in a cluster
  // 1 and 3 are close and should be in a cluster
  // Repeated 3 times for each segment with a slight x offset
  std::vector<nvcluster::AABB> boundingBoxes{
      {{0, 0, 0}, {1, 1, 1}}, {{0, 100, 0}, {1, 101, 1}}, {{1, 0, 0}, {2, 1, 1}}, {{1, 100, 0}, {2, 101, 1}},
      {{1, 0, 0}, {2, 1, 1}}, {{1, 100, 0}, {2, 101, 1}}, {{2, 0, 0}, {3, 1, 1}}, {{2, 100, 0}, {3, 101, 1}},
      {{2, 0, 0}, {3, 1, 1}}, {{2, 100, 0}, {3, 101, 1}}, {{3, 0, 0}, {4, 1, 1}}, {{3, 100, 0}, {4, 101, 1}},
  };

  // Segments
  // segment 0 should contain items 4 to 8
  // segment 1 should contain items 8 to 12
  // segment 2 should contain items 0 to 4
  std::vector<nvcluster::Range> segments{
      {4, 4},
      {8, 4},
      {0, 4},
  };

  // Generate centroids
  std::vector<vec3> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = centroid(boundingBoxes[i]);
  }

  // Input structs
  nvcluster::SpatialElements spatialElements{.boundingBoxes = boundingBoxes.data(),
                                             .centroids     = reinterpret_cast<const float*>(centroids.data()),
                                             .elementCount  = static_cast<uint32_t>(boundingBoxes.size())};
  nvcluster::Input           input{
                .config =
          {
                        .minClusterSize    = 2,
                        .maxClusterSize    = 2,
                        .costUnderfill     = 0.0f,
                        .costOverlap       = 0.0f,
                        .preSplitThreshold = 0,
          },
                .spatialElements = &spatialElements,
                .graph           = nullptr,
  };

  // Clustering
  ScopedContext context;
  ASSERT_EQ(context.init(), nvcluster::Result::SUCCESS);
  nvcluster::SegmentedClusterStorage clustering;
  nvcluster::Result result = nvcluster::generateSegmentedClusters(context.context, input, segments.data(),
                                                                  static_cast<uint32_t>(segments.size()), clustering);
  ASSERT_EQ(result, nvcluster::Result::SUCCESS);

  // Sort everything to validate items in clusters
  sortClusters(clustering);

  // Verify segment order remains consistent
  for(size_t segmentIndex = 0; segmentIndex < 3; ++segmentIndex)
  {
    const nvcluster::Range& segment      = clustering.clusterRangeSegments[segmentIndex];
    const nvcluster::Range& firstCluster = clustering.clusterRanges[segment.offset + 0];
    const uint32_t          firstItem    = clustering.clusterItems[firstCluster.offset + 0];
    EXPECT_EQ(firstItem, segments[segmentIndex].offset);
  }

  // Verify cluster in each segment and items in each cluster
  ASSERT_EQ(clustering.clusterItems.size(), 2 * 2 * 3);
  ASSERT_EQ(clustering.clusterRanges.size(), 2 * 3);
  ASSERT_EQ(clustering.clusterRangeSegments.size(), 3);
  for(size_t segmentIndex = 0; segmentIndex < 3; ++segmentIndex)
  {
    const uint32_t expectedFirstItem = segments[segmentIndex].offset;

    const nvcluster::Range& segment = clustering.clusterRangeSegments[segmentIndex];
    ASSERT_EQ(segment.count, 2);
    const nvcluster::Range& cluster0 = clustering.clusterRanges[segment.offset + 0];
    ASSERT_EQ(cluster0.count, 2);
    EXPECT_EQ(clustering.clusterItems[cluster0.offset + 0], expectedFirstItem + 0);
    EXPECT_EQ(clustering.clusterItems[cluster0.offset + 1], expectedFirstItem + 2);
    const nvcluster::Range& cluster1 = clustering.clusterRanges[segment.offset + 1];
    ASSERT_EQ(cluster1.count, 2);
    EXPECT_EQ(clustering.clusterItems[cluster1.offset + 0], segments[segmentIndex].offset + 1);
    EXPECT_EQ(clustering.clusterItems[cluster1.offset + 1], segments[segmentIndex].offset + 3);
  }
};

// Icosahedron data.
namespace icosahedron {
constexpr float              X         = .525731112119133606f;
constexpr float              Z         = .850650808352039932f;
static std::array<vec3, 12>  positions = {{{-X, 0.0, Z},
                                           {X, 0.0, Z},
                                           {-X, 0.0, -Z},
                                           {X, 0.0, -Z},
                                           {0.0, Z, X},
                                           {0.0, Z, -X},
                                           {0.0, -Z, X},
                                           {0.0, -Z, -X},
                                           {Z, X, 0.0},
                                           {-Z, X, 0.0},
                                           {Z, -X, 0.0},
                                           {-Z, -X, 0.0}}};
static std::array<uvec3, 20> triangles = {{{0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1},
                                           {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3},  {2, 7, 3},
                                           {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
                                           {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5},  {7, 2, 11}}};
}  // namespace icosahedron

// Type of a function to call when creating a triangle. Takes 3 positions as
// inputs.
using triangle_callback = std::function<void(vec3, vec3, vec3)>;

// Recursively subdivides a triangle on a sphere by a factor of 2^depth.
// Calls the callback function on each new triangle.
void subdivide(vec3 v0, vec3 v1, vec3 v2, int depth, triangle_callback& callback)
{
  if(depth == 0)
  {
    callback(v0, v1, v2);
  }
  else
  {
    vec3 v01 = normalize(add(v0, v1));
    vec3 v12 = normalize(add(v1, v2));
    vec3 v20 = normalize(add(v2, v0));
    subdivide(v0, v01, v20, depth - 1, callback);
    subdivide(v1, v12, v01, depth - 1, callback);
    subdivide(v2, v20, v12, depth - 1, callback);
    subdivide(v01, v12, v20, depth - 1, callback);
  }
}

// Makes an icosphere with 20 * (4^depth) triangles.
void makeIcosphere(int depth, triangle_callback& callback)
{
  for(size_t i = 0; i < icosahedron::triangles.size(); i++)
  {
    const vec3 v0 = icosahedron::positions[icosahedron::triangles[i][0]];
    const vec3 v1 = icosahedron::positions[icosahedron::triangles[i][1]];
    const vec3 v2 = icosahedron::positions[icosahedron::triangles[i][2]];
    subdivide(v0, v1, v2, depth, callback);
  }
}

struct GeometryMesh
{
  std::vector<uvec3> triangles;
  std::vector<vec3>  positions;
};

GeometryMesh makeIcosphere(int subdivision)
{
  std::unordered_map<vec3, uint32_t> vertexCache;
  std::vector<uvec3>                 triangles;
  // Our triangle callback function tries to place each of the vertices in the
  // vertex cache; each of the `it` iterators point to the existing value if
  // the vertex was already in the cache, or to a new value at the end of the
  // cache if it's a new vertex.
  triangle_callback callback = [&vertexCache, &triangles](vec3 v0, vec3 v1, vec3 v2) {
    auto [it0, new0] = vertexCache.try_emplace(v0, static_cast<uint32_t>(vertexCache.size()));
    auto [it1, new1] = vertexCache.try_emplace(v1, static_cast<uint32_t>(vertexCache.size()));
    auto [it2, new2] = vertexCache.try_emplace(v2, static_cast<uint32_t>(vertexCache.size()));
    triangles.push_back({it0->second, it1->second, it2->second});
  };
  makeIcosphere(subdivision, callback);
  std::vector<vec3> positions(vertexCache.size());
  for(const auto& [position, index] : vertexCache)
  {
    positions[index] = position;
  }
  return GeometryMesh{triangles, positions};
}

// Tests that the clusterizer respects minClusterSize and maxClusterSize.
TEST(Clusters, TestSizes)
{
  ScopedContext context;
  ASSERT_EQ(context.init(), nvcluster::Result::SUCCESS);

  const GeometryMesh mesh = makeIcosphere(3);

  std::vector<nvcluster::AABB> boundingBoxes(mesh.triangles.size());
  for(size_t i = 0; i < mesh.triangles.size(); i++)
  {
    boundingBoxes[i] = aabb(mesh.triangles[i], mesh.positions.data());
  }

  std::vector<vec3> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = centroid(boundingBoxes[i]);
  }

  nvcluster::SpatialElements spatialElements{.boundingBoxes = boundingBoxes.data(),
                                             .centroids     = reinterpret_cast<const float*>(centroids.data()),
                                             .elementCount  = static_cast<uint32_t>(boundingBoxes.size())};
  for(uint32_t sizeMax = 1; sizeMax < 10; ++sizeMax)
  {
    SCOPED_TRACE("Exact size: " + std::to_string(sizeMax));
    nvcluster::Input          input{.config =
                               {
                                            .minClusterSize    = sizeMax,
                                            .maxClusterSize    = sizeMax,
                                            .costUnderfill     = 0.0f,
                                            .costOverlap       = 0.0f,
                                            .preSplitThreshold = 0,
                               },
                                    .spatialElements = &spatialElements,
                                    .graph           = nullptr};
    nvcluster::ClusterStorage clustering;
    nvcluster::Result         result = nvcluster::generateClusters(context.context, input, clustering);
    ASSERT_EQ(result, nvcluster::Result::SUCCESS);

    // We requested that all clusters have `sizeMax` triangles. When
    // mesh.triangle.size() isn't a multiple of `sizeMax`, though, there'll be
    // one cluster with the remaining triangles. So the minimum cluster size
    // should be
    uint32_t expectedMin = uint32_t(mesh.triangles.size()) % sizeMax;
    if(expectedMin == 0)
      expectedMin = sizeMax;
    // And the largest cluster should have size `sizeMax`.
    // Let's test that's true:
    uint32_t trueMinSize = ~0U, trueMaxSize = 0;
    for(const nvcluster::Range& cluster : clustering.clusterRanges)
    {
      trueMinSize = std::min(trueMinSize, cluster.count);
      trueMaxSize = std::max(trueMaxSize, cluster.count);
    }

    EXPECT_EQ(expectedMin, trueMinSize);
    EXPECT_EQ(sizeMax, trueMaxSize);
  }
}

// Test that preSplitThreshold works.
// Our mesh here is an icosahedron with 1280 triangles, and we set the
// pre-split threshold to 1000, so the code should start by splitting into
// two sets of elements.
TEST(Clusters, TestPreSplit)
{
  ScopedContext context;
  ASSERT_EQ(context.init(), nvcluster::Result::SUCCESS);

  const uint32_t     preSplitThreshold = 1000;
  const GeometryMesh mesh              = makeIcosphere(3);
  // Make sure we'll pre-split at least once:
  EXPECT_GT(mesh.triangles.size(), preSplitThreshold);

  std::vector<nvcluster::AABB> boundingBoxes(mesh.triangles.size());
  for(size_t i = 0; i < mesh.triangles.size(); i++)
  {
    boundingBoxes[i] = aabb(mesh.triangles[i], mesh.positions.data());
  }

  std::vector<vec3> centroids(boundingBoxes.size());
  for(size_t i = 0; i < boundingBoxes.size(); i++)
  {
    centroids[i] = centroid(boundingBoxes[i]);
  }

  nvcluster::SpatialElements spatialElements{.boundingBoxes = boundingBoxes.data(),
                                             .centroids     = reinterpret_cast<const float*>(centroids.data()),
                                             .elementCount  = static_cast<uint32_t>(boundingBoxes.size())};
  nvcluster::Input           input{.config =
                             {
                                           .minClusterSize    = 100,
                                           .maxClusterSize    = 100,
                                           .costUnderfill     = 0.0f,
                                           .costOverlap       = 0.0f,
                                           .preSplitThreshold = preSplitThreshold,
                             },
                                   .spatialElements = &spatialElements,
                                   .graph           = nullptr};
  nvcluster::ClusterStorage  clustering;
  nvcluster::Result          result = nvcluster::generateClusters(context.context, input, clustering);
  ASSERT_EQ(result, nvcluster::Result::SUCCESS);

  // Validate all items exist and are unique
  std::set<uint32_t> uniqueItems(clustering.clusterItems.begin(), clustering.clusterItems.end());
  EXPECT_EQ(uniqueItems.size(), clustering.clusterItems.size());

  // Validate all items are covered by a range exactly once
  std::vector<uint32_t> itemClusterCounts(clustering.clusterItems.size(), 0);
  for(const nvcluster::Range& range : clustering.clusterRanges)
  {
    for(uint32_t i = range.offset; i < range.offset + range.count; i++)
    {
      itemClusterCounts[i]++;
    }
  }
  // Is every element in `itemClusterCounts` equal to 1?
  EXPECT_EQ(std::set(itemClusterCounts.begin(), itemClusterCounts.end()), std::set<uint32_t>{1});

  // Validate most sizes are the maximum
  std::unordered_map<uint32_t, uint32_t> clusterSizeCounts;  // cluster size -> number of clusters with that size
  for(const nvcluster::Range& range : clustering.clusterRanges)
  {
    clusterSizeCounts[range.count]++;
  }

  // This number of clusters had the maximum size:
  const uint32_t maxSizedCount = clusterSizeCounts[input.config.maxClusterSize];
  // This number of clusters were undersized:
  const uint32_t undersizedCount = static_cast<uint32_t>(clustering.clusterRanges.size()) - maxSizedCount;
  // There should be at most this number of undersized clusters.
  // That is, there are ceil(mesh.triangles.size() / preSplitThreshold)
  // sets after pre-splitting. Each set should generate at most 1 undersized cluster.
  const uint32_t expectedUndersized = uint32_t(mesh.triangles.size() + preSplitThreshold - 1) / preSplitThreshold;
  EXPECT_LE(undersizedCount, expectedUndersized);
  EXPECT_GE(maxSizedCount, clustering.clusterRanges.size() - expectedUndersized);
}
