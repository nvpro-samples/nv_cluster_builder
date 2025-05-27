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
#include <clusterizer.hpp>
#include <execution>


#include <nvcluster/nvcluster.h>
#include <ranges>
#include <span>
#include <stddef.h>
#include <vector>
#define PRINT_PERF 0

// Workaround for libc++ std::execution
#include <parallel_execution_libcxx.hpp>

// Scoped profiler for quick and coarse results
// https://stackoverflow.com/questions/31391914/timing-in-an-elegant-way-in-c
#if PRINT_PERF
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#endif

// Set NVCLUSTER_MULTITHREADED to 1 to use parallel processing, or set it to 0
// to use a single thread for all operations, which can be easier to debug.
#ifndef NVCLUSTER_MULTITHREADED
#define NVCLUSTER_MULTITHREADED 1
#endif

#if !NVCLUSTER_MULTITHREADED  // Single-threaded

#define NVLCLUSTER_DEFAULT_EXECUTION_POLICY std::execution::seq

// Start a for loop with the given item index variable name and number of items. The loop body should be followed by NVCLUSTER_PARALLEL_FOR_END.
// batchSize_ controls the batch size in the parallel version below.
#define NVCLUSTER_PARALLEL_FOR_BEGIN(itemIndexVariableName_, numItems_, batchSize_)                                    \
  for(size_t itemIndexVariableName_ = 0; itemIndexVariableName_ < numItems_; itemIndexVariableName_++)
// End a for loop started with NVCLUSTER_PARALLEL_FOR_BEGIN.
#define NVCLUSTER_PARALLEL_FOR_END
// Break out of a for loop started with NVCLUSTER_PARALLEL_FOR_BEGIN.
#define NVCLUSTER_PARALLEL_FOR_BREAK break

#else  // Multi-threaded

#define NVLCLUSTER_DEFAULT_EXECUTION_POLICY std::execution::par_unseq

// This is an iterator that counts upwards from an initial value.
// std::views::iota would almost work for this, but iota on MSVC 2019 doesn't
// support random access, which is necessary for parallelism.
template <class T>
struct iota_iterator
{
  using value_type = T;
  // [iterator.traits] in the C++ standard requires this to be a signed type.
  // We choose int64_t here, because it's conceivable someone could use
  // T == uint32_t and then iterate over more than 2^31 - 1 elements.
  using difference_type                                         = int64_t;
  using pointer                                                 = T*;
  using reference                                               = T&;
  using iterator_category                                       = std::random_access_iterator_tag;
  iota_iterator()                                               = default;
  iota_iterator(const iota_iterator& other) noexcept            = default;
  iota_iterator(iota_iterator&& other) noexcept                 = default;
  iota_iterator& operator=(const iota_iterator& other) noexcept = default;
  iota_iterator& operator=(iota_iterator&& other) noexcept      = default;
  iota_iterator(T i_)
      : i(i_)
  {
  }
  value_type     operator*() const { return i; }
  iota_iterator& operator++()
  {
    ++i;
    return *this;
  }
  iota_iterator operator++(int)
  {
    iota_iterator t(*this);
    ++*this;
    return t;
  }
  iota_iterator& operator--()
  {
    --i;
    return *this;
  }
  iota_iterator operator--(int)
  {
    iota_iterator t(*this);
    --*this;
    return t;
  }
  iota_iterator  operator+(difference_type d) const { return {static_cast<T>(static_cast<difference_type>(i) + d)}; }
  iota_iterator  operator-(difference_type d) const { return {static_cast<T>(static_cast<difference_type>(i) - d)}; }
  iota_iterator& operator+=(difference_type d)
  {
    i = static_cast<T>(static_cast<difference_type>(i) + d);
    return *this;
  }
  iota_iterator& operator-=(difference_type d)
  {
    i = static_cast<T>(static_cast<difference_type>(i) - d);
    return *this;
  }
  bool                 operator==(const iota_iterator& other) const { return i == other.i; }
  bool                 operator!=(const iota_iterator& other) const { return i != other.i; }
  bool                 operator<(const iota_iterator& other) const { return i < other.i; }
  bool                 operator<=(const iota_iterator& other) const { return i <= other.i; }
  bool                 operator>(const iota_iterator& other) const { return i > other.i; }
  bool                 operator>=(const iota_iterator& other) const { return i >= other.i; }
  difference_type      operator-(const iota_iterator& other) const { return i - other.i; }
  friend iota_iterator operator+(difference_type n, const iota_iterator& it) { return it + n; }
  T operator[](difference_type d) const { return static_cast<T>(static_cast<difference_type>(i) + d); }

private:
  T i = 0;
};

// Expresses the range from m_begin to m_end - 1.
template <class T>
struct iota_view
{
  using iterator = iota_iterator<T>;
  iota_view(T begin, T end)
      : m_begin(begin)
      , m_end(end)
  {
  }
  iterator begin() const { return {m_begin}; };
  iterator end() const { return {m_end}; };

private:
  T m_begin, m_end;
};

// Runs a function in parallel for each index from 0 to numItems - 1. Uses
// batches of size BATCHSIZE for reduced overhead and better autovectorization.
//
// BATCHSIZE will also be used as the threshold for when to switch from
// single-threaded to multi-threaded execution. For this reason, it should be set
// to a power of 2 around where multi - threaded is faster than single - threaded for
// the given function.Some examples are :
// * 8192 for trivial workloads(a * x + y)
// * 2048 for animation workloads(multiplication by a single matrix)
// * 512 for more computationally heavy workloads(run XTEA)
// * 1 for full parallelization(load an image)
//
// This is a simpler version of nvh::parallel_batches, which you can find in
// nvpro_core.
template <uint64_t BATCHSIZE = 512, typename F>
inline void parallel_batches(uint64_t numItems, F&& fn)
{
  // For small item counts, it's fastest to use a single thread and avoid the
  // overhead from invoking a parallel executor.
  if(numItems <= BATCHSIZE)
  {
    for(uint64_t i = 0; i < numItems; i++)
    {
      fn(i);
    }
  }
  else
  {
    // Unroll the loop into batches of size BATCHSIZE or less. This worker
    // function will be run in parallel using
    // std::for_each(std::execution::par_unseq).
    const uint64_t numBatches = (numItems + BATCHSIZE - 1) / BATCHSIZE;
    auto           worker     = [&numItems, &fn](const uint64_t batchIndex) {
      const uint64_t start          = BATCHSIZE * batchIndex;
      const uint64_t itemsRemaining = numItems - start;
      // This split is necessary to make MSVC try to auto-vectorize the first
      // loop, which will be the most common case when numItems is large.
      if(itemsRemaining >= BATCHSIZE)
      {
        // Exactly BATCHSIZE items to process
        for(uint64_t i = start; i < start + BATCHSIZE; i++)
        {
          fn(i);
        }
      }
      else
      {
        // Variable-length loop
        for(uint64_t i = start; i < numItems; i++)
        {
          fn(i);
        }
      }
    };

    // This runs the worker above for each batch from 0 to numBatches-1.
    iota_view<uint64_t> batches(0, numBatches);
    std::for_each(std::execution::par_unseq, batches.begin(), batches.end(), worker);
  }
}

// Start a parallel for loop with the given item index variable and number of items. The loop body should be followed by NVCLUSTER_PARALLEL_FOR_END.
#define NVCLUSTER_PARALLEL_FOR_BEGIN(itemIndexVariableName_, numItems_, batchSize_) parallel_batches<batchSize_>(numItems_, [&](uint64_t itemIndexVariableName_)
// End a parallel for loop started with NVCLUSTER_PARALLEL_FOR_BEGIN.
#define NVCLUSTER_PARALLEL_FOR_END )
// Break out of a parallel for loop started with NVCLUSTER_PARALLEL_FOR_BEGIN.
#define NVCLUSTER_PARALLEL_FOR_BREAK return

#endif  // NVCLUSTER_PARALLEL_FOR

class Stopwatch
{
#if PRINT_PERF
public:
  Stopwatch(std::string name)
      : m_name(std::move(name))
      , m_beg(std::chrono::high_resolution_clock::now())
  {
  }
  ~Stopwatch()
  {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_beg);
    std::cout << m_name << " : " << dur.count() << " ms\n";
  }

private:
  std::string                                                 m_name;
  std::chrono::time_point<std::chrono::high_resolution_clock> m_beg;
#else
public:
  template <class T>
  Stopwatch(T&&)
  {
  }
#endif
};

namespace nvcluster_internal {


inline nvcluster::AABB aabbCombine(const nvcluster::AABB& a, const nvcluster::AABB& b)
{
  nvcluster::AABB result;

  for(uint32_t i = 0; i < 3; i++)
  {
    result.bboxMin[i] = a.bboxMin[i] < b.bboxMin[i] ? a.bboxMin[i] : b.bboxMin[i];
    result.bboxMax[i] = a.bboxMax[i] > b.bboxMax[i] ? a.bboxMax[i] : b.bboxMax[i];
  }
  return result;
}

inline nvcluster::AABB aabbEmpty()
{
  nvcluster::AABB result;
  for(uint32_t i = 0; i < 3; i++)
  {
    result.bboxMin[i] = std::numeric_limits<float>::max();
    result.bboxMax[i] = std::numeric_limits<float>::lowest();
  }
  return result;
}

inline void aabbSize(const nvcluster::AABB& aabb, float* size)
{
  for(uint32_t i = 0; i < 3; i++)
  {
    size[i] = aabb.bboxMax[i] - aabb.bboxMin[i];
  }
}


inline nvcluster::AABB aabbIntersect(const nvcluster::AABB& a, const nvcluster::AABB& b)
{
  nvcluster::AABB result;

  for(uint32_t i = 0; i < 3; i++)
  {
    result.bboxMin[i] = std::max(a.bboxMin[i], b.bboxMin[i]);
    result.bboxMax[i] = std::min(a.bboxMax[i], b.bboxMax[i]);
  }

  // Compute a positive-sized bbox starting at bboxmin
  for(uint32_t i = 0; i < 3; i++)
  {
    float s           = result.bboxMax[i] - result.bboxMin[i];
    s                 = (s < 0.f) ? 0.f : s;
    result.bboxMax[i] = result.bboxMin[i] + s;
  }


  return result;
}


// Returns the ceiling of an integer division. Assumes positive values!
template <std::integral T>
T div_ceil(const T& a, const T& b)
{
  return (a + b - 1) / b;
}

// "Functor" to sort item indices based on their centroid coordinates along each
// axis
template <int Axis>
struct CentroidCompare
{
  CentroidCompare(const nvcluster::SpatialElements& bounds)
      : m_centroids(bounds.centroids)
  {
  }
  inline bool operator()(const uint32_t& itemA, const uint32_t& itemB) const
  {
    // For architectural meshes, centroids may form a grid, where many
    // coordinates would be equal. In these cases we still want ordering on the
    // remaining axes so a "composite key" is used.
    constexpr int A0 = Axis;
    constexpr int A1 = (Axis + 1) % 3;
    constexpr int A2 = (Axis + 2) % 3;
    const float*  c0 = m_centroids + 3 * itemA;
    const float*  c1 = m_centroids + 3 * itemB;
    return (c0[A0] < c1[A0]) || (c0[A0] == c1[A0] && c0[A1] < c1[A1])
           || (c0[A0] == c1[A0] && c0[A1] == c1[A1] && c0[A2] < c1[A2]);
  }
  const float* m_centroids = nullptr;
};

// Classic surface area heuristic cost
inline float sahCost(const nvcluster::AABB& aabb, const uint32_t& elementCount)
{
  // Direct calculation of the dimensions of the AABB
  float dx = aabb.bboxMax[0] - aabb.bboxMin[0];
  float dy = aabb.bboxMax[1] - aabb.bboxMin[1];
  float dz = aabb.bboxMax[2] - aabb.bboxMin[2];

  // Half area calculation (avoiding temporary arrays)
  float halfArea = dx * (dy + dz) + dy * dz;

  // Return the cost
  return halfArea * static_cast<float>(elementCount);
}

// Candidate split position within a node
struct Split
{
  int         axis     = -1;
  uint32_t    position = std::numeric_limits<uint32_t>::max();  // index of first item in the right node
  float       cost     = std::numeric_limits<float>::max();
  inline bool valid() { return axis != -1; }
  bool        operator<(const Split& other) const { return cost < other.cost; }
};

// An explicit min function to pass to std::reduce
inline Split minSplitCost(Split a, Split b)
{
  return std::min(a, b);
}

// Returns the sum of adjacency weights resulting from a split of the input node at the index elementIndexInNodeRange, where
// the element at elementIndexInNodeRange is included in the left part of the split.
// Weights are positive for connections to the right of the split and negative for
// connections to the left of the split. Later, the prefix sum of all items will then provide the
// *cut cost* in O(1) for any split position.
float sumAdjacencyWeightsAtSplit(const nvcluster::Graph* graph,  // adjacency graph storing the connections for each node
                                 std::span<const uint32_t> connectionIndicesInSortedElements,  // Flattened list of the connections for each node in the graph. The indices represent the connected elements in the sorted list of elements.
                                 const nvcluster::Range& node,  // Node to split, whose connections are defined by its offset and count in connectionIndicesInSortedElements
                                 uint32_t elementIndexInNodeRange,  // Index at which the node would be split, as a value between 0 and node.count
                                 uint32_t elementIndex  // Index of the element in the sorted list of elements corresponding to the split element
)
{
  float result = 0.0f;
  // Each element has a range of connections in the graph, so we fetch those connections to accumulate the weight contributions
  const nvcluster::Range& elementConnectionRange = graph->nodes[elementIndex];

  // Iterate over the connections of the split element, and accumulate the weights of the connections with positive sign for the right side of the split and negative sign for the left side.
  for(uint32_t connectionIndexInRange = elementConnectionRange.offset;
      connectionIndexInRange < elementConnectionRange.offset + elementConnectionRange.count; ++connectionIndexInRange)
  {
    // Find the index of the connected element within the connections of the node to split
    uint32_t connectedElementIndexInNodeRange = connectionIndicesInSortedElements[connectionIndexInRange] - node.offset;
    // Fetch the weight of the connection between the split element and the current connnection
    float connectingWeight = graph->connectionWeights[connectionIndexInRange];

    // Skip connections to items not in stored in the connections for the split node
    if(connectedElementIndexInNodeRange >= node.count)
      continue;

    // #OLD Add the weight for the earlier connection and subtract it again for
    // #OLD the later connection in the sorted list of items. This is applied as
    // #OLD a gather operation. Since the graph is bidirectional and we only
    // #OLD consider internal connections within the node, itemIndex will be both
    // #OLD during iteration.
    // Add the weight of the connection, but with opposite sign depending on whether the current element is on the left or right side of the split
    result += (elementIndexInNodeRange < connectedElementIndexInNodeRange) ? connectingWeight : -connectingWeight;
  }
  return result;
}

// Compute an aggregate cost of splitting a node at the location splitPositionFromLeft within its element range, balancing multiple factors
template <bool AlignBoth>
inline float splitCost(const nvcluster::Input& input,
                       std::span<const float>  splitWeights,  // Cost of the graph cut for each element in the node
                       const nvcluster::AABB&  leftAabb,  // Bounding box of the elements on the left side of the split
                       const nvcluster::AABB& rightAabb,  // Bounding box of the elements on the right side of the split
                       uint32_t               nodeSize,   // Number of elements in the node
                       uint32_t splitPositionFromLeft  // Location of the split, as the index of the first element in the right side of the split
)
{
  // Make leaves adhere to the min and max leaf size rule
  uint32_t acceptableRemainder    = input.config.maxClusterSize - input.config.minClusterSize;
  uint32_t splitPositionFromRight = nodeSize - splitPositionFromLeft;
  bool     leftAlign              = splitPositionFromLeft % input.config.minClusterSize
                   <= (splitPositionFromLeft / input.config.minClusterSize) * acceptableRemainder;
  bool rightAlign = splitPositionFromRight % input.config.minClusterSize
                    <= (splitPositionFromRight / input.config.minClusterSize) * acceptableRemainder;

  float cost = std::numeric_limits<float>::max();
  // TODO: not always leftAlign if !AlignBoth?
  if(leftAlign && (!AlignBoth || rightAlign))
  {
    float leftCost  = sahCost(leftAabb, splitPositionFromLeft);
    float rightCost = sahCost(rightAabb, splitPositionFromRight);
    cost            = leftCost + rightCost;

#if 1
    // Increase cost for under-filled nodes (want more leaves
    // with maxClusterSize than minClusterSize)
    uint32_t leftItemCount =
        input.config.maxClusterSize * div_ceil(splitPositionFromLeft, input.config.maxClusterSize) - splitPositionFromLeft;
    uint32_t rightItemCount =
        input.config.maxClusterSize * div_ceil(splitPositionFromRight, input.config.maxClusterSize) - splitPositionFromRight;
    cost += sahCost(leftAabb, leftItemCount) * input.config.costUnderfill;
    cost += sahCost(rightAabb, rightItemCount) * input.config.costUnderfill;
#endif

#if 1
    // Increase cost for lef/right bounding box overlap
    nvcluster::AABB intersection = aabbIntersect(leftAabb, rightAabb);
    cost += sahCost(intersection, nodeSize) * input.config.costOverlap;
#endif

#if 1
    // Increase cost for weighted edge connections outside the
    // node and crossing the split plane.
    if(input.graph)
    {
      // Convert "ratio cut" values to SAH relative costs. Costs are all based on
      // SAH, a measure of trace cost for the bounding surface area. Shoehorning a
      // min-cut cost into the mix is problematic because the same metric needs to
      // work at various scales of node sizes and item counts. SAH scales with the
      // the number of items. The cut cost likely scales with the square root of the
      // number of items, assuming they form surfaces in the 3D space, but ratio cut
      // also normalizes by the item count. This attempts to undo that. A good way
      // to verify is to plot maximum costs against varying node sizes.
      float normalizeCutWeights = float(nodeSize * nodeSize);

      // "Ratio cut" - divide by the number of items in each
      // node to avoid degenerate cuts of just the first/last
      // items.
      float cutCost      = splitWeights[splitPositionFromLeft];
      float ratioCutCost = cutCost / float(splitPositionFromLeft) + cutCost / float(splitPositionFromRight);
      cost += ratioCutCost * normalizeCutWeights;
    }
#endif
  }
  return cost;
}

/**
 * @brief Returns the minimum cost split position for one axis. Takes some
 * temporary arrays as arguments to reuse allocations between passes.
 * @tparam Axis consider splits along (x, y, z), (0, 1, 2)
 * @tparam AlignBoth enforce Input::Config::minClusterSize and
 * Input::Config::maxClusterSize from both ends of the node or just one
 * @tparam ExecutionPolicy Optionally parallelize internally or externally
 *
 * Note: std algorithm and structures of arrays are used frequently due to the
 * convenience of std execution to parallelize steps.
 */
template <int Axis, bool AlignBoth>
nvcluster::Result findBestSplit(const nvcluster::Input& input,  // Input elements and adjacency graph
                                const nvcluster::Range& node,   // Range of elements in representing the node to split
                                std::span<const uint32_t> sortedElementIndicesPerAxis,  // List of element indices for the node, sorted along the chosen axis
                                std::span<nvcluster::AABB> nodeLeftBoxes,  // Output bounding boxes of the elements on the left side of the chosen split
                                std::span<nvcluster::AABB> nodeRightBoxes,  // Output bounding boxes of the elements on the right side of the chosen split
                                std::span<const uint32_t> connectionIndicesInSortedElements,  // List of the elements connected node, where each is an index in sortedElementIndicesPerAxis
                                std::span<float> deltaWeights,  // Output list of adjacency weights modifications for each element in the node due to the split
                                std::span<float> splitWeights,  // Output list of cut costs for each split position
                                Split&           split          // Output split information
)
{
  const nvcluster::SpatialElements& inputElements = *input.spatialElements;

  // Populate the left to right bounding boxes, growing with each item.
  // std::transform_exclusive_scan is a prefix sum, e.g. {1, 1, 1} becomes {0,
  // 1, 2} but here the plus operator unions bounding boxes.
  std::transform_exclusive_scan(
      NVLCLUSTER_DEFAULT_EXECUTION_POLICY, sortedElementIndicesPerAxis.begin(), sortedElementIndicesPerAxis.end(),
      nodeLeftBoxes.begin(), aabbEmpty(),
      [](const nvcluster::AABB& a, const nvcluster::AABB& b) {
        //return aabbCombine(a, b);
        nvcluster::AABB result;

        for(uint32_t i = 0; i < 3; i++)
        {
          result.bboxMin[i] = a.bboxMin[i] < b.bboxMin[i] ? a.bboxMin[i] : b.bboxMin[i];
          result.bboxMax[i] = a.bboxMax[i] > b.bboxMax[i] ? a.bboxMax[i] : b.bboxMax[i];
        }
        return result;
      },
      [&bboxes = inputElements.boundingBoxes](const uint32_t& i) { return bboxes[i]; });


  // Populate the right to left bounding boxes, growing with each item.
  // std::transform_inclusive_scan is a prefix post-sum, e.g. {1, 1, 1} becomes
  // {1, 2, 3}. Note the use of rbegin()/rend() to reverse iterate.
  std::transform_inclusive_scan(
      NVLCLUSTER_DEFAULT_EXECUTION_POLICY, sortedElementIndicesPerAxis.rbegin(), sortedElementIndicesPerAxis.rend(),
      nodeRightBoxes.rbegin(),
      [](const nvcluster::AABB& a, const nvcluster::AABB& b) {
        //return aabbCombine(a, b);
        nvcluster::AABB result;

        for(uint32_t i = 0; i < 3; i++)
        {
          result.bboxMin[i] = a.bboxMin[i] < b.bboxMin[i] ? a.bboxMin[i] : b.bboxMin[i];
          result.bboxMax[i] = a.bboxMax[i] > b.bboxMax[i] ? a.bboxMax[i] : b.bboxMax[i];
        }
        return result;
      },
      [&bboxes = inputElements.boundingBoxes](const uint32_t& i) { return bboxes[i]; });

  // Build cumulative adjacency weights if adjacency was provided
  if(input.graph)
  {
    // Build rising and falling edge weights for connections between items
    // within the node along this axis
    NVCLUSTER_PARALLEL_FOR_BEGIN(parallelItemIndex, node.count, 512)
    {
      const uint32_t elementIndex            = sortedElementIndicesPerAxis[parallelItemIndex];
      uint32_t       elementIndexInNodeRange = uint32_t(parallelItemIndex);
      deltaWeights[parallelItemIndex] = sumAdjacencyWeightsAtSplit(input.graph, connectionIndicesInSortedElements, node,
                                                                   elementIndexInNodeRange, elementIndex);
    }
    NVCLUSTER_PARALLEL_FOR_END;

    // Prefix sum scan the delta weights to find the total cut cost at each
    // position
    std::exclusive_scan(NVLCLUSTER_DEFAULT_EXECUTION_POLICY, deltaWeights.begin(), deltaWeights.end(), splitWeights.begin(), 0.0f);

#if !defined(NDEBUG)
    // Smoke check for weight overflow
    for(size_t i = 0; i < splitWeights.size(); i++)
    {
      if(splitWeights[i] >= 1e12)
      {
        return nvcluster::Result::ERROR_WEIGHT_OVERFLOW;
      }
    }
#endif
  }

  // Replace the best split by computing costs at each candidate and reducing.
  // Inputs are the left and right bounding box, skipping index 0 (meaning the
  // left child would be empty).
  // std::transform_reduce combines pairs of results until there is one
  // remaining. It is initialized with the current best split and min_split() is
  // given to keep the split candidate with minimum cost, possibly with parallel
  // execution.
  split = std::transform_reduce(
      NVLCLUSTER_DEFAULT_EXECUTION_POLICY,             // possibly parallel
      nodeLeftBoxes.begin() + 1, nodeLeftBoxes.end(),  // input left bounding box
      nodeRightBoxes.begin() + 1,                      // input right bounding box
      split,                                           // current best split
      minSplitCost,                                    // reduce by taking the minimum cost
      [&splitWeights, &input, &nodeLeftBoxes](const nvcluster::AABB& leftAabb, const nvcluster::AABB& rightAabb) -> Split {
        uint32_t splitPosition = (uint32_t)(&leftAabb - &nodeLeftBoxes.front());  // DANGER: pointer arithmetic to get index :(
        float cost = splitCost<AlignBoth>(input, splitWeights, leftAabb, rightAabb, uint32_t(nodeLeftBoxes.size()), splitPosition);
        return Split{.axis = Axis, .position = splitPosition, .cost = cost};
      });

  return nvcluster::Result::SUCCESS;
}

// For each element in the sorted list of elements along an axis, identify the indices of its connected elements in the sorted list of elements.
// This is done by linearly searching through the list of secondary connections for each element, and scattering the index of the current node in the sorted list
// in the lists of secondary connections. At the end of the function connectionIndicesInSortedElements will contain the indices of the connected elements in the sorted list of elements.
void buildAdjacencyInSortedList(const nvcluster::Input& input,                   // Input elements and adjacency graph
                                std::span<const uint32_t> sortedElementIndices,  // Sorted list of element indices along an axis
                                std::span<uint32_t> connectionIndicesInSortedElements,  // Output list of indices of connected elements in the sorted list of elements
                                std::vector<uint32_t>& backMapping)
{

  NVCLUSTER_PARALLEL_FOR_BEGIN(parallelItemIndex, sortedElementIndices.size(), 4096)
  {
    backMapping[sortedElementIndices[parallelItemIndex]] = uint32_t(parallelItemIndex);
  }
  NVCLUSTER_PARALLEL_FOR_END;

  NVCLUSTER_PARALLEL_FOR_BEGIN(parallelItemIndex, input.graph->connectionCount, 4096)
  {
    connectionIndicesInSortedElements[parallelItemIndex] = backMapping[input.graph->connectionTargets[parallelItemIndex]];
  }
  NVCLUSTER_PARALLEL_FOR_END;

#if 0

  NVCLUSTER_PARALLEL_FOR_BEGIN(parallelItemIndex, sortedElementIndices.size())
  {
    // Fetch an element index from the sorted list
    uint32_t elementIndex = sortedElementIndices[parallelItemIndex];

    // Scatter sortedItemIndex into the connected items lists of each connected
    // item. I.e. use the outgoing connection to find and write the returning
    // connection in other item's connections lists.

    // Fetch the range of graph connections representing the connections of the current element
    const nvcluster::Range& connectionsRange = input.graph->nodes[elementIndex];

    // Fetch the list of connections for the current element
    const uint32_t* connections = input.graph->connectionTargets + connectionsRange.offset;

    // For each connection of the current element, findd whic
    for(uint32_t idx = 0; idx < connectionsRange.count; idx++)
    {
      // Fetch the index of an element connected to the current element
      uint32_t connectedElementIndex = connections[idx];

      // Fetch the range of graph connections representing the connections of the connected element
      const nvcluster::Range& secondaryConnectionsRange = input.graph->nodes[connectedElementIndex];

      // Fetch the list of connections for the connected element
      const uint32_t* secondaryConnections = input.graph->connectionTargets + secondaryConnectionsRange.offset;

      // Extract the sorted list of connected items for the connected element
      uint32_t* secondaryConnectionIndicesInSortedElements =
          connectionIndicesInSortedElements.data() + secondaryConnectionsRange.offset;

      // Linear search to find the connection back to the current node. Once found, store the index of the current node in the sorted list of elements
      // in the list of sorted connected items for the secondary node.
      for(uint32_t i = 0; i < secondaryConnectionsRange.count; ++i)
      {
        if(secondaryConnections[i] == elementIndex)
        {
          secondaryConnectionIndicesInSortedElements[i] = uint32_t(parallelItemIndex);
          break;
        }
      }
    }
  }
  NVCLUSTER_PARALLEL_FOR_END;
#endif
}


// Splits the per-axis sorted element lists at the chosen split position along the given axis,
// so that the elements on the left side of the split are moved to the front of each list, while preserving the element ordering within each partition.
inline void partitionAtSplit(std::span<uint32_t> sortedElementIndicesPerAxis[3], int axis, uint32_t splitPosition, std::span<uint8_t> partitionSides)
{

  // Mark each element as being on the left or right side of the split
  // This is trivially done by traversing the list of elements sorted along the chosen axis, and marking their side depending
  // on whether their index in the list is below or above the split position.
  // While this approach uses more memory than checking the centroid
  for(size_t itemIndex = 0; itemIndex < sortedElementIndicesPerAxis[axis].size(); itemIndex++)
  {
    uint32_t i        = sortedElementIndicesPerAxis[axis][itemIndex];
    partitionSides[i] = itemIndex < splitPosition ? 1 : 0;
  }

  if(axis != 0)
  {
    std::stable_partition(NVLCLUSTER_DEFAULT_EXECUTION_POLICY, sortedElementIndicesPerAxis[0].begin(),
                          sortedElementIndicesPerAxis[0].end(),
                          [&partitionSides](const uint32_t& i) { return partitionSides[i]; });
  }
  if(axis != 1)
  {
    std::stable_partition(NVLCLUSTER_DEFAULT_EXECUTION_POLICY, sortedElementIndicesPerAxis[1].begin(),
                          sortedElementIndicesPerAxis[1].end(),
                          [&partitionSides](const uint32_t& i) { return partitionSides[i]; });
  }
  if(axis != 2)
  {
    std::stable_partition(NVLCLUSTER_DEFAULT_EXECUTION_POLICY, sortedElementIndicesPerAxis[2].begin(),
                          sortedElementIndicesPerAxis[2].end(),
                          [&partitionSides](const uint32_t& i) { return partitionSides[i]; });
  }
}

// Take a node defined by its sorted lists of element indices and recursively splits it along its longest axis until
// the number of elements in each node is less than or equal to maxElementsPerNode.
static void splitAtMedianUntil(const nvcluster::SpatialElements& spatialElements,  // Original definition of the input spatial elements
                               std::vector<uint8_t>& partitionSides,  // Partition identifier (left = 1, right = 0) for each element, used to partition the sorted element lists
                               std::span<uint32_t> nodeSortedElementIndicesPerAxis[3],  // Sorted element indices along each axis for the current node
                               size_t maxElementsPerNode,  // Maximum number of elements allowed per node, used to stop the recursion
                               uint32_t nodeStartIndex,  // Starting index of the node in the complete sorted lists of elements
                               std::vector<nvcluster::Range>& perNodeElementRanges  // Output ranges of the nodes (in the sorted element lists) created by the recursive split
)
{
  uint32_t nodeCount = uint32_t(nodeSortedElementIndicesPerAxis[0].size());
  // If the current node is smaller than the maximum allowed element count, return its range
  if(nodeCount < maxElementsPerNode)
  {
    perNodeElementRanges.push_back(nvcluster::Range{nodeStartIndex, nodeCount});
    return;
  }

  // Compute the AABB of the centroids of the elements referenced by the node. Since the elements are sorted along each axis,
  // the bounds on each coordinate are trivial to compute using the first and last items in the sorted list for that axis.
  // This does not provide the exact AABB for the node (ideallly we should combine the AABBs of each element), but this centroid-based
  // approximation is trivial and sufficient for the purpose of pre-splitting large inputs.
  nvcluster::AABB aabb{{spatialElements.centroids[3 * nodeSortedElementIndicesPerAxis[0].front() + 0],
                        spatialElements.centroids[3 * nodeSortedElementIndicesPerAxis[1].front() + 1],
                        spatialElements.centroids[3 * nodeSortedElementIndicesPerAxis[2].front() + 2]},
                       {spatialElements.centroids[3 * nodeSortedElementIndicesPerAxis[0].back() + 0],
                        spatialElements.centroids[3 * nodeSortedElementIndicesPerAxis[1].back() + 1],
                        spatialElements.centroids[3 * nodeSortedElementIndicesPerAxis[2].back() + 2]}};

  // Deduce the splitting axis from the longest side of the AABB
  float size[3];
  aabbSize(aabb, size);
  int axis = size[0] > size[1] && size[0] > size[2] ? 0 : (size[1] > size[2] ? 1 : 2);

  // Split the sorted elements vectors at the median, preserving the order along each axis
  uint32_t splitPosition = nodeCount / 2;
  partitionAtSplit(nodeSortedElementIndicesPerAxis, axis, splitPosition, partitionSides);

  // Extract the left and right halves of the sorted element lists
  std::span<uint32_t> left[]{
      nodeSortedElementIndicesPerAxis[0].subspan(0, splitPosition),
      nodeSortedElementIndicesPerAxis[1].subspan(0, splitPosition),
      nodeSortedElementIndicesPerAxis[2].subspan(0, splitPosition),
  };
  std::span<uint32_t> right[]{
      nodeSortedElementIndicesPerAxis[0].subspan(splitPosition),
      nodeSortedElementIndicesPerAxis[1].subspan(splitPosition),
      nodeSortedElementIndicesPerAxis[2].subspan(splitPosition),
  };
  // Continue the split recursively on the left and right halves
  splitAtMedianUntil(spatialElements, partitionSides, left, maxElementsPerNode, nodeStartIndex, perNodeElementRanges);
  splitAtMedianUntil(spatialElements, partitionSides, right, maxElementsPerNode, nodeStartIndex + splitPosition, perNodeElementRanges);
}

// Temporary storage for the split node function
struct SplitNodeTemporaries
{
  // Identification of the side on which each element lies when splitting a node (left = 1, right = 0)
  std::span<uint8_t> partitionSides;
  // Bounding boxes of the left children of the currently processed node
  std::span<nvcluster::AABB> leftChildrenBoxes;
  // Bounding boxes of the right children of the currently processed node
  std::span<nvcluster::AABB> rightChildrenBoxes;

  std::span<float>          deltaWeights;
  std::span<float>          splitWeights;
  std::span<const uint32_t> connectionIndicesInSortedElements[3];
  std::span<uint32_t>       sortedElementIndicesPerAxis[3];
};

// Find the lowest cost split on any axis, perform the split (partition
// sortedItems, maintaining order) and write two child nodes to the output.
nvcluster::Result splitNode(const nvcluster::Input& input,        // Input elements and adjacency graph
                            SplitNodeTemporaries&   temporaries,  // Temporary storage for the split node function
                            const nvcluster::Range& node,         // Range of elements in the node to split
                            std::span<nvcluster::Range> outNodes,  // Output nodes storage where child nodes will be added
                            std::atomic<size_t>& outNodesAlloc     // Current number of nodes in the output
)
{
  // Slice structure of arrays by the current node's range
  std::span<nvcluster::AABB> nodeLeftBoxes  = temporaries.leftChildrenBoxes.subspan(node.offset, node.count);
  std::span<nvcluster::AABB> nodeRightBoxes = temporaries.rightChildrenBoxes.subspan(node.offset, node.count);
  std::span<float>           nodeDeltaWeights =
      input.graph ? temporaries.deltaWeights.subspan(node.offset, node.count) : std::span<float>{};
  std::span<float> nodeSplitWeights =
      input.graph ? temporaries.splitWeights.subspan(node.offset, node.count) : std::span<float>{};
  std::span<uint32_t> sortedElementIndicesPerAxis[]{
      temporaries.sortedElementIndicesPerAxis[0].subspan(node.offset, node.count),
      temporaries.sortedElementIndicesPerAxis[1].subspan(node.offset, node.count),
      temporaries.sortedElementIndicesPerAxis[2].subspan(node.offset, node.count),
  };

  // Find a split candidate by looking for the best split along each axis
  Split             split;
  nvcluster::Result splitResult{};
  splitResult = findBestSplit<0, true>(input, node, sortedElementIndicesPerAxis[0], nodeLeftBoxes, nodeRightBoxes,
                                       temporaries.connectionIndicesInSortedElements[0], nodeDeltaWeights, nodeSplitWeights, split);
  if(splitResult != nvcluster::Result::SUCCESS)
  {
    return splitResult;
  }

  splitResult = findBestSplit<1, true>(input, node, sortedElementIndicesPerAxis[1], nodeLeftBoxes, nodeRightBoxes,
                                       temporaries.connectionIndicesInSortedElements[1], nodeDeltaWeights, nodeSplitWeights, split);
  if(splitResult != nvcluster::Result::SUCCESS)
  {
    return splitResult;
  }

  splitResult = findBestSplit<2, true>(input, node, sortedElementIndicesPerAxis[2], nodeLeftBoxes, nodeRightBoxes,
                                       temporaries.connectionIndicesInSortedElements[2], nodeDeltaWeights, nodeSplitWeights, split);
  if(splitResult != nvcluster::Result::SUCCESS)
  {
    return splitResult;
  }

  // Item count is too small to make clusters between min/max size. Fall
  // back to aligning splits from the left so there should be just one
  // cluster outside the range. This should be rare.
  if(!split.valid())
  {

    splitResult = findBestSplit<0, false>(input, node, sortedElementIndicesPerAxis[0], nodeLeftBoxes, nodeRightBoxes,
                                          temporaries.connectionIndicesInSortedElements[0], nodeDeltaWeights,
                                          nodeSplitWeights, split);
    if(splitResult != nvcluster::Result::SUCCESS)
    {
      return splitResult;
    }

    splitResult = findBestSplit<1, false>(input, node, sortedElementIndicesPerAxis[1], nodeLeftBoxes, nodeRightBoxes,
                                          temporaries.connectionIndicesInSortedElements[1], nodeDeltaWeights,
                                          nodeSplitWeights, split);
    if(splitResult != nvcluster::Result::SUCCESS)
    {
      return splitResult;
    }

    splitResult = findBestSplit<2, false>(input, node, sortedElementIndicesPerAxis[2], nodeLeftBoxes, nodeRightBoxes,
                                          temporaries.connectionIndicesInSortedElements[2], nodeDeltaWeights,
                                          nodeSplitWeights, split);
    if(splitResult != nvcluster::Result::SUCCESS)
    {
      return splitResult;
    }
  }


  if(split.position <= 0 || split.position >= sortedElementIndicesPerAxis[0].size() || split.position >= node.count)
  {
    return nvcluster::Result::ERROR_INTERNAL;
  }

  // Split the node at the chosen axis and position
  partitionAtSplit(sortedElementIndicesPerAxis, split.axis, split.position, temporaries.partitionSides);

  // Create child nodes. The atomic outNodesAlloc is used to allocate output
  // nodes when external parallelization is used.
  size_t outNodesOffset        = outNodesAlloc.fetch_add(2);
  outNodes[outNodesOffset + 0] = {node.offset, split.position};
  outNodes[outNodesOffset + 1] = {node.offset + split.position, node.count - split.position};
  return nvcluster::Result::SUCCESS;
};


// Starting from a set of spatial items defined by their bounding boxes and centroids, and an optional adjacency graph describing the connectivity between them, this function groups those items
// into clusters using a kD-Tree-based decomposition
nvcluster::Result clusterize(const nvcluster::Input& input,  // Input elements and adjacency graph
                             //nvcluster::Requirements& outputSizes,  // Output sizes of the resulting set of clusters
                             nvcluster::Output& clusters  // Output set of clusters
)
{
  Stopwatch swClusterize("clusterize");

  if(input.config.minClusterSize <= 0 || input.config.maxClusterSize <= 0 || input.config.minClusterSize > input.config.maxClusterSize)
  {
    return nvcluster::Result::ERROR_INVALID_CONFIG;
  }
  if(!input.spatialElements || input.spatialElements->elementCount != clusters.clusteredElementIndexCount)
  {
    return nvcluster::Result::ERROR_INVALID_BOUNDS;
  }

  const nvcluster::SpatialElements& spatialElements = *input.spatialElements;
  // Early out if there are no elements to cluster. This can happen using the segmented clustering
  // FIXME: why is that?
  if(spatialElements.elementCount == 0)
  {
    return nvcluster::Result::SUCCESS;
  }

  nvcluster::Requirements outputSizes{};
  // Initialize the output sizes: at the beginning no cluster has been created
  outputSizes.maxClusterCount = 0;
  // We already know the list of all the elements referenced by the clusters has the same size
  // as the list of input spatial elements since each of them will be referenced by exactly one cluster
  outputSizes.maxClusteredElementCount = spatialElements.elementCount;


  // Temporary data
  // Used to mark the elements as belonging to the left (1) or right (0) side of the split
  std::vector<uint8_t> partitionSides(spatialElements.elementCount);
  // Bouding boxes of the left children of the currently processed node
  std::vector<nvcluster::AABB> leftChildrenBoxes(spatialElements.elementCount);
  // Bouding boxes of the right children of the currently processed node
  std::vector<nvcluster::AABB> rightChildrenBoxes(spatialElements.elementCount);
  // Difference of adjacency weights for each element due to a split at an element
  std::vector<float> deltaWeights;
  // Adjacency weights resulting from a split at an element
  std::vector<float> splitWeights;
  // For each element, stores the index of its connected elements in the sorted list of elements
  std::vector<uint32_t> connectionIndicesInSortedElements[3];

  // The kD-tree will split the array of spatial elements recursively along the X, Y and Z axes. In order to
  // run the splitting algorithm we first need to sort the input spatial elements along each of those axis,
  // so the splitting will only require a simple partitioning.
  // In order to save memory we will use the storage area for the resulting clustered element indices as a temporary storage
  // for the indices of elements sorted along the X axis.
  std::vector<uint32_t> sortedY(clusters.clusteredElementIndexCount);
  std::vector<uint32_t> sortedZ(clusters.clusteredElementIndexCount);
  // Initialize the array of per-axis element indices so each entry references one element
  for(uint32_t i = 0; i < uint32_t(clusters.clusteredElementIndexCount); i++)
  {
    clusters.clusteredElementIndices[i] = i;
    sortedY[i]                          = i;
    sortedZ[i]                          = i;
  }

  // Sort the elements along the X, Y and Z axes based on the location of their centroids.
  // As mentioned above the storage area for the output clustered element indices is used as a temporary storage for the sorted indices along the X axis
  std::span<uint32_t> sortedElementIndicesPerAxis[3]{
      std::span<uint32_t>(clusters.clusteredElementIndices, clusters.clusteredElementIndexCount), sortedY, sortedZ};
  {
    Stopwatch swSort("sort");
    std::sort(NVLCLUSTER_DEFAULT_EXECUTION_POLICY, sortedElementIndicesPerAxis[0].begin(),
              sortedElementIndicesPerAxis[0].end(), CentroidCompare<0>(spatialElements));
    std::sort(NVLCLUSTER_DEFAULT_EXECUTION_POLICY, sortedElementIndicesPerAxis[1].begin(),
              sortedElementIndicesPerAxis[1].end(), CentroidCompare<1>(spatialElements));
    std::sort(NVLCLUSTER_DEFAULT_EXECUTION_POLICY, sortedElementIndicesPerAxis[2].begin(),
              sortedElementIndicesPerAxis[2].end(), CentroidCompare<2>(spatialElements));
  }

  // Temporary data for connectivity costs
  if(input.graph)
  {
    if(input.graph->nodeCount != spatialElements.elementCount)
    {
      return nvcluster::Result::ERROR_INVALID_GRAPH;
    }
    deltaWeights.resize(spatialElements.elementCount);
    splitWeights.resize(spatialElements.elementCount);

    // Maintain graph adjacency within in each sortedItems array to avoid
    // expensive searches when computing the split costs.
    for(uint32_t axis = 0; axis < 3; ++axis)
    {
      connectionIndicesInSortedElements[axis].resize(input.graph->connectionCount);
    }
  }

  // BVH style recursive bisection. Split nodes recursively until they have the
  // desired number of items. Unlike a BVH build, the hierarchy is not stored
  // and leaf nodes are immediately emitted as clusters. The current list of
  // nodes is double buffered nodes for each level.
  std::vector<nvcluster::Range> perNodeElementIndexRanges[2];
  perNodeElementIndexRanges[0].reserve((2 * spatialElements.elementCount) / input.config.maxClusterSize);
  perNodeElementIndexRanges[1].reserve((2 * spatialElements.elementCount) / input.config.maxClusterSize);

  uint32_t              currentNodeRangeList = 0;
  uint32_t              nextNodeRangeList    = 1;
  std::atomic<uint32_t> underflowClusters    = 0;
  uint32_t sanitizedPreSplitThreshold = std::max(input.config.preSplitThreshold, input.config.maxClusterSize * 2);
  if(input.config.preSplitThreshold == 0 || spatialElements.elementCount < sanitizedPreSplitThreshold)
  {
    perNodeElementIndexRanges[currentNodeRangeList].push_back({0, spatialElements.elementCount});
  }
  else
  {
    Stopwatch swSplitMedian("splitMedian");
    // Performance optimization. If there are more than preSplitThreshold items in the root node,
    // create child nodes by performing simple median splits on the input element lists until each node contains a maximum of sanitizedPreSplitThreshold elements.
    // This reduces the overally computational cost of the BVH build by applying the more accurate (and costly) node splitting algorithm only on smaller nodes.
    splitAtMedianUntil(*input.spatialElements, partitionSides, sortedElementIndicesPerAxis, sanitizedPreSplitThreshold,
                       0, perNodeElementIndexRanges[currentNodeRangeList]);
  }

  // Bisect nodes until no internal nodes are left. The nodes array is double buffered
  // for simplicity - could also be a thread safe queue. Leaf nodes are removed
  // and written to the output.
  std::vector<uint32_t> backmapping[3];

  for(uint32_t i = 0; i < 3; i++)
  {
    backmapping[i].resize(sortedElementIndicesPerAxis->size());
  }
  while(!perNodeElementIndexRanges[currentNodeRangeList].empty())
  {
    // If connectivity is used, convert connected indices to sorted item indices
    // to get constant lookup time for items within nodes.
    if(input.graph != nullptr)
    {
      Stopwatch sw("buikdadj");
      for(uint32_t axis = 0; axis < 3; ++axis)
      //NVCLUSTER_PARALLEL_FOR_BEGIN_FORCE(axis, 3)
      {
        //std::ranges::fill(connectionIndicesInSortedElements[axis], 0xffffffffu);
        buildAdjacencyInSortedList(input, sortedElementIndicesPerAxis[axis], connectionIndicesInSortedElements[axis],
                                   backmapping[axis]);
      }
      //NVCLUSTER_PARALLEL_FOR_END;
    }

    std::atomic<size_t> nodesBAlloc = 0;
    perNodeElementIndexRanges[nextNodeRangeList].resize(perNodeElementIndexRanges[nextNodeRangeList].capacity());  // conservative over-allocation

    SplitNodeTemporaries intermediates{
        .partitionSides     = partitionSides,
        .leftChildrenBoxes  = leftChildrenBoxes,
        .rightChildrenBoxes = rightChildrenBoxes,
        .deltaWeights       = deltaWeights,
        .splitWeights       = splitWeights,
        .connectionIndicesInSortedElements = {connectionIndicesInSortedElements[0], connectionIndicesInSortedElements[1],
                                              connectionIndicesInSortedElements[2]},
        .sortedElementIndicesPerAxis = {sortedElementIndicesPerAxis[0], sortedElementIndicesPerAxis[1],
                                        sortedElementIndicesPerAxis[2]},
    };

    // Process all nodes in the current level.
    {
      std::atomic<nvcluster::Result> result = nvcluster::Result::SUCCESS;
      NVCLUSTER_PARALLEL_FOR_BEGIN(parallelItemIndex, perNodeElementIndexRanges[currentNodeRangeList].size(), 1)
      {
        if(result != nvcluster::Result::SUCCESS)
        {
          NVCLUSTER_PARALLEL_FOR_BREAK;
        }
        const nvcluster::Range& node = perNodeElementIndexRanges[currentNodeRangeList][parallelItemIndex];
        // Emit leaf nodes and split internal nodes
        if(node.count <= input.config.maxClusterSize)
        {
          if(node.count == 0)
          {
            result = nvcluster::Result::ERROR_INTERNAL;
            NVCLUSTER_PARALLEL_FOR_BREAK;
          }
          clusters.clusterRanges[std::atomic_ref(outputSizes.maxClusterCount)++] = node;
          if(node.count < input.config.minClusterSize)
          {
            underflowClusters++;
          }
        }
        else
        {
          nvcluster::Result res = splitNode(input, intermediates, node, perNodeElementIndexRanges[nextNodeRangeList], nodesBAlloc);
          if(res != nvcluster::Result::SUCCESS)
          {
            result = res;
            NVCLUSTER_PARALLEL_FOR_BREAK;
          }
        }
      }
      NVCLUSTER_PARALLEL_FOR_END;
      if(result != nvcluster::Result::SUCCESS)
      {
        return result;
      }
    }

    perNodeElementIndexRanges[nextNodeRangeList].resize(nodesBAlloc);  // resize down to what was used before the swap
    perNodeElementIndexRanges[currentNodeRangeList].clear();

    currentNodeRangeList = (currentNodeRangeList + 1) % 2;
    nextNodeRangeList    = (nextNodeRangeList + 1) % 2;
  }

  // It is possible to have less than the minimum number of items per cluster,
  // but there should be at most one.
  if(input.config.preSplitThreshold == 0)
  {
    if(underflowClusters > 1)
    {
      return nvcluster::Result::ERROR_INTERNAL;
    }
  }
  else
  {
    if(underflowClusters > div_ceil(spatialElements.elementCount, sanitizedPreSplitThreshold))
    {
      return nvcluster::Result::ERROR_INTERNAL;
    }
  }

  clusters.clusteredElementIndexCount = outputSizes.maxClusteredElementCount;
  clusters.clusterCount               = outputSizes.maxClusterCount;

  return nvcluster::Result::SUCCESS;
}

}  // namespace nvcluster_internal
