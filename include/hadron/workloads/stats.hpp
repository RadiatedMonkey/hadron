#pragma once

#include <cstdint>

namespace Hadron::Workload {
    struct WorkloadInfo;

    void computeUniformStats(const WorkloadInfo& info, uint64_t seed);
}
