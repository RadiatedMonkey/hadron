#pragma once

#include <cstdint>

namespace Hadron::Workload {
    struct WorkloadInfo;

    void computeUniformMean(const WorkloadInfo& info, uint64_t seed);
}
