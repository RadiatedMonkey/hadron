#pragma once

#include <memory>

namespace Hadron {
    class Device;
}

namespace Hadron::Workload {
    struct WorkloadInfo {
        std::shared_ptr<Device> device;
    };
}
