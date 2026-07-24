#include <hadron/vulkan/instance.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/workloads/mean.hpp>
#include <hadron/workloads/workload.hpp>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

int main() {
    auto logger = spdlog::stdout_color_mt("logger");

    spdlog::set_level(spdlog::level::trace);
    spdlog::trace("Logger initialized");

    try {
        auto instance = Hadron::Instance::create();
        auto device = instance->createDevice();

        Hadron::Workload::WorkloadInfo info = {
            .device = device
        };

        Hadron::Workload::computeUniformMean(info, 0);
    } catch(const std::exception& e) {
        spdlog::error("exception: {}", e.what());
        return 1;
    } catch(...) {
        spdlog::error("unknown exception occurred");
        return 1;
    }

    spdlog::trace("exiting");

    return 0;
}
