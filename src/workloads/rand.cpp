#include <hadron/workloads/rand.hpp>
#include <hadron/workloads/workload.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/vulkan/pipeline.hpp>
#include <hadron/vulkan/storage.hpp>
#include <hadron/vulkan/fence.hpp>
#include <hadron/util.hpp>

#include <array>
#include <iostream>

#include <volk.h>

struct PushConstants {
    uint64_t seed;
    uint32_t numCount;
    uint32_t dispatched;
    VkDeviceAddress outBuffer;
};

namespace Hadron::Workload {
    static constexpr const char* RAND_SHADER_PATH = "/shaders/rand.spv";
    static constexpr uint32_t NUM_COUNT = 4096;
    static constexpr uint32_t WORKGROUP_SIZE = 32;

    template<typename T, size_t N> requires std::is_floating_point_v<T>
    void printBuckets(const std::array<T, N>& results) {
        static constexpr size_t BUCKET_COUNT = 10;

        std::array<uint32_t, BUCKET_COUNT> buckets = {};
        for(T num : results) {
            double ratio = static_cast<double>(BUCKET_COUNT) * num;
            uint32_t bucket = static_cast<uint32_t>(std::floor(ratio));

            if(bucket >= 10) bucket = 9;

            buckets[bucket]++;
        }

        for(uint32_t i = 0; i < BUCKET_COUNT; i++) {
            uint32_t bucket = buckets[i];
            std::cout << i << ": (" << bucket << "): " << std::string(bucket, '*') << std::endl;
        }

        std::cout << "Expected average of " << static_cast<double>(NUM_COUNT) / static_cast<double>(BUCKET_COUNT) << " per bucket" << std::endl;
    }

    void randomWorkload(const WorkloadInfo& info, uint64_t seed) {
        PipelineConfig pipelineConfig = {
            .shaderConfig = { .moduleName = "rand", .entryPoint = "main" }
        };

        auto cmds = info.device->createCmdBuffer();
        auto pipeline = info.device->createPipeline(pipelineConfig);
        auto fence = info.device->createFence();

        // Create output buffer
        auto outputBuffer = info.device->createStorageBuffer<double>(NUM_COUNT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        auto hostBuffer = info.device->createHostBuffer<double>(NUM_COUNT, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        cmds.begin();

        vkCmdBindPipeline(cmds.handle(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle());

        uint32_t dispatchCount = (NUM_COUNT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        PushConstants pushConstants = {
            .seed = seed,
            .numCount = NUM_COUNT,
            .dispatched = dispatchCount * WORKGROUP_SIZE,
            .outBuffer = outputBuffer->address()
        };

        vkCmdPushConstants(cmds.handle(), pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pushConstants);
        vkCmdDispatch(cmds.handle(), dispatchCount, 1, 1);

        VkBufferMemoryBarrier2 barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.buffer = outputBuffer->handle();
        barrier.size = VK_WHOLE_SIZE;
        barrier.offset = 0;

        VkDependencyInfo depInfo = {};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(cmds.handle(), &depInfo);

        VkBufferCopy2 copyRegion = {};
        copyRegion.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
        copyRegion.size = NUM_COUNT * sizeof(double);

        VkCopyBufferInfo2 copyInfo = {};
        copyInfo.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
        copyInfo.regionCount = 1;
        copyInfo.pRegions = &copyRegion;
        copyInfo.srcBuffer = outputBuffer->handle();
        copyInfo.dstBuffer = hostBuffer->handle();

        vkCmdCopyBuffer2(cmds.handle(), &copyInfo);

        cmds.end();

        VkCommandBufferSubmitInfo cmdInfo = {};
        cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cmdInfo.commandBuffer = cmds.handle();

        VkSubmitInfo2 submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdInfo;

        LOG_VKRESULT(vkQueueSubmit2(info.device->queue(), 1, &submitInfo, fence.handle()), "Failed to submit command buffer");

        fence.await();

        auto mapped = hostBuffer->map();
        std::array<double, NUM_COUNT> output = {};
        std::memcpy(output.data(), mapped.get(), NUM_COUNT * sizeof(double));

        printBuckets(output);
    }
}
