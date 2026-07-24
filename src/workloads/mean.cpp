#include <hadron/workloads/mean.hpp>
#include <hadron/workloads/workload.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/vulkan/pipeline.hpp>
#include <hadron/vulkan/storage.hpp>
#include <hadron/vulkan/fence.hpp>
#include <hadron/util.hpp>

#include <array>
#include <iostream>

#include <volk.h>

struct RandConstants {
    uint64_t seed;
    uint32_t numCount;
    uint32_t dispatched;
    VkDeviceAddress outBuffer;
};

struct MeanConstants {
    VkDeviceAddress randBuffer;
    VkDeviceAddress meanBuffer;
    uint32_t randBufferSize;
    uint32_t dispatched;
};

namespace Hadron::Workload {
    static constexpr uint32_t NUM_COUNT = 4096;
    static constexpr uint32_t WORKGROUP_SIZE = 256;
    static constexpr uint32_t DISPATCH_COUNT = (NUM_COUNT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

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

    void computeUniformMean(const WorkloadInfo& info, uint64_t seed) {
        static constexpr const char* RAND_PATH = SOURCE_DIR "/src/shaders/workloads/rand";
        static constexpr const char* MEAN_PATH = SOURCE_DIR "/src/shaders/workloads/mean";

        PipelineConfig pipelineConfig = {
            .shaderConfig = { .moduleName = RAND_PATH, .entryPoint = nullptr }
        };

        auto cmds = info.device->createCmdBuffer();
        auto fence = info.device->createFence();

        auto randPipeline = info.device->createPipeline(pipelineConfig);

        pipelineConfig.shaderConfig.moduleName = MEAN_PATH;
        auto meanPipeline = info.device->createPipeline(pipelineConfig);

        auto inputBuffer = info.device->createStorageBuffer<double>(NUM_COUNT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        auto outputBuffer = info.device->createStorageBuffer<double>(NUM_COUNT / WORKGROUP_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        auto hostRngBuffer = info.device->createHostBuffer<double>(NUM_COUNT, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        auto hostMeanBuffer = info.device->createHostBuffer<double>(DISPATCH_COUNT, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        cmds.begin();

        RandConstants randConstants = {
            .seed = seed,
            .numCount = NUM_COUNT,
            .dispatched = DISPATCH_COUNT * WORKGROUP_SIZE,
            .outBuffer = inputBuffer->address()
        };

        vkCmdBindPipeline(cmds.handle(), VK_PIPELINE_BIND_POINT_COMPUTE, randPipeline->handle());
        vkCmdPushConstants(cmds.handle(), randPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RandConstants), &randConstants);
        vkCmdDispatch(cmds.handle(), DISPATCH_COUNT, 1, 1);

        spdlog::info("Dispatching {} rng workgroups", DISPATCH_COUNT);

        // Barrier waiting for rng results before computing mean.
        VkBufferMemoryBarrier2 barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.buffer = inputBuffer->handle();
        barrier.size = VK_WHOLE_SIZE;

        /// Barrier waiting for rng results before copying to host buffer.
        VkBufferMemoryBarrier2 hostBarrier = {};
        hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        hostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        hostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        hostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        hostBarrier.buffer = inputBuffer->handle();
        hostBarrier.size = VK_WHOLE_SIZE;

        VkBufferMemoryBarrier2 barriers[] = { barrier, hostBarrier };

        VkDependencyInfo depInfo = {};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.bufferMemoryBarrierCount = 2;
        depInfo.pBufferMemoryBarriers = barriers;

        vkCmdPipelineBarrier2(cmds.handle(), &depInfo);

        VkBufferCopy2 copyRegion = {};
        copyRegion.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2;
        copyRegion.size = NUM_COUNT * sizeof(double);

        VkCopyBufferInfo2 copyInfo = {};
        copyInfo.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2;
        copyInfo.srcBuffer = inputBuffer->handle();
        copyInfo.dstBuffer = hostRngBuffer->handle();
        copyInfo.regionCount = 1;
        copyInfo.pRegions = &copyRegion;

        vkCmdCopyBuffer2(cmds.handle(), &copyInfo);

        MeanConstants meanConstants = {
            .randBuffer = inputBuffer->address(),
            .meanBuffer = outputBuffer->address(),
            .randBufferSize = NUM_COUNT,
            .dispatched = DISPATCH_COUNT * WORKGROUP_SIZE
        };

        vkCmdBindPipeline(cmds.handle(), VK_PIPELINE_BIND_POINT_COMPUTE, meanPipeline->handle());
        vkCmdPushConstants(cmds.handle(), meanPipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MeanConstants), &meanConstants);
        vkCmdDispatch(cmds.handle(), DISPATCH_COUNT, 1, 1);

        spdlog::info("Dispatching {} mean workgroups", DISPATCH_COUNT);

        VkBufferMemoryBarrier2 meanBarrier = {};
        meanBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        meanBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        meanBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        meanBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        meanBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        meanBarrier.buffer = outputBuffer->handle();
        meanBarrier.size = VK_WHOLE_SIZE;

        depInfo.bufferMemoryBarrierCount = 1;
        depInfo.pBufferMemoryBarriers = &meanBarrier;

        vkCmdPipelineBarrier2(cmds.handle(), &depInfo);

        copyRegion.size = DISPATCH_COUNT * sizeof(double);

        copyInfo.srcBuffer = outputBuffer->handle();
        copyInfo.dstBuffer = hostMeanBuffer->handle();
        copyInfo.regionCount = 1;
        copyInfo.pRegions = &copyRegion;

        vkCmdCopyBuffer2(cmds.handle(), &copyInfo);

        cmds.end();

        VkCommandBuffer cmdHandle = cmds.handle();

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdHandle;

        LOG_VKRESULT(
            vkQueueSubmit(info.device->queue(), 1, &submitInfo, fence.handle()),
            "Failed to submit commands"
        );

        fence.await();

        auto mappedRng = hostRngBuffer->map();
        std::array<double, NUM_COUNT> randData = {};
        std::memcpy(randData.data(), mappedRng.get(), NUM_COUNT * sizeof(double));

        printBuckets(randData);

        auto mappedMean = hostMeanBuffer->map();
        std::array<double, DISPATCH_COUNT> meanData = {};
        std::memcpy(meanData.data(), mappedMean.get(), DISPATCH_COUNT * sizeof(double));

        double summedMeans = 0.0;
        for (double mean : meanData) {
            summedMeans += mean;
        }
        spdlog::info("Total mean is: {}", summedMeans / meanData.size());

        for (double mean : meanData) {
            spdlog::info("mean for block is {}", mean);
        }
    }
}
