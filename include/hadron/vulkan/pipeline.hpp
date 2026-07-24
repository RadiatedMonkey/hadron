#pragma once

#include <string>
#include <memory>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <hadron/vulkan/shader.hpp>

namespace Hadron {
    class Device;
    class Shader;

    struct PipelineConfig {
        ShaderConfig shaderConfig;
    };

    class ComputePipeline {
    public:
        ComputePipeline(ComputePipeline&& other) noexcept;
        ComputePipeline(const ComputePipeline&) = delete;
        ~ComputePipeline();

        VkPipeline handle();
        VkPipelineLayout layout();

    private:
        friend Device;

        ComputePipeline(std::shared_ptr<Device> device, const PipelineConfig& config);

        void destroyResources();

        std::shared_ptr<Device> mDevice = nullptr;
        std::shared_ptr<Shader> mShader = nullptr;

        VkDescriptorSetLayout mBindlessLayout = VK_NULL_HANDLE;
        VkDescriptorSet mBindlessSet = VK_NULL_HANDLE;
        VkDescriptorPool mDescriptorPool = VK_NULL_HANDLE;
        VkPipelineLayout mLayout = VK_NULL_HANDLE;
        VkPipeline mPipeline = VK_NULL_HANDLE;
    };
}
