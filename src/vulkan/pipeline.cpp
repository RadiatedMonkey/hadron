#include <hadron/vulkan/pipeline.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/vulkan/shader.hpp>
#include <hadron/util.hpp>

#include <spdlog/spdlog.h>
#include <volk.h>

#include <array>
#include <vulkan/vulkan_core.h>

static constexpr uint32_t MAX_BINDLESS_RESOURCES = 100;

namespace Hadron {
    ComputePipeline::ComputePipeline(std::shared_ptr<Device> device, const PipelineConfig& config) : mDevice(std::move(device)) {
        static constexpr uint32_t DESCRIPTOR_TYPE_COUNT = 2;

        std::array<VkDescriptorSetLayoutBinding, DESCRIPTOR_TYPE_COUNT> bindings = {};
        std::array<VkDescriptorBindingFlags, DESCRIPTOR_TYPE_COUNT> flags = {};
        std::array<VkDescriptorType, DESCRIPTOR_TYPE_COUNT> descriptorTypes = {
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        };

        for (uint32_t i = 0; i < bindings.size(); i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = descriptorTypes[i];
            bindings[i].descriptorCount = MAX_BINDLESS_RESOURCES;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            flags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        }

        VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCi = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = flags.size(),
            .pBindingFlags = flags.data()
        };

        VkDescriptorSetLayoutCreateInfo setLayoutCi = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &bindingFlagsCi,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            .bindingCount = bindings.size(),
            .pBindings = bindings.data()
        };

        CHECK_VKRESULT(
            vkCreateDescriptorSetLayout(mDevice->handle(), &setLayoutCi, nullptr, &mBindlessLayout),
            "Failed to create bindless descriptor set layout",
            {
                destroyResources();
            }
        );

        spdlog::debug("Created descriptor set layout");

        std::array<VkDescriptorPoolSize, DESCRIPTOR_TYPE_COUNT> poolSizes = {};
        for (uint32_t i = 0; i < poolSizes.size(); i++) {
            poolSizes[i].type = descriptorTypes[i];
            poolSizes[i].descriptorCount = MAX_BINDLESS_RESOURCES;
        }

        VkDescriptorPoolCreateInfo poolCi = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = poolSizes.size(),
            .pPoolSizes = poolSizes.data(),
        };

        CHECK_VKRESULT(
            vkCreateDescriptorPool(mDevice->handle(), &poolCi, nullptr, &mDescriptorPool),
            "Failed to create descriptor pool",
            {
                destroyResources();
            }
        );

        spdlog::debug("Created descriptor pool");

        VkDescriptorSetAllocateInfo allocCi = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = mDescriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &mBindlessLayout
        };

        CHECK_VKRESULT(
            vkAllocateDescriptorSets(mDevice->handle(), &allocCi, &mBindlessSet),
            "Failed to allocate descriptor set",
            {
                destroyResources();
            }
        );

        spdlog::debug("Allocated bindless descriptor set");

        Shader shader = mDevice->createShader(config.shaderConfig);
        std::vector<VkPushConstantRange> pushConstants = shader.pushConstants();

        spdlog::info("Registered {} push constant block(s)", pushConstants.size());

        VkPipelineLayoutCreateInfo layoutCi = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &mBindlessLayout,
            .pushConstantRangeCount = static_cast<uint32_t>(pushConstants.size()),
            .pPushConstantRanges = pushConstants.data(),
        };

        CHECK_VKRESULT(
            vkCreatePipelineLayout(mDevice->handle(), &layoutCi, nullptr, &mLayout),
            "Failed to create compute pipeline layout",
            {
                destroyResources();
            }
        );

        spdlog::debug("Created compute pipeline layout");

        std::string entryPointName = shader.entryPoint();
        spdlog::debug("Using shader entrypoint \"{}\"", entryPointName);

        VkPipelineShaderStageCreateInfo stageCi = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader.handle(),
            .pName = entryPointName.c_str(),
        };

        VkComputePipelineCreateInfo pipelineCi = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stageCi,
            .layout = mLayout,
        };

        CHECK_VKRESULT(
            vkCreateComputePipelines(mDevice->handle(), VK_NULL_HANDLE, 1, &pipelineCi, nullptr, &mPipeline),
            "Failed to create compute pipeline",
            {
                destroyResources();
            }
        );

        spdlog::debug("Created compute pipeline");
    }

    ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept :
        mDevice(std::move(other.mDevice)), mPipeline(other.mPipeline), mLayout(other.mLayout),
        mDescriptorPool(other.mDescriptorPool), mBindlessLayout(other.mBindlessLayout), mBindlessSet(other.mBindlessSet)
     {
        other.mPipeline = VK_NULL_HANDLE;
        other.mLayout = VK_NULL_HANDLE;
        other.mDescriptorPool = VK_NULL_HANDLE;
        other.mBindlessLayout = VK_NULL_HANDLE;
        other.mBindlessSet = VK_NULL_HANDLE;
    }

    ComputePipeline::~ComputePipeline() {
        destroyResources();
    }

    VkPipelineLayout ComputePipeline::layout() {
        return mLayout;
    }

    VkPipeline ComputePipeline::handle() {
        return mPipeline;
    }

    void ComputePipeline::destroyResources() {
        if (mPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(mDevice->handle(), mPipeline, nullptr);
            mPipeline = VK_NULL_HANDLE;

            spdlog::debug("Destroyed pipeline");
        }

        if (mLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(mDevice->handle(), mLayout, nullptr);
            mLayout = VK_NULL_HANDLE;

            spdlog::debug("Destroyed pipeline layout");
        }

        if (mBindlessLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(mDevice->handle(), mBindlessLayout, nullptr);
            mBindlessLayout = VK_NULL_HANDLE;

            spdlog::debug("Destroyed bindless descriptor set layout");
        }

        if (mBindlessLayout != VK_NULL_HANDLE) {
            vkFreeDescriptorSets(mDevice->handle(), mDescriptorPool, 1, &mBindlessSet);
            mBindlessSet = VK_NULL_HANDLE;

            spdlog::debug("Freed bindless descriptor set");
        }

        if (mDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(mDevice->handle(), mDescriptorPool, nullptr);
            mDescriptorPool = VK_NULL_HANDLE;

            spdlog::debug("Destroyed descriptor pool");
        }
    }
}
