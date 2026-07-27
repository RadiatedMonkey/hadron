#include <hadron/vulkan/command.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/util.hpp>

#include <volk.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_core.h>

namespace Hadron {
    VkCommandBuffer Commands::handle() {
        return mBuffer;
    }

    Commands::Commands(std::shared_ptr<Device> device, VkCommandBuffer buffer)
        : mDevice(std::move(device)), mBuffer(buffer) {}

    Commands::Commands(Commands&& other) noexcept : mDevice(std::move(other.mDevice)), mBuffer(other.mBuffer) {
        other.mBuffer = VK_NULL_HANDLE;
    }

    Commands::~Commands() {
        if (mBuffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(mDevice->handle(), mDevice->cmdPool(), 1, &mBuffer);
            mBuffer = VK_NULL_HANDLE;

            spdlog::debug("Freed command buffer");
        }
    }

    void Commands::begin(bool oneTimeSubmit) {
        VkCommandBufferBeginInfo beginInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = oneTimeSubmit ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0U
        };

        LOG_VKRESULT(vkBeginCommandBuffer(mBuffer, &beginInfo), "Failed to start command buffer");
    }

    void Commands::end() {
        LOG_VKRESULT(vkEndCommandBuffer(mBuffer), "Failed to end command buffer");
    }
}
