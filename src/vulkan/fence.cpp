#include <hadron/vulkan/fence.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/util.hpp>

#include <volk.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_core.h>

namespace Hadron {
    Fence::Fence(std::shared_ptr<Device> device, bool signaled) : mDevice(std::move(device)) {
        VkFenceCreateInfo fenceCi = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0U
        };

        LOG_VKRESULT(
            vkCreateFence(mDevice->handle(), &fenceCi, nullptr, &mFence),
            "Failed to create fence"
        );

        spdlog::trace("Created fence");
    }

    Fence::Fence(Fence&& other) noexcept : mDevice(std::move(other.mDevice)), mFence(other.mFence) {
        other.mFence = VK_NULL_HANDLE;
    }

    Fence::~Fence() {
        if (mFence != VK_NULL_HANDLE) {
            vkDestroyFence(mDevice->handle(), mFence, nullptr);
            mFence = VK_NULL_HANDLE;

            spdlog::trace("Destroyed fence");
        }
    }

    VkFence Fence::handle() {
        return mFence;
    }

    void Fence::await(uint64_t timeout) const {
        LOG_VKRESULT(
            vkWaitForFences(mDevice->handle(), 1, &mFence, VK_TRUE, timeout),
            "Failed to await fence signal"
        );
    }

    void Fence::reset() {
        LOG_VKRESULT(
            vkResetFences(mDevice->handle(), 1, &mFence),
            "Failed to reset fence"
        );
    }
}
