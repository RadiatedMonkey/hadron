#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <memory>

namespace Hadron {
    class Device;

    class Fence {
    public:
        Fence(Fence&& other) noexcept;
        Fence(const Fence&) = delete;
        ~Fence();

        VkFence handle();

        void await(uint64_t timeout = UINT64_MAX) const;
        void reset();

    private:
        friend Device;

        Fence(std::shared_ptr<Device> device, bool signaled);

        std::shared_ptr<Device> mDevice = nullptr;
        VkFence mFence = VK_NULL_HANDLE;
    };
}
