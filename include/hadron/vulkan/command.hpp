#pragma once

#include <memory>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace Hadron {
    class Device;

    class Commands {
    public:
        Commands(Commands&& rhs) noexcept;
        Commands(const Commands&) = delete;
        ~Commands();

        void begin();
        void end();
        void reset();
        VkCommandBuffer handle();

    private:
        friend Device;

        Commands(std::shared_ptr<Device> device, VkCommandBuffer buffer);

        std::shared_ptr<Device> mDevice = nullptr;

        VkCommandBuffer mBuffer = VK_NULL_HANDLE;
    };
}
