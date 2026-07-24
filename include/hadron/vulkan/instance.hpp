#pragma once

#include <memory>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace Hadron {
    class Device;

    class Instance : public std::enable_shared_from_this<Instance> {
    public:
        static std::shared_ptr<Instance> create();

        Instance(Instance&& other) noexcept;
        Instance(const Instance&) = delete;
        ~Instance();

        std::shared_ptr<Device> createDevice();

        VkInstance handle() {
            return mInstance;
        }

    private:
        Instance();

        void destroyResources();

        VkDebugUtilsMessengerEXT mDebug = VK_NULL_HANDLE;
        VkInstance mInstance = VK_NULL_HANDLE;
    };
}
