#pragma once

#include <array>
#include <memory>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <slang/slang-com-ptr.h>
#include <slang/slang.h>

#include <hadron/vulkan/fence.hpp>
#include <hadron/vulkan/command.hpp>

namespace Hadron {
    static constexpr std::array<const char*, 1> kShaderIncludePaths = {
        SOURCE_DIR "/src/shaders"
    };

    static constexpr std::array<const char*, 4> kDeviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_KHR_MAP_MEMORY_2_EXTENSION_NAME
    };

    class Instance;
    class ComputePipeline;
    class Shader;
    struct PipelineConfig;
    struct ShaderConfig;

    class StorageBuffer;
    template<typename T> class HostBuffer;

    class Device : public std::enable_shared_from_this<Device> {
    public:
        Device(Device&& other) noexcept;
        Device(const Device&) = delete;

        /// @warning The destructor should only run once all device-created resources have been destroyed.
        ~Device();

        /// @brief The raw Vulkan handle to this device.
        VkDevice handle();

        /// @brief The device queue.
        VkQueue queue();

        Fence createFence(bool signaled = false);

        Shader createShader(const ShaderConfig& config);

        std::shared_ptr<ComputePipeline> createPipeline(const PipelineConfig& config);

        /// @brief Creates a host-visible, host-coherent buffer.
        /// @param size The size in elements of the buffer to create.
        ///
        /// @note This can be used as a staging buffer to populate device local buffers.
        template<typename T>
        std::shared_ptr<HostBuffer<T>> createHostBuffer(uint64_t size, VkBufferUsageFlags usageFlags);

        /// @brief Creates a device-local storage buffer of specified size.
        /// @param size The amount of elements of type `T` to fit into this buffer.
        ///
        /// @note This buffer cannot be mapped to host memory directly and must be populated by
        /// transferring from a staging buffer.
        template<typename T>
        std::shared_ptr<StorageBuffer> createStorageBuffer(uint64_t size, VkBufferUsageFlags usageFlags);

        /// @brief The raw Vulkan command pool handle.
        VkCommandPool cmdPool();

        /// @brief Creates a command buffer that can be recorded into.
        ///
        /// @warning The command buffer must be begun by the caller using `vkBeginCommandBuffer`.
        Commands createCmdBuffer();

        VkPhysicalDevice physicalDevice();

        Slang::ComPtr<slang::ISession>& localSlangSession();
    private:
        friend Instance;

        Device(std::shared_ptr<Instance> instance);

        void destroyResources();

        std::shared_ptr<Instance> mInstance = nullptr;

        Slang::ComPtr<slang::IGlobalSession> mGlobalSession;
        Slang::ComPtr<slang::ISession> mLocalSession;

        VkPhysicalDevice mPhysicalDevice;
        VkCommandPool mPool = VK_NULL_HANDLE;
        VkQueue mQueue = VK_NULL_HANDLE;
        VkDevice mDevice = VK_NULL_HANDLE;

        uint32_t mQueueFamilyIndex = UINT32_MAX;
    };
}
