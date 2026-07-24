#pragma once

#include <hadron/vulkan/device.hpp>
#include <hadron/vulkan/pipeline.hpp>
#include <hadron/util.hpp>

#include <memory>

#include <spdlog/spdlog.h>
#include <volk.h>

namespace Hadron {
    class Device;

    template<typename T>
    class HostBuffer;

    struct BufferReference {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    uint32_t findMemoryType(uint32_t typeBits, VkPhysicalDeviceMemoryProperties memProperties, VkMemoryPropertyFlags propertyFlags);

    /// Represents a pointer to the base of a mapped memory buffer.
    ///
    /// Upon destruction, this pointer automatically unmaps the memory.
    template<typename T>
    class Mapped {
    public:
        T* get() {
            return mPtr;
        }

        const T* get() const {
            return mPtr;
        }

        Mapped(Mapped&& other) noexcept : mBuffer(std::move(other.mBuffer)), mPtr(other.mPtr) {
            other.mPtr = nullptr;
        }

        Mapped(const Mapped&) = delete;

        ~Mapped() {
            VkMemoryUnmapInfo unmapInfo = {};
            unmapInfo.sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO;
            unmapInfo.memory = mBuffer->memory();

            VkResult result = vkUnmapMemory2KHR(mBuffer->mDevice->handle(), &unmapInfo);
            if (result != VK_SUCCESS) {
                spdlog::error("Failed to unmap memory");
                // No exception is thrown here since this error is not critical
                // and throwing an exception in a destructor will likely cause termination.
            }

            spdlog::debug("Unmapped buffer from host memory");
        }

        /// The size of the underlying memory.
        VkDeviceSize size() {
            return mBuffer->size();
        }

    private:
        friend HostBuffer<T>;

        Mapped(std::shared_ptr<HostBuffer<T>> buffer) : mBuffer(std::move(buffer)) {
            VkMemoryMapInfo mapInfo = {};
            mapInfo.sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO;
            mapInfo.size = mBuffer->size();
            mapInfo.memory = mBuffer->memory();
            mapInfo.offset = 0;

            LOG_VKRESULT(
                vkMapMemory2KHR(mBuffer->mDevice->handle(), &mapInfo, reinterpret_cast<void**>(&mPtr)),
                "Failed to map memory to host buffer"
            );

            spdlog::debug("Mapped buffer to host memory");
        }

        std::shared_ptr<HostBuffer<T>> mBuffer;
        T* mPtr = nullptr;
    };

    /// @brief A host-visible, host-coherent buffer.
    template<typename T>
    class HostBuffer : public std::enable_shared_from_this<HostBuffer<T>> {
    public:
        ~HostBuffer() {
            freeMemory();
            destroyBuffer();
        }

        /// @brief Maps the buffer into host address space.
        ///
        /// @note The buffer is automatically unmapped once the mapped pointer is dropped.
        Mapped<T> map() {
            return Mapped<T>(this->shared_from_this());
        }

        VkBuffer handle() {
            return mBuffer;
        }

        VkDeviceSize size() {
            return mSize;
        }

        VkDeviceMemory memory() {
            return mMemory;
        }

    private:
        friend Device;
        friend Mapped<T>;

        /// @note This constructor is private to ensure that a host buffer can only be constructed inside of
        /// a shared_ptr. This prevents buffer mapping breaking due to no shared pointers existing.
        ///
        /// @param device The device to create this buffer on.
        /// @param size The number of elements of type `T` to fit into this buffer.
        /// @param usageFlags Buffer usage flags
        HostBuffer(std::shared_ptr<Device> device, uint64_t size, VkBufferUsageFlags usageFlags)
            : mDevice(std::move(device)), mSize(size * sizeof(T))
        {
            VkBufferCreateInfo bufferCi = {};
            bufferCi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCi.size = mSize;
            bufferCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            // Allow both copying to and from host buffers.
            bufferCi.usage = usageFlags;

            LOG_VKRESULT(
                vkCreateBuffer(mDevice->handle(), &bufferCi, nullptr, &mBuffer),
                "Failed to create host buffer"
            );

            spdlog::debug("Created host buffer");

            VkBufferMemoryRequirementsInfo2 memReqInfo = {};
            memReqInfo.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
            memReqInfo.buffer = mBuffer;

            VkMemoryRequirements2 memReqs = {};
            memReqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;

            vkGetBufferMemoryRequirements2(mDevice->handle(), &memReqInfo, &memReqs);

            uint32_t memoryTypeIndex = findMemoryType(
                memReqs.memoryRequirements.memoryTypeBits,
                mDevice->memProperties().memoryProperties,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            );

            VkMemoryAllocateInfo memoryCi = {};
            memoryCi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            memoryCi.memoryTypeIndex = memoryTypeIndex;
            memoryCi.allocationSize = memReqs.memoryRequirements.size;

            CHECK_VKRESULT(
                vkAllocateMemory(mDevice->handle(), &memoryCi, nullptr, &mMemory),
                "Failed to allocate host buffer memory",
                destroyBuffer()
            );

            spdlog::debug("Allocated {} bytes of memory for the host buffer", mSize);

            VkBindBufferMemoryInfo bindInfo = {};
            bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
            bindInfo.buffer = mBuffer;
            bindInfo.memory = mMemory;
            bindInfo.memoryOffset = 0;

            CHECK_VKRESULT(
                vkBindBufferMemory2(mDevice->handle(), 1, &bindInfo),
                "Failed to bind staging buffer memory to buffer",
                {
                    freeMemory();
                    destroyBuffer();
                }
            );

            spdlog::debug("Bound host buffer and memory");
        }

        void destroyBuffer() {
            if (mBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(mDevice->handle(), mBuffer, nullptr);
                mBuffer = VK_NULL_HANDLE;
                spdlog::debug("Destroyed host buffer");
            }
        }

        void freeMemory() {
            if (mMemory != VK_NULL_HANDLE) {
                vkFreeMemory(mDevice->handle(), mMemory, nullptr);
                mMemory = VK_NULL_HANDLE;
                spdlog::debug("Freed host buffer memory");
            }
        }

        std::shared_ptr<Device> mDevice = nullptr;

        VkDeviceSize mSize = 0;
        VkBuffer mBuffer = VK_NULL_HANDLE;
        VkDeviceMemory mMemory = VK_NULL_HANDLE;
    };

    class StorageBuffer {
    public:
        ~StorageBuffer();

        VkDeviceAddress address();
        VkBuffer handle();

    private:
        friend Device;

        /// @brief Creates a storage buffer.
        ///
        /// @param device The device to create the storage buffer on.
        /// @param size The size in bytes of the buffer to be created.
        /// @param usageFlags Usage flags for this buffer (on top of VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT and VK_BUFFER_USAGE_STORAGE_BUFFER_BIT).
        StorageBuffer(std::shared_ptr<Device> device, VkDeviceSize size, VkBufferUsageFlags usageFlags);

        void freeMemory();
        void destroyBuffer();

        std::shared_ptr<Device> mDevice = nullptr;

        VkDeviceSize mSize = 0;
        VkBuffer mBuffer = VK_NULL_HANDLE;
        VkDeviceMemory mMemory = VK_NULL_HANDLE;
    };

    // These are defined here to avoid a circular dependency between storage.hpp and device.hpp.

    template<typename T>
    std::shared_ptr<HostBuffer<T>> Device::createHostBuffer(uint64_t size, VkBufferUsageFlags usageFlags) {
        return std::shared_ptr<HostBuffer<T>>(new HostBuffer<T>(shared_from_this(), size, usageFlags));
    }

    template<typename T>
    std::shared_ptr<StorageBuffer> Device::createStorageBuffer(uint64_t size, VkBufferUsageFlags usageFlags) {
        return std::shared_ptr<StorageBuffer>(new StorageBuffer(shared_from_this(), size * sizeof(T), usageFlags));
    }
}
