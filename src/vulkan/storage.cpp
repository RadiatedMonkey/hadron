#include <hadron/vulkan/storage.hpp>
#include <hadron/vulkan/device.hpp>

#include <spdlog/spdlog.h>
#include <volk.h>

namespace Hadron {
    uint32_t findMemoryType(uint32_t typeBits, VkPhysicalDeviceMemoryProperties memProperties, VkMemoryPropertyFlags propertyFlags) {
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            bool hardwareSupported = typeBits & (1 << i);
            bool hasFlags = (memProperties.memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags;

            if (hardwareSupported && hasFlags) {
                return i;
            }
        }

        return UINT32_MAX;
    }

    StorageBuffer::StorageBuffer(std::shared_ptr<Device> device, VkDeviceSize size, VkBufferUsageFlags usageFlags)
        : mDevice(std::move(device)), mSize(size)
    {
        VkBufferCreateInfo bufferCi = {};
        bufferCi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCi.size = mSize;
        bufferCi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        // bufferCi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferCi.usage = usageFlags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        LOG_VKRESULT(
            vkCreateBuffer(mDevice->handle(), &bufferCi, nullptr, &mBuffer),
            "Failed to create storage buffer"
        );

        spdlog::debug("Created storage buffer");

        VkBufferMemoryRequirementsInfo2 memReqInfo = {};
        memReqInfo.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
        memReqInfo.buffer = mBuffer;

        VkMemoryRequirements2 memReqs = {};
        memReqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;

        vkGetBufferMemoryRequirements2(mDevice->handle(), &memReqInfo, &memReqs);

        uint32_t memoryTypeIndex = findMemoryType(
            memReqs.memoryRequirements.memoryTypeBits,
            mDevice->memProperties().memoryProperties,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        VkMemoryAllocateFlagsInfo flagsInfo = {};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo memoryCi = {};
        memoryCi.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryCi.pNext = &flagsInfo;
        memoryCi.memoryTypeIndex = memoryTypeIndex;
        memoryCi.allocationSize = memReqs.memoryRequirements.size;

        CHECK_VKRESULT(
            vkAllocateMemory(mDevice->handle(), &memoryCi, nullptr, &mMemory),
            "Failed to allocate storage buffer memory",
            destroyBuffer()
        );

        spdlog::debug("Allocated {} bytes of memory for a storage buffer", memoryCi.allocationSize);

        VkBindBufferMemoryInfo bindInfo = {};
        bindInfo.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
        bindInfo.buffer = mBuffer;
        bindInfo.memory = mMemory;
        bindInfo.memoryOffset = 0;

        CHECK_VKRESULT(
            vkBindBufferMemory2(mDevice->handle(), 1, &bindInfo),
            "Failed to bind storage buffer memory to buffer",
            {
                freeMemory();
                destroyBuffer();
            }
        );

        spdlog::debug("Bound memory and buffer together");
    }

    StorageBuffer::~StorageBuffer() {
        freeMemory();
        destroyBuffer();
    }

    VkDeviceAddress StorageBuffer::address() {
        VkBufferDeviceAddressInfo addressInfo = {};
        addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addressInfo.buffer = mBuffer;

        return vkGetBufferDeviceAddress(mDevice->handle(), &addressInfo);
    }

    VkBuffer StorageBuffer::handle() {
        return mBuffer;
    }

    void StorageBuffer::destroyBuffer() {
        if (mBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(mDevice->handle(), mBuffer, nullptr);
            mBuffer = VK_NULL_HANDLE;

            spdlog::debug("Destroyed storage buffer");
        }
    }

    void StorageBuffer::freeMemory() {
        if (mMemory != VK_NULL_HANDLE) {
            vkFreeMemory(mDevice->handle(), mMemory, nullptr);
            mMemory = VK_NULL_HANDLE;

            spdlog::debug("Freed storage buffer memory");
        }
    }
}
