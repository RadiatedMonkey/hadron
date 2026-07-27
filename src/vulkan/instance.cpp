#include <hadron/vulkan/instance.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/util.hpp>

#include <iostream>

#include <volk.h>
#include <spdlog/spdlog.h>
#include <vulkan/vulkan_core.h>

namespace Hadron {
    unsigned int debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
    ) {
        if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
            spdlog::error("{}", pCallbackData->pMessage);
        } else {
            spdlog::info("{}", pCallbackData->pMessage);
        }

        return VK_FALSE;
    }

    static constexpr std::array<const char*, 3> kEnabledInstanceExtensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    static constexpr std::array<const char*, 1> kEnabledInstanceLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    static constexpr std::array<VkValidationFeatureEnableEXT, 1> kValidationFeatures = {
        VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
    };

    std::shared_ptr<Instance> Instance::create() {
        // Due to private constructor, make_shared does not work.
        return std::shared_ptr<Instance>(new Instance());
    }

    Instance::Instance() {
        VkResult result = volkInitialize();
        if (result != VK_SUCCESS) {
            throw std::runtime_error("volkInitialize failed");
        }

        VkApplicationInfo appInfo = {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "Hadron",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "Hadron",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = VK_API_VERSION_1_3
        };

        VkValidationFeaturesEXT validationFeatures = {
            .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
            .enabledValidationFeatureCount = kValidationFeatures.size(),
            .pEnabledValidationFeatures = kValidationFeatures.data()
        };

        VkInstanceCreateInfo instanceCi = {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = &validationFeatures,
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = kEnabledInstanceLayers.size(),
            .ppEnabledLayerNames = kEnabledInstanceLayers.data(),
            .enabledExtensionCount = kEnabledInstanceExtensions.size(),
            .ppEnabledExtensionNames = kEnabledInstanceExtensions.data()
        };

        result = vkCreateInstance(&instanceCi, nullptr, &mInstance);
        if (result != VK_SUCCESS) {
            spdlog::error("Failed to create instance: {}", static_cast<uint32_t>(result));
            throw std::runtime_error("Failed to create instance");
        }

        volkLoadInstance(mInstance);

        spdlog::debug("Created instance");

        VkDebugUtilsMessengerCreateInfoEXT debugCi = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
            .pfnUserCallback = debugCallback
        };

        CHECK_VKRESULT(
            vkCreateDebugUtilsMessengerEXT(mInstance, &debugCi, nullptr, &mDebug),
            "Failed to create debug utils messenger",
            {
                destroyResources();
            }
        );

        spdlog::debug("Created debug utils messenger");
    }

    Instance::Instance(Instance&& other) noexcept : mInstance(other.mInstance), mDebug(other.mDebug) {
        other.mInstance = VK_NULL_HANDLE;
        other.mDebug = VK_NULL_HANDLE;
    }

    Instance::~Instance() {
        destroyResources();
    }

    void Instance::destroyResources() {
        if (mDebug != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(mInstance, mDebug, nullptr);
            mDebug = VK_NULL_HANDLE;

            spdlog::debug("Destroyed debug utils messenger");
        }

        if (mInstance != VK_NULL_HANDLE) {
            vkDestroyInstance(mInstance, nullptr);
            mInstance = VK_NULL_HANDLE;

            volkFinalize();

            spdlog::debug("Destroyed instance");
        }
    }

    std::shared_ptr<Device> Instance::createDevice() {
        // make_shared cannot access private Device constructor.
        return std::shared_ptr<Device>(new Device(shared_from_this()));
    }
}
