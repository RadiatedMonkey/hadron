#include <hadron/vulkan/instance.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/util.hpp>

#include <iostream>

#include <volk.h>
#include <spdlog/spdlog.h>

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

    static constexpr const char* ENABLED_INSTANCE_EXTENSIONS[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME
    };

    static constexpr const char* ENABLED_INSTANCE_LAYERS[] = {
        "VK_LAYER_KHRONOS_validation"
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

        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Compute";
        appInfo.applicationVersion = 1;
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkValidationFeatureEnableEXT printFeature = VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT;

        VkValidationFeaturesEXT validationFeatures = {};
        validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
        validationFeatures.enabledValidationFeatureCount = 1;
        validationFeatures.pEnabledValidationFeatures = &printFeature;

        VkInstanceCreateInfo instanceCi = {};
        instanceCi.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCi.pNext = &validationFeatures;
        instanceCi.enabledExtensionCount = std::size(ENABLED_INSTANCE_EXTENSIONS);
        instanceCi.ppEnabledExtensionNames = ENABLED_INSTANCE_EXTENSIONS;
        instanceCi.enabledLayerCount = std::size(ENABLED_INSTANCE_LAYERS);
        instanceCi.ppEnabledLayerNames = ENABLED_INSTANCE_LAYERS;
        instanceCi.pApplicationInfo = &appInfo;

        result = vkCreateInstance(&instanceCi, nullptr, &mInstance);
        if (result != VK_SUCCESS) {
            spdlog::error("Failed to create instance: {}", static_cast<uint32_t>(result));
            throw std::runtime_error("Failed to create instance");
        }

        volkLoadInstance(mInstance);

        spdlog::debug("Created instance");

        VkDebugUtilsMessengerCreateInfoEXT debugCi = {};
        debugCi.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;

        debugCi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;

        debugCi.pfnUserCallback = debugCallback;

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
