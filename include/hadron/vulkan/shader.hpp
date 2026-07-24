#pragma once

#include <memory>
#include <string>
#include <vector>

#include <slang/slang-com-ptr.h>
#include <slang/slang.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace slang {
    class IModule;
}

namespace Hadron {
    class Device;

    struct ShaderConfig {
        /// @brief The shader module name.
        const char* moduleName;

        /// @brief The shader entrypoint to use.
        ///
        /// Set this to `nullptr` to use the first available entrypoint found in the shader module.
        const char* entryPoint = nullptr;
    };

    class Shader {
    public:
        Shader(Shader&& other) noexcept;
        Shader(const Shader&) = delete;
        ~Shader();

        /// @brief Returns a handle to the Vulkan shader module.
        VkShaderModule handle();

        /// @brief Returns the name of the entry point.
        std::string entryPoint() const;

        std::vector<VkPushConstantRange> pushConstants() const;

    private:
        friend Device;

        Shader(std::shared_ptr<Device> device, const ShaderConfig& config);

        std::shared_ptr<Device> mDevice;

        Slang::ComPtr<slang::IComponentType> mComposedProgram = nullptr;
        Slang::ComPtr<slang::IEntryPoint> mEntryPoint = nullptr;
        slang::ProgramLayout* mLayout = nullptr;
        slang::IModule* mSlangModule = nullptr;
        VkShaderModule mModule = VK_NULL_HANDLE;
    };
}
