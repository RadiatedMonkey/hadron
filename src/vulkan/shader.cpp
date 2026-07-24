#include <hadron/vulkan/shader.hpp>
#include <hadron/vulkan/device.hpp>
#include <hadron/util.hpp>

#include <array>

#include <volk.h>
#include <spdlog/spdlog.h>

namespace Hadron {
    Shader::Shader(std::shared_ptr<Device> device, const ShaderConfig& config)
        : mDevice(std::move(device))
    {
        Slang::ComPtr<slang::ISession>& session = mDevice->localSlangSession();

        Slang::ComPtr<slang::IBlob> diagnosticBlob;
        mSlangModule = session->loadModule(config.moduleName, diagnosticBlob.writeRef());

        if (diagnosticBlob != nullptr) {
            spdlog::warn(
                "Info while loading module `{}`: {}",
                config.moduleName,
                reinterpret_cast<const char*>(diagnosticBlob->getBufferPointer())
            );
        }

        if (!mSlangModule) {
            spdlog::error("Failed to load shader module `{}`", config.moduleName);
            throw std::runtime_error("Failed to compile shader module");
        }

        spdlog::trace("Loaded module `{}`", config.moduleName);

        SlangResult result;
        if (config.entryPoint == nullptr) {
            // Load first found entrypoint
            result = mSlangModule->getDefinedEntryPoint(0, mEntryPoint.writeRef());
        } else {
            // Load specific entrypoint
            result = mSlangModule->findEntryPointByName(config.entryPoint, mEntryPoint.writeRef());
        }

        LOG_SRESULT(result, "Failed to find entrypoint");

        std::array<slang::IComponentType*, 2> componentTypes = {
            mSlangModule, mEntryPoint
        };

        diagnosticBlob = nullptr;
        result = session->createCompositeComponentType(
            componentTypes.data(), componentTypes.size(),
            mComposedProgram.writeRef(), diagnosticBlob.writeRef()
        );

        if (diagnosticBlob != nullptr) {
            spdlog::warn(
                "Info while creating composite component type for `{}`: {}",
                config.moduleName,
                reinterpret_cast<const char*>(diagnosticBlob->getBufferPointer())
            );
        }

        LOG_SRESULT(result, "Failed to create composite component type");

        diagnosticBlob = nullptr;
        Slang::ComPtr<slang::IBlob> spirvCode = nullptr;

        result = mComposedProgram->getEntryPointCode(
            0, 0, spirvCode.writeRef(), diagnosticBlob.writeRef()
        );

        if (diagnosticBlob != nullptr) {
            spdlog::warn(
                "Info while loading entry point SPIR-V `{}`: {}",
                config.moduleName,
                reinterpret_cast<const char*>(diagnosticBlob->getBufferPointer())
            );
        }

        LOG_SRESULT(result, "Failed to load SPIR-V entrypoint");

        spdlog::trace("Loaded SPIR-V entrypoint");

        mLayout = mComposedProgram->getLayout(0, diagnosticBlob.writeRef());

        if (diagnosticBlob != nullptr) {
            spdlog::warn("Info while loading program layout: {}", reinterpret_cast<const char*>(diagnosticBlob->getBufferPointer()));
        }

        if (!mLayout) {
            spdlog::error("Failed to get program layout");
            throw std::runtime_error("Failed to get program layout");
        }

        spdlog::trace("Loaded program layout");

        VkShaderModuleCreateInfo moduleCi = {};
        moduleCi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCi.codeSize = spirvCode->getBufferSize();
        moduleCi.pCode = reinterpret_cast<const uint32_t*>(spirvCode->getBufferPointer());

        LOG_VKRESULT(
            vkCreateShaderModule(mDevice->handle(), &moduleCi, nullptr, &mModule),
            "Failed to create Vulkan shader module"
        );
    }

    Shader::Shader(Shader&& other) noexcept :
        mDevice(std::move(other.mDevice)),
        mSlangModule(std::move(other.mSlangModule)),
        mComposedProgram(std::move(other.mComposedProgram)),
        mEntryPoint(std::move(other.mEntryPoint))
    {
        mLayout = other.mLayout;
        other.mLayout = nullptr;

        mModule = other.mModule;
        other.mModule = VK_NULL_HANDLE;
    }

    Shader::~Shader() {
        if (mModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(mDevice->handle(), mModule, nullptr);
            mModule = VK_NULL_HANDLE;

            spdlog::trace("Destroyed shader module");
        }
    }

    VkShaderModule Shader::handle() {
        return mModule;
    }

    std::string Shader::entryPoint() const {
        slang::FunctionReflection* reflection = mEntryPoint->getFunctionReflection();

        const char* entryPointName = reflection->getName();
        return std::string(entryPointName);
    }

    std::vector<VkPushConstantRange> Shader::pushConstants() const {
        uint32_t paramCount = mLayout->getParameterCount();

        std::vector<VkPushConstantRange> pushConstants = {};
        for (uint32_t i = 0; i < paramCount; i++) {
            slang::VariableLayoutReflection* varLayout = mLayout->getParameterByIndex(i);
            slang::TypeLayoutReflection* typeLayout = varLayout->getTypeLayout();

            slang::ParameterCategory category = varLayout->getCategory();
            if (category == slang::ParameterCategory::PushConstantBuffer) {
                size_t offset = varLayout->getOffset(slang::ParameterCategory::PushConstantBuffer);

                slang::TypeLayoutReflection* elementLayout = typeLayout->getElementTypeLayout();
                size_t size = elementLayout->getSize(slang::ParameterCategory::Uniform);

                VkPushConstantRange constant = {};
                constant.offset = offset;
                constant.size = size;
                constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

                pushConstants.push_back(constant);
            }
        }

        return pushConstants;
    }
}
