typedef double Scalar;

struct PushConstants {
    Scalar* inputBuffer;
    Scalar* outputBuffer;
    uint inputBufferSize;
    uint dispatchedThreads;
}

[[vk::push_constant]]
PushConstants params;

static const uint WORKGROUP_SIZE = 256;
static const uint MIN_WAVE_SIZE = 32;
static const uint MAX_WAVES_PER_WORKGROUP = WORKGROUP_SIZE / MIN_WAVE_SIZE;

groupshared Scalar sdata[MAX_WAVES_PER_WORKGROUP];

void computeWorkgroupMeans(
    uint dispatchThreadID, uint groupThreadID, uint groupID,
    Scalar* inBuffer, Scalar* outBuffer, uint bufferSize
) {
    // Perform reduction per wave.

    uint reductionsPerThread = bufferSize / params.dispatchedThreads;
    uint stride = params.dispatchedThreads;

    Scalar curr = 0.0f;
    for (uint i = dispatchThreadID; i < bufferSize; i += stride) {
        curr += inBuffer[i];
    }

    uint waveIndex = groupThreadID / WaveGetLaneCount();

    curr = WaveActiveSum(curr);
    if (WaveIsFirstLane()) {
        sdata[waveIndex] = curr;
    }

    GroupMemoryBarrierWithGroupSync();

    // Perform reduction on entire workgroup
    // Done by wave 0

    if (waveIndex == 0) {
        curr = (WaveGetLaneIndex() < WaveGetNumWaves()) ? sdata[WaveGetLaneIndex()] : 0.0;
        curr = WaveActiveSum(curr);

        if (WaveIsFirstLane()) {
            outBuffer[groupID.x] = curr  / (WORKGROUP_SIZE * reductionsPerThread);
        }
    }
}

[[shader("compute")]]
[[numthreads(WORKGROUP_SIZE, 1, 1)]]
void main(
    uint3 dispatchThreadID: SV_DispatchThreadID,
    uint3 groupThreadID: SV_GroupThreadID,
    uint3 groupID: SV_GroupID
) {
    computeWorkgroupMeans(
        dispatchThreadID.x, groupThreadID.x, groupID.x,
        params.inputBuffer, params.outputBuffer, params.inputBufferSize
    );
}
