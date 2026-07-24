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

[[shader("compute")]]
[[numthreads(WORKGROUP_SIZE, 1, 1)]]
void main(
    uint3 dispatchThreadID: SV_DispatchThreadID,
    uint3 groupThreadID: SV_GroupThreadID,
    uint3 groupID: SV_GroupID
) {
    // =================================================================
    // Reduction on local waves

    uint stride = params.dispatchedThreads;

    Scalar curr = 0.0f;
    for (uint i = dispatchThreadID.x; i < params.inputBufferSize; i += stride) {
        curr += params.inputBuffer[i];
    }

    uint waveIndex = groupThreadID.x / WaveGetLaneCount();

    curr = WaveActiveSum(curr);
    if (WaveIsFirstLane()) {
        sdata[waveIndex] = curr;
    }

    GroupMemoryBarrierWithGroupSync();

    // ================================================================
    // Reduction on entire group (done by first wave in the group)

    if (waveIndex == 0) {
        curr = (WaveGetLaneIndex() < WaveGetNumWaves()) ? sdata[WaveGetLaneIndex()] : 0.0;
        curr = WaveActiveSum(curr);

        if (WaveIsFirstLane()) {
            params.outputBuffer[groupID.x] = curr / WORKGROUP_SIZE;
        }
    }
}
