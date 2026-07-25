typedef double Scalar;

struct PushConstants {
    Scalar* inputBuffer;
    Scalar* meanBuffer;
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

void computeWorkgroupVariances(
    uint dispatchThreadID, uint groupThreadID, uint groupID,
    Scalar bufferMean, Scalar* inBuffer, Scalar* outBuffer, uint bufferSize
) {
    // Perform reduction per wave.

    uint stride = params.dispatchedThreads;

    Scalar curr = 0.0f;
    for (uint i = dispatchThreadID; i < bufferSize; i += stride) {
        Scalar val = inBuffer[i];
        curr += (val - bufferMean) * (val - bufferMean);
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
            outBuffer[groupID.x] = curr;
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
    Scalar totalMean = 0.0;

    // We received workgroup means from the means pipeline.
    // These still need to be averaged together to be usable.

    uint stride = WaveGetLaneCount();
    uint workgroupCount = params.dispatchedThreads / WORKGROUP_SIZE;

    for (uint i = WaveGetLaneIndex(); i < workgroupCount; i += stride) {
        totalMean += params.meanBuffer[i];
    }

    totalMean = WaveActiveSum(totalMean) / workgroupCount;
    computeWorkgroupVariances(
        dispatchThreadID.x, groupThreadID.x, groupID.x, totalMean,
        params.inputBuffer, params.outputBuffer, params.inputBufferSize
    );
}
