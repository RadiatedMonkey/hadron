static const uint WAVE_LANE_COUNT = 32;
static const uint WORKGROUP_SIZE = 256;
static const uint WAVES_PER_WORKGROUP = (WORKGROUP_SIZE + WAVE_LANE_COUNT - 1) / WAVE_LANE_COUNT;

typedef double FScalar;

struct PushConstants {
    FScalar *buffer;
    uint bufferSize;
    uint dispatched;
}

[vk::push_constant]
PushConstants params;

groupshared FScalar sdata[WAVES_PER_WORKGROUP];

/// Computes the partial mean per workgroup.
///
/// This function writes the partial means into `params.buffer` with the workgroup index
/// representing the index into the buffer.
///
/// If too few workgroups were dispatched to adequately cover the full buffer,
/// individual threads will reduce multiple sites.
///
/// If multiple workgroups were dispatched, this still needs to be reduced further to obtain
/// a single digit.
void computeWorkgroupMeans(
    uint dispatchThreadID, // The ID of the current thread
    uint groupThreadID,
    FScalar *buffer,
    uint bufferSize
) {
    if (dispatchThreadID >= bufferSize) {
        return;
    }

    uint stride = params.dispatched;

    // ==============================================================================
    // Reduce within the current wave. This reduces by 32x (on NVIDIA/newer AMD cards)
    //
    // If too little workgroups were dispatched to cover the entire buffer,
    // singular threads will reduce multiple sites.

    FScalar reduction = 0.0f;
    for (uint i = dispatchThreadID; i < bufferSize; i += stride) {
        if (i >= bufferSize) {
            break;
        }

        reduction += buffer[i];
    }

    reduction = WaveActiveSum(reduction);
    uint waveIndex = groupThreadID / WaveGetLaneCount();

    if (WaveIsFirstLane()) {
        sdata[waveIndex] = reduction;
    }

    GroupMemoryBarrierWithGroupSync(); // Wait for all waves to push their means

    // ==============================================================================
    // Reduce within the current group. This reduces by `WAVES_PER_WORKGROUP`.

    static_assert(WAVES_PER_WORKGROUP <= 32, "Cannot reduce further, too many waves in workgroup");

    // Only the first wave within the group gets to reduce.
    if (waveIndex == 0) {
        uint lane = WaveGetLaneIndex();

        reduction = (lane < WaveGetNumWaves()) ? sdata[lane] : 0.0f;
        reduction = WaveActiveSum(reduction);

        if (lane == 0) {
            // Divide by `WORKGROUP_SIZE` to compute the average of this workgroup
            buffer[dispatchThreadID.x / WORKGROUP_SIZE] = reduction / WORKGROUP_SIZE;
            printf("wrote %lf\n", reduction / WORKGROUP_SIZE);
        }
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID: SV_DispatchThreadID,
    uint3 groupThreadID: SV_GroupThreadID
) {
    computeWorkgroupMeans(dispatchThreadID.x, groupThreadID.x, params.buffer, params.bufferSize);
}
