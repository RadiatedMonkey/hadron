static const uint WAVE_LANE_COUNT = 32;
static const uint WORKGROUP_SIZE = 256;
static const uint WAVES_PER_WORKGROUP = (WORKGROUP_SIZE + WAVE_LANE_COUNT - 1) / WAVE_LANE_COUNT;

typedef double FScalar;

struct PushConstants {
    FScalar* buffer;
    uint bufferSize;
    uint dispatched;
}

[[vk::push_constant]]
PushConstants params;

groupshared FScalar sdata[WAVES_PER_WORKGROUP];

void computeWorkgroupVariances(
    uint dispatchThreadID,
    uint groupThreadID,
    FScalar* buffer,
    uint bufferSize
) {

}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID: SV_DispatchThreadID,
    uint3 groupThreadID: SV_GroupThreadID
) {

}
