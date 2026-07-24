import xoshiro;

static const uint WORKGROUP_SIZE = 256;

struct PushConstants {
    uint64_t seed;
    uint numCount;
    uint dispatched;
    double *outBuffer;
}

[vk::push_constant]
PushConstants params;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dispatchID: SV_DispatchThreadID) {
    if (dispatchID.x >= params.numCount) {
        return;
    }

    XoshiroRng rng = XoshiroRng(dispatchID, params.seed);
    for (uint i = dispatchID.x; i < params.numCount; i += params.dispatched) {
        if (i >= params.numCount) {
            break;
        }

        params.outBuffer[i] = rng.nextDouble();
    }
}
