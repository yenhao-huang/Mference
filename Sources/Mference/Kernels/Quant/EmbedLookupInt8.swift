import Metal

final class EmbedLookupInt8 {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline("embed_lookup_int8")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                table: MTLBuffer, tableOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                out: MTLBuffer, outOffset: Int = 0,
                tokenId: UInt32, d: UInt32, outScale: Float) {
        precondition(d.isMultiple(of: 64))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var token = tokenId
        var dimension = d
        var scale = outScale
        encoder.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.maxTotalThreadsPerThreadgroup, 256),
                height: 1, depth: 1))
        encoder.endEncoding()
    }
}
