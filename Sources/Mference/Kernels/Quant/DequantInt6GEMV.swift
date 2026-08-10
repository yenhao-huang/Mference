import Metal

/// MLX-affine Q6 matrix-vector multiplication. OptiQ uses group 128.
final class DequantInt6GEMV {
    private static let rowsPerThreadgroup = 8
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try context.pipeline(
            "dequant_int6_gemv_simd",
            constants: [],
            maxTotalThreadsPerThreadgroup: 256)
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: UInt32, n: UInt32,
                groupSize: UInt32) {
        precondition(groupSize == 128)
        precondition(n.isMultiple(of: groupSize))
        precondition((n * 6).isMultiple(of: 32))
        precondition(weightsOffset.isMultiple(of: 4))
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(biases, offset: biasesOffset, index: 2)
        encoder.setBuffer(x, offset: xOffset, index: 3)
        encoder.setBuffer(y, offset: yOffset, index: 4)
        var rows = m
        var columns = n
        var group = groupSize
        encoder.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&columns, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&group, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }
}
