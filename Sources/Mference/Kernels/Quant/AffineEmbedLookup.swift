import Metal

final class AffineEmbedLookup {
    private enum Implementation {
        case int4(EmbedLookupInt4)
        case int8(EmbedLookupInt8)
    }

    private let implementation: Implementation

    init(context: MetalContext, weightBits: Int, groupSize: Int) throws {
        switch (weightBits, groupSize) {
        case (4, 64): implementation = .int4(try EmbedLookupInt4(context: context))
        case (8, 64): implementation = .int8(try EmbedLookupInt8(context: context))
        default:
            throw ModelError.indexCorrupt(
                detail: "unsupported embedding Q\(weightBits)/group-\(groupSize)")
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,
                table: MTLBuffer, tableOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                out: MTLBuffer, outOffset: Int = 0,
                tokenId: UInt32, d: UInt32, outScale: Float) {
        switch implementation {
        case .int4(let kernel):
            kernel.encode(commandBuffer: commandBuffer,
                          table: table, tableOffset: tableOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          out: out, outOffset: outOffset,
                          tokenId: tokenId, d: d, outScale: outScale)
        case .int8(let kernel):
            kernel.encode(commandBuffer: commandBuffer,
                          table: table, tableOffset: tableOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          out: out, outOffset: outOffset,
                          tokenId: tokenId, d: d, outScale: outScale)
        }
    }
}
