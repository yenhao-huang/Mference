import Testing
import Foundation
import Metal
@testable import Mference
import MferenceValidationSupport

/// Validates the tied 4-bit embedding lookup used by Gemma 4.
@Suite struct EmbedLookupTests {

    private struct Sizes {
        static let V = 16
        static let D = 128
        static let groupsPerRow = D / Quantization.groupSize
    }

    private static func buildTable4(seed: UInt64)
        -> (packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
        var rng = SeedTree(seed).key("embed-lookup-int4-table")
        var rows: [[Float]] = []
        rows.reserveCapacity(Sizes.V)
        for _ in 0..<Sizes.V {
            rows.append((0..<Sizes.D).map { _ in rng.uniform(-1.0, 1.0) })
        }
        var packed = [UInt8]( repeating: 0, count: Sizes.V * (Sizes.D / 2))
        var scales = [UInt16](repeating: 0, count: Sizes.V * Sizes.groupsPerRow)
        var biases = [UInt16](repeating: 0, count: Sizes.V * Sizes.groupsPerRow)
        for v in 0..<Sizes.V {
            let q = Quantization.quantizeInt4Affine(rows[v])
            for i in 0..<(Sizes.D / 2) { packed[v * (Sizes.D / 2) + i] = q.packed[i] }
            for g in 0..<Sizes.groupsPerRow {
                scales[v * Sizes.groupsPerRow + g] = q.scales[g]
                biases[v * Sizes.groupsPerRow + g] = q.biases[g]
            }
        }
        return (packed, scales, biases)
    }

    @Test func embedLookupInt4_withSqrtDScale_matchesReference() throws {
        let (packed, scales, biases) = Self.buildTable4(seed: 0x131)
        let ctx = try MetalContext()
        let kernel = try EmbedLookupInt4(context: ctx)

        guard let tableBuf = ctx.device.makeBuffer(
                bytes: packed, length: packed.count,
                options: .storageModeShared),
              let scalesBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let biasesBuf = ctx.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: Sizes.D) else {
            Issue.record("alloc failed"); return
        }
        let token: UInt32 = 9
        // sqrt(D) for the toy D=128 (5.65...). Mirrors the per-model sqrt(H)
        // scale; the kernel treats it as a runtime float.
        let outScale = Float(Sizes.D).squareRoot()

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      table: tableBuf, scales: scalesBuf, biases: biasesBuf,
                      out: outBuf,
                      tokenId: token, d: UInt32(Sizes.D),
                      outScale: outScale)
        cb.commit(); cb.waitUntilCompleted()

        let ref = EmbedLookupRef.applyInt4(
            tablePacked: packed, tableScales: scales, tableBiases: biases,
            tokenId: Int(token), d: Sizes.D, outScale: outScale)
        let actual = Fp16Buffer.read(outBuf, count: Sizes.D)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.quantInt4, "rel=\(rel)")
    }

    /// outScale=1.0 must produce raw dequant (sanity for the disable path).
    @Test func embedLookupInt4_unitScale_matchesRawDequant() throws {
        let (packed, scales, biases) = Self.buildTable4(seed: 0x132)
        let ctx = try MetalContext()
        let kernel = try EmbedLookupInt4(context: ctx)

        guard let tableBuf = ctx.device.makeBuffer(
                bytes: packed, length: packed.count,
                options: .storageModeShared),
              let scalesBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let biasesBuf = ctx.device.makeBuffer(
                bytes: biases, length: biases.count * MemoryLayout<UInt16>.size,
                options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: Sizes.D) else {
            Issue.record("alloc failed"); return
        }
        let token: UInt32 = 2
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      table: tableBuf, scales: scalesBuf, biases: biasesBuf,
                      out: outBuf,
                      tokenId: token, d: UInt32(Sizes.D), outScale: 1.0)
        cb.commit(); cb.waitUntilCompleted()

        let ref = EmbedLookupRef.applyInt4(
            tablePacked: packed, tableScales: scales, tableBiases: biases,
            tokenId: Int(token), d: Sizes.D, outScale: 1.0)
        let actual = Fp16Buffer.read(outBuf, count: Sizes.D)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.quantInt4, "rel=\(rel)")
    }

    @Test func embedLookupInt8_matchesScalarReference() throws {
        let vocab = 5
        let d = 128
        let groups = d / 64
        var rng = SeedTree(0x8_0731).key("embed-lookup-int8")
        let packed = (0..<(vocab * d)).map { _ in UInt8(truncatingIfNeeded: rng.next()) }
        let scales = (0..<(vocab * groups)).map { _ in
            Quantization.bf16Bits(rng.uniform(0.002, 0.01))
        }
        let biases = (0..<(vocab * groups)).map { _ in
            Quantization.bf16Bits(rng.uniform(-0.2, 0.0))
        }
        let token = 3
        let outScale: Float = 0.75
        let reference = (0..<d).map { i -> Float in
            let aux = token * groups + i / 64
            return (Float(packed[token * d + i])
                    * Quantization.bf16ToFloat(scales[aux])
                    + Quantization.bf16ToFloat(biases[aux])) * outScale
        }
        let ctx = try MetalContext()
        let kernel = try EmbedLookupInt8(context: ctx)
        let w = try #require(ctx.device.makeBuffer(
            bytes: packed, length: packed.count, options: .storageModeShared))
        let s = try #require(ctx.device.makeBuffer(
            bytes: scales, length: scales.count * 2, options: .storageModeShared))
        let b = try #require(ctx.device.makeBuffer(
            bytes: biases, length: biases.count * 2, options: .storageModeShared))
        let y = try #require(Fp16Buffer.make(ctx.device, count: d))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, table: w, scales: s, biases: b,
                      out: y, tokenId: UInt32(token), d: UInt32(d),
                      outScale: outScale)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let actual = Fp16Buffer.read(y, count: d)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < Tolerance.quantInt8, "INT8 embed rel=\(rel)")
    }
}
