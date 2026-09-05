// visionocr — batch text recognition via Apple's Vision framework.
//
// Reads image paths (argv or stdin, one per line) and writes one JSON object per
// line to stdout: the recognised text lines with normalised, top-left-origin
// boxes plus the pixel height of each box.
//
// Why this exists: the vision model downscales anything longer than 2576px,
// which destroys the small article line that names the wine on a price tag.
// Vision runs locally and free, so it can be pointed at the native 5712px frame
// — but Vision downscales internally too, so the work is in tiling the frame
// small enough that the small print survives.
//
// Build:  swiftc -O tools/visionocr.swift -o tools/visionocr

import Foundation
import CoreGraphics
import CoreImage
import ImageIO
import Vision

// --------------------------------------------------------------------------
// Options
// --------------------------------------------------------------------------

struct Options {
    var languages: [String] = ["ro-RO", "en-US"]
    var correction = true
    var fast = false
    // Longest edge of a tile handed to Vision. Vision resamples its input, so
    // this is the real resolution knob: smaller tiles = more effective pixels
    // per character. 0 disables tiling.
    var maxTile = 0
    var overlap = 0.15
    var upscale = 1.0
    var minHeight: Float = 0.0
    var rect: [Double]? = nil          // x0,y0,x1,y1 normalised, top-left origin
    var paths: [String] = []
    var listLanguages = false
    var dumpTo: String? = nil          // write the oriented image here (debug)
}

func parseArgs() -> Options {
    var o = Options()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--languages": o.languages = (it.next() ?? "").split(separator: ",").map(String.init)
        case "--no-correction": o.correction = false
        case "--fast": o.fast = true
        case "--max-tile": o.maxTile = Int(it.next() ?? "0") ?? 0
        case "--overlap": o.overlap = Double(it.next() ?? "0.15") ?? 0.15
        case "--upscale": o.upscale = Double(it.next() ?? "1") ?? 1
        case "--min-height": o.minHeight = Float(it.next() ?? "0") ?? 0
        case "--rect":
            o.rect = (it.next() ?? "").split(separator: ",").compactMap { Double($0) }
        case "--dump": o.dumpTo = it.next()
        case "--list-languages": o.listLanguages = true
        case "-": break  // read stdin
        default: o.paths.append(a)
        }
    }
    return o
}

let ciContext = CIContext(options: [.useSoftwareRenderer: false])

// --------------------------------------------------------------------------
// Image loading
// --------------------------------------------------------------------------

/// A CGImage with EXIF orientation baked in, so every coordinate reported here
/// matches what a human sees and what PIL's `exif_transpose` produces.
///
/// Core Image's `oriented(forExifOrientation:)` does the transform; hand-rolling
/// the affine table is a reliable way to end up 180 degrees out, because
/// CGContext's origin is bottom-left while EXIF's is top-left.
func loadOriented(_ path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let raw = CGImageSourceCreateImageAtIndex(
              src, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
    else { return nil }

    let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
    let orientation = (props?[kCGImagePropertyOrientation] as? Int32) ?? 1
    if orientation == 1 { return raw }

    let ci = CIImage(cgImage: raw).oriented(forExifOrientation: orientation)
    return ciContext.createCGImage(ci, from: ci.extent) ?? raw
}

func scaled(_ img: CGImage, by factor: Double) -> CGImage {
    if abs(factor - 1.0) < 0.001 { return img }
    let w = Int((Double(img.width) * factor).rounded())
    let h = Int((Double(img.height) * factor).rounded())
    guard let ctx = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else { return img }
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage() ?? img
}

func writePNG(_ img: CGImage, to path: String) {
    guard let dest = CGImageDestinationCreateWithURL(
        URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

// --------------------------------------------------------------------------
// Recognition
// --------------------------------------------------------------------------

struct Line {
    var text: String
    var conf: Float
    var x0: Double, y0: Double, x1: Double, y1: Double   // 0-1, origin top-left
}

/// Recognise one tile and map its boxes back into the frame's 0-1 coordinates.
///
/// The tile is a real crop, not a `regionOfInterest`: an ROI is resampled out of
/// Vision's downscaled copy of the whole frame, which is exactly the loss this
/// is here to avoid.
func recognize(_ img: CGImage, region: CGRect, o: Options) -> [Line] {
    guard var crop = img.cropping(to: region), crop.width > 8, crop.height > 8 else { return [] }
    crop = scaled(crop, by: o.upscale)

    let req = VNRecognizeTextRequest()
    req.recognitionLevel = o.fast ? .fast : .accurate
    req.usesLanguageCorrection = o.correction
    req.minimumTextHeight = o.minHeight
    if !o.languages.isEmpty { req.recognitionLanguages = o.languages }
    if #available(macOS 13.0, *) { req.revision = VNRecognizeTextRequestRevision3 }

    // .up tells Vision the crop is already upright, so it stops guessing an
    // orientation per tile — a narrow strip of digits is otherwise read upside
    // down often enough to matter.
    let handler = VNImageRequestHandler(cgImage: crop, orientation: .up, options: [:])
    do { try handler.perform([req]) } catch { return [] }

    let imgW = Double(img.width), imgH = Double(img.height)
    let rx = Double(region.origin.x), ry = Double(region.origin.y)
    let rw = Double(region.width), rh = Double(region.height)

    var out: [Line] = []
    for obs in (req.results ?? []) {
        guard let best = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox           // 0-1 within the crop, origin BOTTOM-left
        let ax0 = rx + b.minX * rw
        let ax1 = rx + b.maxX * rw
        let ay0 = ry + (1 - b.maxY) * rh  // flip to a top-left origin
        let ay1 = ry + (1 - b.minY) * rh
        out.append(Line(
            text: best.string, conf: best.confidence,
            x0: ax0 / imgW, y0: ay0 / imgH, x1: ax1 / imgW, y1: ay1 / imgH
        ))
    }
    return out
}

func offsets(_ length: Int, _ budget: Int, _ overlap: Double) -> [(Int, Int)] {
    if budget <= 0 || length <= budget { return [(0, length)] }
    var n = 2
    var span = Double(length)
    while true {
        span = Double(length) / (Double(n) - Double(n - 1) * overlap)
        if span <= Double(budget) || n > 40 { break }
        n += 1
    }
    let step = (Double(length) - span) / Double(n - 1)
    return (0..<n).map { i in
        let s = Int((Double(i) * step).rounded())
        return (s, min(length, s + Int(span.rounded())))
    }
}

/// Overlapping tiles no longer than `maxTile` on either axis.
func tileRegions(_ img: CGImage, o: Options) -> [CGRect] {
    let cols = offsets(img.width, o.maxTile, o.overlap)
    let rows = offsets(img.height, o.maxTile, o.overlap)
    var out: [CGRect] = []
    for (y0, y1) in rows {
        for (x0, x1) in cols {
            out.append(CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
        }
    }
    return out
}

/// Overlapping tiles see the same text twice. Keep the more confident reading;
/// treat two boxes as the same line when they overlap heavily.
func dedupe(_ lines: [Line]) -> [Line] {
    func iou(_ a: Line, _ b: Line) -> Double {
        let ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
        let iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
        let inter = ix * iy
        let ua = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
        return ua <= 0 ? 0 : inter / ua
    }
    var kept: [Line] = []
    // Longer text first, then confidence: a tile that caught the whole line
    // beats a neighbour that caught half of it.
    for l in lines.sorted(by: { ($0.text.count, $0.conf) > ($1.text.count, $1.conf) }) {
        if !kept.contains(where: { iou($0, l) > 0.45 }) { kept.append(l) }
    }
    return kept.sorted { ($0.y0, $0.x0) < ($1.y0, $1.x0) }
}

// --------------------------------------------------------------------------
// Output
// --------------------------------------------------------------------------

func jsonString(_ s: String) -> String {
    let data = try! JSONSerialization.data(withJSONObject: [s], options: [])
    var t = String(data: data, encoding: .utf8)!
    t.removeFirst(); t.removeLast()
    return t
}

func round4(_ v: Double) -> String { String(format: "%.4f", v) }

func emit(path: String, w: Int, h: Int, lines: [Line], error: String?) {
    var parts: [String] = ["\"path\":\(jsonString(path))"]
    if let e = error {
        parts.append("\"error\":\(jsonString(e))")
        parts.append("\"lines\":[]")
    } else {
        parts.append("\"width\":\(w)")
        parts.append("\"height\":\(h)")
        let items = lines.map { l -> String in
            let hpx = Int(((l.y1 - l.y0) * Double(h)).rounded())
            return "{\"t\":\(jsonString(l.text)),\"c\":\(String(format: "%.3f", l.conf))," +
                   "\"b\":[\(round4(l.x0)),\(round4(l.y0)),\(round4(l.x1)),\(round4(l.y1))],\"h\":\(hpx)}"
        }
        parts.append("\"lines\":[\(items.joined(separator: ","))]")
    }
    print("{\(parts.joined(separator: ","))}")
    fflush(stdout)
}

// --------------------------------------------------------------------------

let opts = parseArgs()

if opts.listLanguages {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    if #available(macOS 13.0, *) { req.revision = VNRecognizeTextRequestRevision3 }
    print(((try? req.supportedRecognitionLanguages()) ?? []).joined(separator: "\n"))
    exit(0)
}

var paths = opts.paths
if paths.isEmpty {
    while let line = readLine(strippingNewline: true), !line.isEmpty { paths.append(line) }
}

for path in paths {
    guard let full = loadOriented(path) else {
        emit(path: path, w: 0, h: 0, lines: [], error: "could not decode image")
        continue
    }
    // A --rect crop becomes the frame: boxes are reported against it, so the
    // caller works in one coordinate space per call.
    var img = full
    if let r = opts.rect, r.count == 4 {
        let rect = CGRect(
            x: r[0] * Double(full.width), y: r[1] * Double(full.height),
            width: (r[2] - r[0]) * Double(full.width),
            height: (r[3] - r[1]) * Double(full.height)
        ).integral
        img = full.cropping(to: rect) ?? full
    }
    if let dump = opts.dumpTo { writePNG(img, to: dump) }

    var all: [Line] = []
    for region in tileRegions(img, o: opts) {
        all.append(contentsOf: recognize(img, region: region, o: opts))
    }
    emit(path: path, w: img.width, h: img.height, lines: dedupe(all), error: nil)
}
