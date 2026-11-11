// Murmr's computer-audio helper for meeting notes on macOS 13+ (whispa/meetings.py).
//
// Captures what the Mac is playing (the other people in a Teams / Zoom / Meet call) with
// ScreenCaptureKit and writes it to stdout as raw float32 mono samples at --rate (default
// 48000). Only sound is used: the video stream is the smallest ScreenCaptureKit allows,
// one 2x2 frame a second, and it's dropped unread.
//
// macOS asks for "Screen & System Audio Recording" the first time. Murmr starts this
// helper itself, so the permission is Murmr's (see mac/launcher.c).
//
//   murmr-audiotap [--rate 48000] [--exclude-pid PID]   capture until stdin closes
//   murmr-audiotap --check                              print granted / denied
//   murmr-audiotap --request                            show macOS's prompt
//
// stderr: "started" once audio flows; "denied: ..." (exit 2) or "error: ..." (exit 1).
//
// Build (universal, see .github/workflows/mac-launcher.yml):
//   swiftc -O -target arm64-apple-macos13.0 -o a mac/audiotap.swift
//   swiftc -O -target x86_64-apple-macos13.0 -o b mac/audiotap.swift
//   lipo -create -output whispa/resources/murmr-audiotap a b
import CoreAudio
import CoreGraphics
import CoreMedia
import Foundation
import ScreenCaptureKit

func say(_ line: String) {
    FileHandle.standardError.write((line + "\n").data(using: .utf8)!)
}

func fail(_ line: String, _ code: Int32) -> Never {
    say(line)
    exit(code)
}

var rate = 48000
var excludePid: pid_t = 0
var mode = "capture"
var args = CommandLine.arguments.dropFirst()
while let arg = args.popFirst() {
    switch arg {
    case "--check": mode = "check"
    case "--request": mode = "request"
    case "--rate": rate = Int(args.popFirst() ?? "") ?? 48000
    case "--exclude-pid": excludePid = pid_t(args.popFirst() ?? "") ?? 0
    default: fail("error: unknown argument \(arg)", 64)
    }
}

if mode == "check" {
    print(CGPreflightScreenCaptureAccess() ? "granted" : "denied")
    exit(0)
}
if mode == "request" {
    print(CGRequestScreenCaptureAccess() ? "granted" : "denied")
    exit(0)
}

final class Tap: NSObject, SCStreamOutput, SCStreamDelegate {
    let out = FileHandle.standardOutput
    var announced = false

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sb.isValid, let asbd = sb.formatDescription?.audioStreamBasicDescription
        else { return }
        let frames = sb.numSamples
        let channels = max(1, Int(asbd.mChannelsPerFrame))
        let planar = asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0
        guard frames > 0, asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0, asbd.mBitsPerChannel == 32
        else { return }
        var mono = [Float](repeating: 0, count: frames)
        do {
            try sb.withAudioBufferList { list, _ in
                if planar || channels == 1 {
                    let used = list.prefix(channels)
                    for buf in used {
                        guard let data = buf.mData else { continue }
                        let p = data.assumingMemoryBound(to: Float.self)
                        for i in 0..<min(frames, Int(buf.mDataByteSize) / 4) { mono[i] += p[i] }
                    }
                    if used.count > 1 {
                        let scale = 1 / Float(used.count)
                        for i in 0..<frames { mono[i] *= scale }
                    }
                } else if let data = list.first?.mData {
                    let p = data.assumingMemoryBound(to: Float.self)
                    let n = min(frames, Int(list.first!.mDataByteSize) / 4 / channels)
                    for i in 0..<n {
                        var sum: Float = 0
                        for c in 0..<channels { sum += p[i * channels + c] }
                        mono[i] = sum / Float(channels)
                    }
                }
            }
        } catch {
            return
        }
        if !announced {
            announced = true
            if Int(asbd.mSampleRate) != rate { say("rate \(Int(asbd.mSampleRate))") }
        }
        mono.withUnsafeBytes { out.write(Data($0)) }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fail("error: stopped: \(error.localizedDescription)", 1)
    }
}

let tap = Tap()
var running: SCStream?

if !CGPreflightScreenCaptureAccess() {
    fail("denied: Screen & System Audio Recording isn't allowed for Murmr", 2)
}

Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first else { fail("error: no display found", 1) }
        // Murmr's own sounds (read aloud, chimes) aren't part of the meeting.
        let mine = content.applications.filter { $0.processID == excludePid || $0.processID == getpid() }
        let filter = SCContentFilter(display: display, excludingApplications: mine, exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = rate
        config.channelCount = 1
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        config.showsCursor = false
        let stream = SCStream(filter: filter, configuration: config, delegate: tap)
        try stream.addStreamOutput(tap, type: .audio, sampleHandlerQueue: DispatchQueue(label: "audio"))
        try stream.addStreamOutput(tap, type: .screen, sampleHandlerQueue: DispatchQueue(label: "screen"))
        try await stream.startCapture()
        running = stream
        say("started")
    } catch {
        let ns = error as NSError
        if ns.domain == SCStreamErrorDomain && ns.code == SCStreamError.Code.userDeclined.rawValue {
            fail("denied: \(error.localizedDescription)", 2)
        }
        fail("error: \(error.localizedDescription)", 1)
    }
}

// Stop when Murmr closes our stdin, or quits without closing it.
Thread.detachNewThread {
    while !FileHandle.standardInput.availableData.isEmpty {}
    exit(0)
}
signal(SIGTERM) { _ in exit(0) }
signal(SIGINT) { _ in exit(0) }
dispatchMain()
