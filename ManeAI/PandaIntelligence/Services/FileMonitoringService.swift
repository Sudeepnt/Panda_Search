import Foundation
import Combine
import CoreServices

/// Watches indexed roots with FSEvents and re-indexes changed supported files.
/// This keeps the local memory current without repeatedly rescanning the disk.
@MainActor
final class FileMonitoringService: ObservableObject {
    @Published private(set) var watchedRoots: [URL] = []
    @Published private(set) var lastUpdate: String?

    private let indexingService: DocumentIndexingService
    private var stream: FSEventStreamRef?
    private var pendingPaths = Set<String>()
    private var debounceTask: Task<Void, Never>?
    private let storedRootsKey = "PandaIntelligence.WatchedLibraryRoots"
    private let supportedExtensions: Set<String> = [
        "txt", "md", "markdown", "csv", "tsv", "pdf", "doc", "docx", "pages", "odt", "rtf", "rtfd", "epub", "mobi",
        "xlsx", "xls", "xlsm", "numbers", "ods", "ppt", "pptx", "key", "odp",
        "json", "jsonl", "ndjson", "xml", "yaml", "yml", "toml", "ini", "conf", "config", "env", "log", "sql",
        "html", "htm", "css", "scss", "less", "js", "jsx", "ts", "tsx", "mjs", "cjs", "py", "pyw", "swift",
        "m", "h", "c", "cc", "cpp", "cxx", "hpp", "java", "kt", "kts", "go", "rs", "rb", "php", "sh", "bash", "zsh", "fish", "graphql", "gql", "tex",
        "svelte", "vue", "astro", "cu", "cuh", "wgsl", "glsl", "metal", "plist", "strings", "stringsdata", "cmake", "mk", "make", "lock", "properties", "pbxproj", "entitlements", "ps1", "bat", "vim", "nix", "jinja", "jinja2", "template", "tmpl", "example", "inp", "dia", "d", "eml", "msg", "vcf", "ics",
        "png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "tiff", "bmp", "svg", "ico", "icns", "avif", "raw", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "raf", "pef", "srw",
        "mp3", "wav", "m4a", "aac", "aiff", "flac", "ogg", "wma",
        "zip", "rar", "7z", "tar", "gz", "tgz", "bz2", "xz", "iso", "dmg", "pkg", "sqlite", "db", "bin", "dat", "psd", "ai", "eps", "sketch", "blend", "fig", "obj", "stl", "fbx", "dwg", "jar"
    ]

    init(indexingService: DocumentIndexingService) {
        self.indexingService = indexingService
    }

    deinit {
        if let stream {
            FSEventStreamStop(stream)
            FSEventStreamInvalidate(stream)
            FSEventStreamRelease(stream)
        }
    }

    func restorePersistedWatches() {
        let paths = UserDefaults.standard.stringArray(forKey: storedRootsKey) ?? []
        // A previous development build could persist "/" as a watch root.
        // Never restore that broad watch: it captures temporary screenshots,
        // caches, and build artifacts instead of the user's chosen library.
        let filteredPaths = paths.filter { $0 != "/" }
        if filteredPaths.count != paths.count {
            UserDefaults.standard.set(filteredPaths, forKey: storedRootsKey)
        }
        let roots = filteredPaths.map { URL(fileURLWithPath: $0, isDirectory: true) }
        startWatching(roots, persist: false)
    }

    func startWatching(_ root: URL, persist: Bool = true) {
        var roots = watchedRoots
        if !roots.contains(root) { roots.append(root) }
        startWatching(roots, persist: persist)
    }

    func startWatching(_ roots: [URL], persist: Bool = true) {
        stopWatching()
        watchedRoots = Array(Set(roots)).sorted { $0.path < $1.path }
        guard !watchedRoots.isEmpty else { return }

        if persist {
            UserDefaults.standard.set(watchedRoots.map(\.path), forKey: storedRootsKey)
        }

        let paths = watchedRoots.map(\.path) as CFArray
        var context = FSEventStreamContext(
            version: 0,
            info: Unmanaged.passUnretained(self).toOpaque(),
            retain: nil,
            release: nil,
            copyDescription: nil
        )

        stream = FSEventStreamCreate(
            kCFAllocatorDefault,
            fileEventsCallback,
            &context,
            paths,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            1.5,
            FSEventStreamCreateFlags(kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagUseCFTypes)
        )

        guard let stream else { return }
        FSEventStreamSetDispatchQueue(stream, DispatchQueue(label: "com.reelsynth.pandaintelligence.file-events"))
        FSEventStreamStart(stream)
        lastUpdate = "Watching \(watchedRoots.count) library location\(watchedRoots.count == 1 ? "" : "s")"
    }

    func stopWatching() {
        debounceTask?.cancel()
        debounceTask = nil
        pendingPaths.removeAll()
        if let stream {
            FSEventStreamStop(stream)
            FSEventStreamInvalidate(stream)
            FSEventStreamRelease(stream)
        }
        stream = nil
    }

    fileprivate func receive(paths: [String]) {
        for path in paths {
            let url = URL(fileURLWithPath: path)
            guard supportedExtensions.contains(url.pathExtension.lowercased()),
                  FileManager.default.fileExists(atPath: path) else { continue }
            pendingPaths.insert(path)
        }
        guard !pendingPaths.isEmpty else { return }

        debounceTask?.cancel()
        debounceTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            guard !Task.isCancelled, let self else { return }
            let paths = self.pendingPaths
            self.pendingPaths.removeAll()
            self.lastUpdate = "Updating \(paths.count) changed file\(paths.count == 1 ? "" : "s")"
            for path in paths {
                _ = await self.indexingService.indexFileIfNeeded(URL(fileURLWithPath: path))
            }
            self.lastUpdate = "Library is up to date"
        }
    }
}

private let fileEventsCallback: FSEventStreamCallback = { _, info, eventCount, eventPaths, _, _ in
    guard let info else { return }
    let monitor = Unmanaged<FileMonitoringService>.fromOpaque(info).takeUnretainedValue()
    let paths = unsafeBitCast(eventPaths, to: NSArray.self) as? [String] ?? []
    let relevantPaths = Array(paths.prefix(Int(eventCount)))
    Task { @MainActor in
        monitor.receive(paths: relevantPaths)
    }
}
