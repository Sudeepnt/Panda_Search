import SwiftUI
import AppKit
import SwiftData
import ImageIO

/// The normal Panda Intelligence app window. Ctrl+W still opens the compact quick-search overlay.
struct ContentView: View {
    private static let currentFullContentIndexVersion = 2
    @EnvironmentObject var apiService: APIService
    @EnvironmentObject var indexingService: DocumentIndexingService
    @EnvironmentObject var fileMonitoringService: FileMonitoringService

    @State private var prompt = ""
    @State private var history: [String] = []
    @State private var results: [SearchResult] = []
    @State private var isSearching = false
    @State private var didSearch = false
    @State private var isIndexingLibrary = false
    @State private var libraryStatus: String?
    @State private var brainStatus = "Panda Intelligence is ready for your task"
    @State private var isShowingImageIndex = false
    // This is intentionally completion-based: an app restart or interrupted
    // scan must resume on the next prompt instead of permanently looking done.
    @AppStorage("PandaIntelligence.HasCompletedLibraryIndex") private var hasCompletedLibraryIndex = false
    // Versioned separately so existing filename/OCR-only media gets replaced
    // once with rich Qwen-VL descriptions without redoing text documents.
    @AppStorage("PandaIntelligence.QwenVisualIndexVersion") private var qwenVisualIndexVersion = 0
    // Bump when the supported non-video file set or extraction pipeline grows.
    // Version 2 adds camera RAW, design/container metadata, and additional
    // source/config/mail/calendar formats without re-running the VLM for
    // unchanged images.
    @AppStorage("PandaIntelligence.FullContentIndexVersion") private var fullContentIndexVersion = 0
    @AppStorage("PandaIntelligence.ImageScanSessionPrepared") private var imageScanSessionPrepared = false

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider().overlay(.white.opacity(0.08))
            workspace
        }
        .frame(minWidth: 1000, minHeight: 680)
        .background(PandaPalette.canvas)
        .preferredColorScheme(.dark)
        .task {
            // Reconcile the derived table on every launch. This is cheap for
            // the local metadata store and removes deleted, hidden, excluded,
            // or older duplicate rows without touching Finder files.
            indexingService.deduplicateImageRecords()
            indexingService.pruneMissingImageRecords()
            // The upgrade is automatic: after Full Disk Access has been
            // granted, Panda rebuilds only image/video records with Qwen.
            if qwenVisualIndexVersion < 1 || fullContentIndexVersion < Self.currentFullContentIndexVersion { startFullDiskScan() }
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                pandaAvatar(52)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Panda Intelligence").font(.system(size: 20, weight: .bold, design: .rounded))
                    Text("PRIVATE FILE INTELLIGENCE")
                        .font(.system(size: 9, weight: .bold)).tracking(1.1).foregroundStyle(PandaPalette.mint)
                }
            }
            .padding(.top, 42)

            Button(action: resetTask) {
                Label("New Task", systemImage: "plus")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(maxWidth: .infinity).padding(.vertical, 17)
                    .background(PandaPalette.panel, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 16).stroke(.white.opacity(0.14)))
            }
            .buttonStyle(.plain).padding(.top, 44)

            Button(action: addLibraryFolder) {
                Label("Add Library Folder", systemImage: "folder.badge.plus")
                    .font(.system(size: 13, weight: .semibold))
            }
            .buttonStyle(.plain)
            .foregroundStyle(PandaPalette.mint)
            .padding(.top, 18)

            Button {
                isShowingImageIndex = true
                didSearch = false
            } label: {
                Label("Image Index", systemImage: "tablecells")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .foregroundStyle(isShowingImageIndex ? PandaPalette.mint : .white.opacity(0.76))
            .padding(.vertical, 8)
            .padding(.horizontal, 10)
            .background(isShowingImageIndex ? PandaPalette.panel : .clear, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .padding(.top, 8)

            Button(action: startFullDiskScan) {
                Label(isIndexingLibrary ? "Scanning Local Mac…" : "Scan Local Mac", systemImage: "internaldrive")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.white.opacity(0.76))
            .padding(.vertical, 8)
            .padding(.horizontal, 10)
            .background(PandaPalette.panel.opacity(0.5), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .padding(.top, 12)
            .disabled(isIndexingLibrary)

            Text("HISTORY")
                .font(.system(size: 11, weight: .semibold)).tracking(1).foregroundStyle(.white.opacity(0.35))
                .padding(.top, 46)

            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(history.indices.reversed(), id: \.self) { index in
                        HStack(spacing: 11) {
                            Image(systemName: "clock").foregroundStyle(PandaPalette.mint.opacity(0.85))
                            Text(history[index]).lineLimit(2).font(.system(size: 13, weight: .medium))
                            Spacer(minLength: 0)
                            Circle().fill(PandaPalette.mint).frame(width: 7, height: 7)
                        }
                        .padding(13)
                        .background(PandaPalette.panel.opacity(0.7), in: RoundedRectangle(cornerRadius: 14))
                    }
                }.padding(.top, 16)
            }
            .scrollIndicators(.hidden)

            Spacer()
            HStack(spacing: 10) {
                Circle().fill(PandaPalette.mint.opacity(0.35)).frame(width: 40, height: 40).overlay(Text("S").bold())
                Text("Sudeep").font(.system(size: 14, weight: .medium))
                Spacer()
                Image(systemName: "chevron.up.chevron.down").font(.system(size: 10, weight: .bold)).foregroundStyle(.white.opacity(0.55))
            }
            .padding(12).background(PandaPalette.panel.opacity(0.5), in: RoundedRectangle(cornerRadius: 15))
            .padding(.bottom, 22)
        }
        .padding(.horizontal, 28).frame(width: 305).background(PandaPalette.sidebar)
    }

    @ViewBuilder
    private var workspace: some View {
        if isShowingImageIndex {
            ImageIndexWorkspace(onBack: { isShowingImageIndex = false })
        } else {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Circle().fill(PandaPalette.mint).frame(width: 8, height: 8).shadow(color: PandaPalette.mint, radius: 8)
                Text(topStatus)
                    .font(.system(size: 16, weight: .medium)).foregroundStyle(.white.opacity(0.72))
                Spacer()
                pandaAvatar(58)
            }
            .padding(.horizontal, 36).padding(.vertical, 17).background(PandaPalette.topbar)
            Divider().overlay(.white.opacity(0.07))

            ZStack {
                LinearGradient(colors: [PandaPalette.glow.opacity(0.28), .clear, PandaPalette.mint.opacity(0.05)], startPoint: .topLeading, endPoint: .bottomTrailing)
                ScrollView {
                    VStack(spacing: 22) {
                        pandaAvatar(118).padding(.top, 54).shadow(color: PandaPalette.mint.opacity(0.28), radius: 28)
                        VStack(spacing: 8) {
                            Text(didSearch ? "I found everything for you" : "What can I find for you?")
                                .font(.system(size: 29, weight: .bold, design: .rounded))
                            Text(didSearch ? resultSubtitle : "Ask in plain English. Your files stay on this Mac.")
                                .font(.system(size: 15, weight: .medium)).foregroundStyle(.white.opacity(0.55))
                        }
                        if isSearching {
                            ProgressView().controlSize(.large).tint(PandaPalette.mint).padding(.vertical, 42)
                        } else if didSearch {
                            resultCard
                        } else {
                            starterCard
                        }
                    }
                    .frame(maxWidth: 900).padding(.horizontal, 42).padding(.bottom, 150)
                }
                .scrollIndicators(.hidden)
            }
            promptBar.padding(.horizontal, 36).padding(.vertical, 22).background(PandaPalette.canvas)
        }
        }
    }

    private var resultSubtitle: String {
        results.isEmpty ? "No matching files were found in your indexed folders." : "Your matching files are ready to review."
    }

    private var topStatus: String {
        if isSearching { return brainStatus }
        if isIndexingLibrary { return libraryStatus ?? "Panda Intelligence is indexing your files…" }
        return brainStatus
    }

    private var starterCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Panda Intelligence searches only folders you choose", systemImage: "lock.shield")
                .font(.system(size: 15, weight: .semibold)).foregroundStyle(PandaPalette.mint)
            Text("Try “find my invoices”, “photos from Goa”, or “the video where pricing was discussed”.")
                .font(.system(size: 16)).foregroundStyle(.white.opacity(0.62))
        }
        .frame(maxWidth: 650, alignment: .leading).padding(25)
        .background(PandaPalette.result, in: RoundedRectangle(cornerRadius: 23))
        .overlay(RoundedRectangle(cornerRadius: 23).stroke(.white.opacity(0.08)))
    }

    private var resultCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: results.isEmpty ? "magnifyingglass" : "sparkles").foregroundStyle(PandaPalette.mint)
                Text(results.isEmpty ? "Search complete" : "Done — \(results.count) matching files found").font(.system(size: 16, weight: .semibold))
                Spacer()
                Image(systemName: results.isEmpty ? "minus.circle" : "checkmark.circle.fill").foregroundStyle(results.isEmpty ? .white.opacity(0.45) : PandaPalette.mint)
            }.padding(22)
            if results.isEmpty {
                Text("Index a folder first, then ask Panda Intelligence to find anything inside it.")
                    .foregroundStyle(.white.opacity(0.55)).padding(.horizontal, 22).padding(.bottom, 22)
            } else {
                Divider().overlay(.white.opacity(0.08))
                ForEach(results.prefix(5)) { result in
                    TaskResultRow(result: result)
                    if result.id != results.prefix(5).last?.id { Divider().overlay(.white.opacity(0.07)).padding(.leading, 76) }
                }
            }
        }
        .frame(maxWidth: 760).background(PandaPalette.result, in: RoundedRectangle(cornerRadius: 24))
        .overlay(RoundedRectangle(cornerRadius: 24).stroke(.white.opacity(0.1)))
    }

    private var promptBar: some View {
        HStack(spacing: 14) {
            pandaAvatar(42)
            TextField("Ask Panda Intelligence to find, organize, move, or rename anything…", text: $prompt)
                .textFieldStyle(.plain).font(.system(size: 16)).onSubmit(submitPrompt)
            Button(action: submitPrompt) {
                Image(systemName: isSearching ? "ellipsis" : "arrow.up").font(.system(size: 18, weight: .bold))
                    .frame(width: 48, height: 48).background(PandaPalette.mint, in: Circle()).foregroundStyle(PandaPalette.canvas)
            }
            .buttonStyle(.plain).disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isSearching)
            .opacity(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.45 : 1)
        }
        .padding(12).background(PandaPalette.panel, in: RoundedRectangle(cornerRadius: 22))
        .overlay(RoundedRectangle(cornerRadius: 22).stroke(.white.opacity(0.12)))
    }

    private func pandaAvatar(_ size: CGFloat) -> some View {
        Image("pandaimage").resizable().scaledToFill().frame(width: size, height: size)
            .background(PandaPalette.mint.opacity(0.13), in: Circle()).clipShape(Circle())
            .overlay(Circle().stroke(.white.opacity(0.12)))
    }

    private func resetTask() { prompt = ""; results = []; didSearch = false; isShowingImageIndex = false }

    private func addLibraryFolder() {
        Task {
            guard let folder = await SecurityBookmarks.shared.selectDirectory(
                message: "Choose a folder for Panda Intelligence to index and keep up to date"
            ) else { return }
            guard isLocalMacFolder(folder) else {
                libraryStatus = "Only folders stored on this Mac are supported"
                return
            }
            await index(folder: folder, usesSecurityBookmark: true)
        }
    }

    private func isLocalMacFolder(_ folder: URL) -> Bool {
        let home = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL.path
        let path = folder.standardizedFileURL.path
        let homePrefix = home.hasSuffix("/") ? home : home + "/"
        guard path == home || path.hasPrefix(homePrefix) else { return false }
        return !path.hasPrefix(homePrefix + "Library/CloudStorage/") &&
            !path.hasPrefix(homePrefix + "Library/Mobile Documents/")
    }

    private func startFullDiskScan() {
        guard !isIndexingLibrary else { return }
        Task {
            // If a prior full pass completed but left a few transient
            // filename-only rows, recover those paths directly instead of
            // walking every mounted volume again.
            if hasCompletedLibraryIndex && qwenVisualIndexVersion < 1 {
                isIndexingLibrary = true
                await retryIncompleteImages()
                isIndexingLibrary = false
                qwenVisualIndexVersion = 1
            }
            guard fullContentIndexVersion < Self.currentFullContentIndexVersion else { return }
            if !imageScanSessionPrepared {
                // A visual pass is resumable. Existing rows with a completed
                // title + explanation are retained, while rows without those
                // fields are upgraded exactly once by the normal hash cache.
                indexingService.deduplicateImageRecords()
                indexingService.pruneMissingImageRecords()
                imageScanSessionPrepared = true
            }
            indexingService.lastScanFailureCount = 0
            // This pass covers documents, text, spreadsheets, presentations,
            // archives/metadata, and audio. Videos are intentionally omitted
            // until their dedicated frame/transcript pipeline is enabled.
            for root in FullDiskAccess.finderRoots() {
                if Task.isCancelled { break }
                await index(folder: root, usesSecurityBookmark: false, reindexVisualMedia: false, imageOnly: false)
            }
            if !Task.isCancelled && indexingService.lastScanFailureCount == 0 {
                await retryIncompleteImages()
                hasCompletedLibraryIndex = true
                qwenVisualIndexVersion = 1
                fullContentIndexVersion = Self.currentFullContentIndexVersion
                imageScanSessionPrepared = false
            } else if indexingService.lastScanFailureCount > 0 {
                libraryStatus = "Indexed with \(indexingService.lastScanFailureCount) failures — will retry next launch"
            }
        }
    }

    private func index(folder: URL, usesSecurityBookmark: Bool, reindexVisualMedia: Bool = false, imageOnly: Bool = false) async {
        isIndexingLibrary = true
        defer { isIndexingLibrary = false }
        libraryStatus = "Finding files…"

        // The monitor only observes locations that the person explicitly chose.
        // It lets new and changed files become searchable without another scan.
        // A root watch would also receive transient screenshots and cache
        // events. Only persist watches for folders the user explicitly picked.
        if usesSecurityBookmark {
            fileMonitoringService.startWatching(folder)
        }

        let allExtensions = Set([
            // Images (visual records are already cached and hash-deduped).
            "png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "tiff", "bmp", "svg", "ico", "icns", "avif",
            // Camera RAW formats are still local image records; the backend
            // rasterizes them when the installed image codecs support it.
            "raw", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "raf", "pef", "srw",
            // Audio (Whisper transcript + semantic embedding).
            "mp3", "wav", "m4a", "aac", "aiff", "flac", "ogg", "wma",
            // Documents and office exports.
            "pdf", "doc", "docx", "pages", "odt", "rtf", "rtfd", "epub", "mobi",
            "xlsx", "xls", "xlsm", "csv", "tsv", "numbers", "ods",
            "ppt", "pptx", "key", "odp",
            // Plain text, source, data, markup, and configuration.
            "txt", "md", "markdown", "json", "jsonl", "ndjson", "xml", "yaml", "yml", "toml",
            "ini", "conf", "config", "env", "log", "sql", "html", "htm", "css", "scss", "less",
            "js", "jsx", "ts", "tsx", "mjs", "cjs", "py", "pyw", "swift", "m", "h", "c", "cc",
            "cpp", "cxx", "hpp", "java", "kt", "kts", "go", "rs", "rb", "php", "sh", "bash", "zsh",
            "fish", "graphql", "gql", "tex", "svelte", "vue", "astro", "cu", "cuh", "wgsl", "glsl",
            "metal", "plist", "strings", "stringsdata", "cmake", "mk", "make", "lock", "properties",
            "pbxproj", "entitlements", "ps1", "bat", "vim", "nix", "jinja", "jinja2", "template", "tmpl",
            "example", "inp", "dia", "d", "eml", "msg", "vcf", "ics",
            // Containers are indexed by filename/path/format; contents remain untouched.
            "zip", "rar", "7z", "tar", "gz", "tgz", "bz2", "xz", "iso", "dmg", "pkg", "app", "sqlite", "db", "bin", "dat",
            "psd", "ai", "eps", "sketch", "blend", "fig", "obj", "stl", "fbx", "dwg", "jar"
        ])
        let extensions = imageOnly
            ? Set(["png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "tiff", "bmp", "svg", "ico", "icns", "avif", "raw", "cr2", "cr3", "nef", "arw", "dng", "orf", "rw2", "raf", "pef", "srw"])
            : allExtensions

        if usesSecurityBookmark {
            do {
                let urls = try SecurityBookmarks.shared.listFilesRecursively(in: folder, extensions: Array(extensions))
                await index(urls: urls)
            } catch {
                libraryStatus = "Could not read \(folder.lastPathComponent)"
            }
        } else {
            await indexFullDisk(from: folder, extensions: extensions, reindexVisualMedia: reindexVisualMedia)
        }
    }

    private func retryIncompleteImages() async {
        let incompleteImages = indexingService.incompleteImageURLs()
        guard !incompleteImages.isEmpty else { return }
        libraryStatus = "Finishing \(incompleteImages.count) visual descriptions…"
        await LocalVectorOperationGate.semaphore.wait()
        _ = await indexingService.indexFilesIfNeededConcurrent(
            incompleteImages,
            maxConcurrency: 8
        )
        await LocalVectorOperationGate.semaphore.signal()
    }

    private func index(urls: [URL]) async {
        guard !urls.isEmpty else {
            libraryStatus = "No supported files found"
            return
        }

        var completed = 0
        for batch in urls.chunked(into: 24) {
            libraryStatus = "Indexing \(completed + 1)–\(min(completed + batch.count, urls.count)) of \(urls.count) files"
            _ = await indexingService.indexFilesIfNeededConcurrent(batch, maxConcurrency: 4)
            completed += batch.count
        }
        libraryStatus = "Indexed \(completed) files"
    }

    private func indexFullDisk(from root: URL, extensions: Set<String>, reindexVisualMedia: Bool) async {
        var discovered = 0
        var indexed = 0
        var unavailableLocations = 0

        // Index each batch before the scanner enumerates the next one. This
        // gives a whole-Mac scan real back-pressure instead of buffering paths.
        // One file at a time keeps an interactive search responsive while the
        // local index is growing. Large batches can monopolize the backend.
        await FullDiskAccess.scanBatches(from: root, extensions: extensions, batchSize: 8) { update in
            unavailableLocations += update.unavailableLocations
            let batch = update.files
            guard !batch.isEmpty else { return }
            discovered += batch.count
            let visualExtensions: Set<String> = ["png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "tiff", "bmp", "mp4", "mov", "m4v", "avi", "mkv", "webm"]
            let shouldReplaceFallbackCaption = reindexVisualMedia && visualExtensions.contains(batch[0].pathExtension.lowercased())
            libraryStatus = shouldReplaceFallbackCaption
                ? "Qwen is understanding \(batch[0].lastPathComponent)…"
                : "Indexing \(indexed + 1)–\(indexed + batch.count) (\(discovered) found): \(batch[0].lastPathComponent)"
            await LocalVectorOperationGate.semaphore.wait()
            let results = await indexingService.indexFilesIfNeededConcurrent(batch, maxConcurrency: 8, forceReindex: shouldReplaceFallbackCaption)
            indexingService.lastScanFailureCount += results.filter { !$0.isSuccess }.count
            await LocalVectorOperationGate.semaphore.signal()
            indexed += batch.count
        }
        if indexed == 0 {
            libraryStatus = unavailableLocations > 0
                ? "No files indexed — \(unavailableLocations) protected locations were unavailable"
                : "No supported files found"
        } else if unavailableLocations > 0 {
            libraryStatus = "Indexed \(indexed) files — \(unavailableLocations) protected locations skipped"
        } else {
            libraryStatus = "Indexed \(indexed) files"
        }
    }

    private func submitPrompt() {
        let query = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty, !isSearching else { return }
        history.append(query)
        prompt = "" // The request stays in history; the workspace shows only results.
        didSearch = true
        isSearching = true
        brainStatus = "Loading Panda brain…"

        // The first request makes the product useful immediately: begin the
        // local library build while the vector search handles anything already
        // indexed. Subsequent requests never start a duplicate scan.
        if !hasCompletedLibraryIndex && qwenVisualIndexVersion < 1 {
            startFullDiskScan()
        }
        Task {
            defer { isSearching = false }
            // LanceDB's local native bindings are not safe for simultaneous
            // writes and vector queries. Use the same gate as the scanner so
            // a person’s search never overlaps an indexing write.
            await LocalVectorOperationGate.semaphore.wait()
            let modelStatus = try? await apiService.getOllamaStatus()
            brainStatus = modelStatus?.available == true
                ? "Panda brain is understanding your files…"
                : "Searching your local library…"
            results = (try? await apiService.search(query: query, limit: 12).results) ?? []
            await LocalVectorOperationGate.semaphore.signal()
            brainStatus = results.isEmpty
                ? "Panda Intelligence is ready for your task"
                : "Panda found \(results.count) matching file\(results.count == 1 ? "" : "s")"
        }
    }
}

private enum LocalVectorOperationGate {
    static let semaphore = AsyncSemaphore(limit: 1)
}

private enum FullDiskAccess {
    struct ScanUpdate {
        let files: [URL]
        let unavailableLocations: Int
    }

    /// Panda's initial library is deliberately limited to this Mac's local
    /// user space. Removable volumes and cloud mounts are not requested or
    /// traversed; the user can add an explicitly chosen local folder later.
    static func finderRoots() -> [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        var roots: [URL] = []
        let excludedTopLevel = Set([
            "Library", "Applications", ".antigravity", ".gemini", ".codex",
            ".cache", ".npm", ".pnpm-store", ".Trash", ".Trashes"
        ])
        // Enumerate visible home children first. A package such as Music's
        // library can block a whole-home URL enumerator; isolating children
        // lets the remaining Finder locations continue independently.
        if let children = try? FileManager.default.contentsOfDirectory(
            at: home,
            includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) {
            roots.append(contentsOf: children.filter { child in
                guard !excludedTopLevel.contains(child.lastPathComponent),
                      let values = try? child.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey]) else { return false }
                return values.isDirectory == true && values.isSymbolicLink != true
            })
        }
        return roots
    }

    static func scanBatches(
        from root: URL,
        extensions: Set<String>,
        batchSize: Int = 12,
        onUpdate: @escaping @MainActor (ScanUpdate) async -> Void
    ) async {
        await withCheckedContinuation { continuation in
            Task.detached(priority: .utility) {
                // Search only local personal folders rather than mounted
                // drives, cloud mounts, macOS binaries, or caches.
                let manager = FileManager.default
                let excludedRoots = ["/System", "/private", "/dev", "/proc", "/cores", "/usr", "/bin", "/sbin", "/opt", "/Library", "/Applications"]
                // These locations are application caches and source build
                // artifacts, not a person's files. Scanning them overwhelms
                // the local embedding service and delays real searches.
                let excludedDirectoryNames: Set<String> = [
                    "Library", "node_modules", ".git", ".cache", ".npm", ".pnpm-store",
                    ".gemini", ".codex", ".antigravity", "DerivedData", "Pods", "Carthage", "Cache", "Caches",
                    "frameThumbnail", "Proxy", "Temp", ".Trash", "CapCut", ".next", ".turbo",
                    "Photos Library.photoslibrary", "Photo Booth Library.photobooth",
                    "dist", "build", "out", "target", ".cargo", ".rustup", ".lmstudio",
                    ".venv", ".venv_paddlevl", "site-packages", ".Trashes", ".Spotlight-V100",
                    ".fseventsd", ".DocumentRevisions-V100", ".TemporaryItems"
                ]
                // Video understanding only reads representative frames, so a
                // normal personal video does not need to be skipped merely
                // because its source file is larger than an image. Keep the
                // same one-gigabyte safety ceiling enforced by the backend.
                let maximumFileSize = 1_024 * 1_024 * 1_024
                var batch: [URL] = []
                var pendingDirectories: [URL] = [root]
                while let directory = pendingDirectories.popLast() {
                    if Task.isCancelled { break }
                    if excludedRoots.contains(where: { directory.path == $0 || directory.path.hasPrefix($0 + "/") }) {
                        continue
                    }
                    if !excludedDirectoryNames.isDisjoint(with: Set(directory.pathComponents)) {
                        continue
                    }
                    guard let children = try? manager.contentsOfDirectory(
                        at: directory,
                        // File sizes are fetched only for candidate regular
                        // files below so enumeration stays lightweight.
                        includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
                        options: [.skipsHiddenFiles, .skipsPackageDescendants]
                    ) else {
                        // TCC or an unplugged volume can deny one directory;
                        // continue with the rest of Finder and surface the
                        // unavailable location in the final status.
                        await onUpdate(ScanUpdate(files: [], unavailableLocations: 1))
                        continue
                    }

                    for url in children {
                        if Task.isCancelled { break }
                        if excludedRoots.contains(where: { url.path == $0 || url.path.hasPrefix($0 + "/") }) ||
                            !excludedDirectoryNames.isDisjoint(with: Set(url.pathComponents)) {
                            continue
                        }
                        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey, .fileSizeKey]),
                              values.isSymbolicLink != true else { continue }
                        if values.isDirectory == true {
                            pendingDirectories.append(url)
                            continue
                        }
                        guard values.isRegularFile == true,
                              (values.fileSize ?? 0) <= maximumFileSize,
                              extensions.contains(url.pathExtension.lowercased()) else { continue }
                        batch.append(url)
                        if batch.count == batchSize {
                            await onUpdate(ScanUpdate(files: batch, unavailableLocations: 0))
                            batch.removeAll(keepingCapacity: true)
                        }
                    }
                    // Keep a partial batch while walking adjacent folders so
                    // the VLM receives full concurrent batches even when a
                    // project contains many tiny image directories. The
                    // remainder is flushed once the walk finishes below.
                }
                if !batch.isEmpty {
                    await onUpdate(ScanUpdate(files: batch, unavailableLocations: 0))
                }
                continuation.resume()
            }
        }
    }

}

private extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map { Array(self[$0..<Swift.min($0 + size, count)]) }
    }
}

private struct TaskResultRow: View {
    let result: SearchResult
    var body: some View {
        HStack(spacing: 15) {
            LocalResultPreview(result: result)
            VStack(alignment: .leading, spacing: 4) {
                Text(result.fileName).font(.system(size: 15, weight: .semibold)).lineLimit(1)
                Text(result.filePath).font(.system(size: 12)).foregroundStyle(.white.opacity(0.47)).lineLimit(1)
            }
            Spacer()
            Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: result.filePath)]) }
                .buttonStyle(.bordered).tint(PandaPalette.mint.opacity(0.55))
        }.padding(18)
    }
}

/// A local, inspectable record of every image Panda has successfully indexed.
/// The file hash in IndexedFile is the scan-once key; an edited file receives
/// a new record, while an unchanged image never asks the VLM again.
private struct ImageIndexWorkspace: View {
    @Query(
        filter: #Predicate<IndexedFile> { $0.mediaType == "image" },
        sort: \IndexedFile.indexedAt,
        order: .reverse
    ) private var images: [IndexedFile]

    let onBack: () -> Void
    @State private var selectedPath: String?
    @State private var filter = ""

    private var filteredImages: [IndexedFile] {
        guard !filter.isEmpty else { return images }
        return images.filter {
            $0.fileName.localizedCaseInsensitiveContains(filter) ||
            $0.filePath.localizedCaseInsensitiveContains(filter) ||
            ($0.visualTitle ?? "").localizedCaseInsensitiveContains(filter) ||
            ($0.visualExplanation ?? "").localizedCaseInsensitiveContains(filter)
        }
    }

    private var selectedImage: IndexedFile? {
        filteredImages.first { $0.filePath == selectedPath } ?? filteredImages.first
    }

    private var describedImageCount: Int {
        images.reduce(into: 0) { count, image in
            let hasTitle = !(image.visualTitle?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
            let explanation = image.visualExplanation?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let hasExplanation = !explanation.isEmpty && !explanation.hasPrefix("Image file:")
            if hasTitle && hasExplanation { count += 1 }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Button(action: onBack) { Label("Search", systemImage: "chevron.left") }
                    .buttonStyle(.borderless)
                    .foregroundStyle(PandaPalette.mint)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Image Index").font(.system(size: 22, weight: .bold, design: .rounded))
                    Text("\(images.count) indexed • \(describedImageCount) VLM-described • \(max(0, images.count - describedImageCount)) waiting")
                        .font(.system(size: 12)).foregroundStyle(.white.opacity(0.53))
                }
                Spacer()
                TextField("Filter index", text: $filter)
                    .textFieldStyle(.roundedBorder).frame(width: 220)
            }
            .padding(.horizontal, 30).padding(.vertical, 20).background(PandaPalette.topbar)

            if images.isEmpty {
                ContentUnavailableView(
                    "No indexed images yet",
                    systemImage: "photo.stack",
                    description: Text("Scan the Local Mac or Add Library Folder. Panda will add each supported image here after its first scan.")
                )
                .foregroundStyle(.white.opacity(0.75))
            } else {
                HSplitView {
                    VStack(spacing: 0) {
                        imageTableHeader
                        Divider().overlay(.white.opacity(0.1))
                        ScrollView {
                            LazyVStack(spacing: 0) {
                                ForEach(filteredImages) { image in
                                    ImageIndexRow(image: image, isSelected: image.filePath == selectedImage?.filePath) {
                                        selectedPath = image.filePath
                                    }
                                    Divider().overlay(.white.opacity(0.07))
                                }
                            }
                        }
                    }
                    .frame(minWidth: 620)
                    ImageIndexPreview(image: selectedImage)
                        .frame(minWidth: 310, idealWidth: 385)
                }
            }
        }
        .background(PandaPalette.canvas)
        .preferredColorScheme(.dark)
    }

    private var imageTableHeader: some View {
        HStack(spacing: 12) {
            Text("IMAGE").frame(width: 56, alignment: .leading)
            Text("TITLE").frame(width: 180, alignment: .leading)
            Text("PATH — CLICK TO PREVIEW").frame(maxWidth: .infinity, alignment: .leading)
            Text("INDEXED").frame(width: 78, alignment: .leading)
        }
        .font(.system(size: 10, weight: .bold)).tracking(0.9).foregroundStyle(.white.opacity(0.42))
        .padding(.horizontal, 16).padding(.vertical, 10)
    }
}

private struct ImageIndexRow: View {
    let image: IndexedFile
    let isSelected: Bool
    let select: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            LocalImageThumbnail(path: image.filePath).frame(width: 56, height: 48)
            VStack(alignment: .leading, spacing: 3) {
                Text(image.visualTitle?.isEmpty == false ? image.visualTitle! : image.fileName)
                    .font(.system(size: 13, weight: .semibold)).lineLimit(1)
                Text(image.visualExplanation ?? "Waiting for the local vision description")
                    .font(.system(size: 11)).foregroundStyle(.white.opacity(0.52)).lineLimit(2)
            }
            .frame(width: 180, alignment: .leading)
            Button(action: select) {
                Text(image.filePath).font(.system(size: 11)).foregroundStyle(PandaPalette.mint).lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain).help("Preview this image in Panda")
            Text(image.indexedAt, format: .dateTime.month(.abbreviated).day())
                .font(.system(size: 11)).foregroundStyle(.white.opacity(0.5)).frame(width: 78, alignment: .leading)
        }
        .padding(.horizontal, 16).padding(.vertical, 9)
        .background(isSelected ? PandaPalette.mint.opacity(0.12) : .clear)
        .contentShape(Rectangle()).onTapGesture(perform: select)
    }
}

private struct ImageIndexPreview: View {
    let image: IndexedFile?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let image, let nsImage = NSImage(contentsOf: URL(fileURLWithPath: image.filePath)) {
                Image(nsImage: nsImage).resizable().scaledToFit().frame(maxWidth: .infinity, maxHeight: 300)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                Text(image.visualTitle?.isEmpty == false ? image.visualTitle! : image.fileName)
                    .font(.system(size: 18, weight: .bold))
                Text(image.visualExplanation ?? "No visual explanation is available yet.")
                    .font(.system(size: 13)).foregroundStyle(.white.opacity(0.7))
                Divider().overlay(.white.opacity(0.1))
                Text(image.filePath).font(.system(size: 11)).foregroundStyle(PandaPalette.mint).textSelection(.enabled)
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: image.filePath)]) }
                    .buttonStyle(.bordered).tint(PandaPalette.mint)
            } else {
                ContentUnavailableView("Select an image", systemImage: "photo", description: Text("Click a path in the table to open its preview here."))
            }
            Spacer()
        }
        .padding(22).background(PandaPalette.result)
    }
}

private final class ImageThumbnailCache {
    static let shared: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        // Keep the table light even when a library contains thousands of
        // images. The selected image still renders at full preview size.
        cache.countLimit = 300
        return cache
    }()
}

private struct LocalImageThumbnail: View {
    let path: String

    @State private var thumbnail: NSImage?

    var body: some View {
        Group {
            if let image = thumbnail {
                Image(nsImage: image).resizable().scaledToFill()
            } else {
                Image(systemName: "photo").foregroundStyle(PandaPalette.mint)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).background(PandaPalette.mint.opacity(0.13))
        .clipShape(RoundedRectangle(cornerRadius: 7))
        .task(id: path) {
            if let cached = ImageThumbnailCache.shared.object(forKey: path as NSString) {
                thumbnail = cached
                return
            }

            // ImageIO thumbnail decoding is intentionally detached from the
            // main actor. Loading full-resolution NSImages in a row body made
            // large indexes consume the UI thread and slowed the scan.
            guard let cgImage = await Self.loadCGImage(path: path), !Task.isCancelled else { return }
            let image = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
            ImageThumbnailCache.shared.setObject(image, forKey: path as NSString)
            thumbnail = image
        }
    }

    private static func loadCGImage(path: String) async -> CGImage? {
        await Task.detached(priority: .utility) {
            let url = URL(fileURLWithPath: path)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
            let options: CFDictionary = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: 192,
                kCGImageSourceCreateThumbnailWithTransform: true,
            ] as CFDictionary
            return CGImageSourceCreateThumbnailAtIndex(source, 0, options)
        }.value
    }
}

private struct LocalResultPreview: View {
    let result: SearchResult

    var body: some View {
        Group {
            if result.mediaType == .image,
               let image = NSImage(contentsOf: URL(fileURLWithPath: result.filePath)) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: result.mediaType?.icon ?? "doc")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(PandaPalette.mint)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(PandaPalette.mint.opacity(0.15))
            }
        }
        .frame(width: 58, height: 58)
        .background(PandaPalette.mint.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(.white.opacity(0.1)))
    }
}

private enum PandaPalette {
    static let canvas = Color(red: 0.018, green: 0.035, blue: 0.045)
    static let sidebar = Color(red: 0.012, green: 0.025, blue: 0.033)
    static let panel = Color(red: 0.055, green: 0.09, blue: 0.1)
    static let result = Color(red: 0.075, green: 0.11, blue: 0.12)
    static let topbar = Color(red: 0.06, green: 0.12, blue: 0.13)
    static let glow = Color(red: 0.10, green: 0.48, blue: 0.42)
    static let mint = Color(red: 0.16, green: 0.90, blue: 0.66)
}
