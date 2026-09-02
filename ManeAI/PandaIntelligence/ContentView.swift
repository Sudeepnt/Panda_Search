import SwiftUI
import AppKit
import SwiftData
import ImageIO

/// The normal Panda Intelligence app window. Ctrl+W still opens the compact quick-search overlay.
struct ContentView: View {
    private static let currentFullContentIndexVersion = 3
    @EnvironmentObject var apiService: APIService
    @EnvironmentObject var indexingService: DocumentIndexingService
    @EnvironmentObject var fileMonitoringService: FileMonitoringService
    @EnvironmentObject var sidecarManager: SidecarManager

    @State private var prompt = ""
    // History keeps lightweight result snapshots — never image bytes. Image
    // cards re-open their local file path when a conversation is restored.
    @State private var history: [SearchHistoryEntry] = []
    @State private var didLoadSavedHistory = false
    @AppStorage("PandaIntelligence.SearchHistorySnapshots") private var savedHistoryData = Data()
    @State private var results: [SearchResult] = []
    @State private var searchError: String?
    @State private var isSearching = false
    @State private var didSearch = false
    @State private var isIndexingLibrary = false
    @State private var libraryStatus: String?
    @State private var brainStatus = "Panda Intelligence is ready for your task"
    @State private var isShowingImageIndex = false
    @State private var pendingFileAction: PendingFileAction?
    @FocusState private var isPromptFocused: Bool
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
        HStack(alignment: .top, spacing: 12) {
            sidebar
            workspace
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(PandaPalette.workspace)
                .clipShape(RoundedRectangle(cornerRadius: 26, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous).stroke(PandaPalette.stroke))
        }
        .padding(12)
        .frame(minWidth: 1000, minHeight: 680)
        .background(PandaPalette.canvas)
        .preferredColorScheme(.dark)
        .alert(item: $pendingFileAction) { pending in
            Alert(
                title: Text(pending.title),
                message: Text(pending.message),
                primaryButton: pending.isDestructive
                    ? .destructive(Text("Confirm")) { executeFileAction(pending.action) }
                    : .default(Text("Confirm")) { executeFileAction(pending.action) },
                secondaryButton: .cancel()
            )
        }
        .task {
            loadSavedHistoryIfNeeded()
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
            .padding(.top, 22)

            Button(action: resetTask) {
                Label("New Task", systemImage: "plus")
                    .font(.system(size: 16, weight: .semibold))
                    .frame(maxWidth: .infinity).padding(.vertical, 17)
                    .background(PandaPalette.panel, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 16).stroke(.white.opacity(0.14)))
            }
            .buttonStyle(.plain).padding(.top, 32)

            Button(action: addLibraryFolder) {
                Label("Add Library Folder", systemImage: "folder.badge.plus")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 11)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain).foregroundStyle(PandaPalette.mint)
            .background(PandaPalette.panel.opacity(0.72), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(PandaPalette.stroke))
            .padding(.top, 14)

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
            .padding(.vertical, 11)
            .padding(.horizontal, 12)
            .background(isShowingImageIndex ? PandaPalette.elevated : PandaPalette.panel.opacity(0.45), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(PandaPalette.stroke))
            .padding(.top, 8)

            Button(action: startFullDiskScan) {
                Label(isIndexingLibrary ? "Scanning Local Mac…" : "Scan Local Mac", systemImage: "internaldrive")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.white.opacity(0.76))
            .padding(.vertical, 11)
            .padding(.horizontal, 12)
            .background(PandaPalette.panel.opacity(0.45), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(PandaPalette.stroke))
            .padding(.top, 8)
            .disabled(isIndexingLibrary)

            Text("HISTORY")
                .font(.system(size: 11, weight: .semibold)).tracking(1).foregroundStyle(.white.opacity(0.35))
                .padding(.top, 32)

            ScrollView {
                LazyVStack(spacing: 8) {
                    ForEach(Array(history.reversed())) { entry in
                        HStack(spacing: 3) {
                            Button {
                                restoreHistory(entry)
                            } label: {
                                HStack(spacing: 11) {
                                    Image(systemName: "clock").foregroundStyle(PandaPalette.mint.opacity(0.85))
                                    Text(entry.query).lineLimit(2).font(.system(size: 13, weight: .medium))
                                    Spacer(minLength: 0)
                                    Text("\(entry.results.count)")
                                        .font(.system(size: 11, weight: .bold, design: .rounded))
                                        .foregroundStyle(PandaPalette.mint)
                                }
                                .padding(.leading, 13)
                                .padding(.vertical, 13)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)

                            Menu {
                                Button {
                                    copyHistoryPrompt(entry.query)
                                } label: {
                                    Label("Copy prompt", systemImage: "doc.on.doc")
                                }
                                Button(role: .destructive) {
                                    deleteHistoryEntry(entry)
                                } label: {
                                    Label("Delete conversation", systemImage: "trash")
                                }
                            } label: {
                                Image(systemName: "ellipsis")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundStyle(.white.opacity(0.55))
                                    .frame(width: 34, height: 36)
                            }
                            .menuStyle(.borderlessButton)
                        }
                        .background(PandaPalette.panel.opacity(0.72), in: RoundedRectangle(cornerRadius: 14))
                        .overlay(RoundedRectangle(cornerRadius: 14).stroke(PandaPalette.stroke))
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
            .padding(12).background(PandaPalette.panel.opacity(0.72), in: RoundedRectangle(cornerRadius: 15))
            .overlay(RoundedRectangle(cornerRadius: 15).stroke(PandaPalette.stroke))
            .padding(.bottom, 4)
        }
        .padding(.horizontal, 20)
        .frame(width: 286)
        .frame(maxHeight: .infinity)
        .background(PandaPalette.sidebar)
        .clipShape(RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 26, style: .continuous).stroke(PandaPalette.stroke))
    }

    @ViewBuilder
    private var workspace: some View {
        if isShowingImageIndex {
            ImageIndexWorkspace(onBack: { isShowingImageIndex = false })
        } else {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                HStack(spacing: 9) {
                    Circle().fill(PandaPalette.mint).frame(width: 7, height: 7).shadow(color: PandaPalette.mint, radius: 7)
                    Text(topStatus)
                        .font(.system(size: 13, weight: .medium)).foregroundStyle(.white.opacity(0.72))
                }
                .padding(.horizontal, 13)
                .padding(.vertical, 10)
                .background(PandaPalette.panel.opacity(0.78), in: Capsule())
                .overlay(Capsule().stroke(PandaPalette.stroke))
                Spacer()
                pandaAvatar(46)
            }
            .padding(.horizontal, 20).padding(.vertical, 14).background(PandaPalette.topbar)
            Divider().overlay(.white.opacity(0.07))

            ZStack {
                LinearGradient(
                    colors: [PandaPalette.glow.opacity(0.14), PandaPalette.workspace, PandaPalette.panel.opacity(0.22)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                ScrollView {
                    VStack(spacing: 22) {
                        pandaAvatar(104).padding(.top, 44).shadow(color: .black.opacity(0.35), radius: 22, y: 12)
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
                    // Search results are a Finder-like workspace, not a
                    // narrow chat bubble. Let the grid use the available
                    // Mac window while retaining comfortable side margins.
                    .frame(maxWidth: 1280).padding(.horizontal, 42).padding(.bottom, 150)
                }
                .scrollIndicators(.hidden)
            }
            promptBar
                .padding(.horizontal, 22)
                .padding(.bottom, 20)
                .padding(.top, 10)
                .background(PandaPalette.workspace)
        }
        }
    }

    private var resultSubtitle: String {
        if let searchError, results.isEmpty { return searchError }
        return results.isEmpty ? "No matching files were found in your indexed folders." : "Your matching files are ready to review."
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
                Text(results.isEmpty ? "Search complete" : "Done — \(results.count) relevant files found").font(.system(size: 16, weight: .semibold))
                Spacer()
                if !results.isEmpty {
                    Text("Use result numbers for actions")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.white.opacity(0.42))
                }
                Image(systemName: results.isEmpty ? "minus.circle" : "checkmark.circle.fill").foregroundStyle(results.isEmpty ? .white.opacity(0.45) : PandaPalette.mint)
            }.padding(22)
            if results.isEmpty {
                Text(searchError ?? "No indexed file matched that request. Try a broader description or scan the Local Mac.")
                    .foregroundStyle(.white.opacity(0.55)).padding(.horizontal, 22).padding(.bottom, 22)
            } else {
                Divider().overlay(.white.opacity(0.08))
                LazyVGrid(
                    columns: resultGridColumns,
                    spacing: 14
                ) {
                    ForEach(Array(results.enumerated()), id: \.element.id) { index, result in
                        TaskResultRow(
                            result: result,
                            serial: index + 1,
                            isExpanded: results.count == 1,
                            onOpen: { NSWorkspace.shared.open(URL(fileURLWithPath: result.filePath)) },
                            onShowInFinder: { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: result.filePath)]) },
                            onAction: { action in handleResultAction(action, result: result, serial: index + 1) }
                        )
                    }
                }
                .padding(18)
            }
        }
        .frame(maxWidth: 1280).background(PandaPalette.result, in: RoundedRectangle(cornerRadius: 24))
        .overlay(RoundedRectangle(cornerRadius: 24).stroke(.white.opacity(0.1)))
    }

    private var resultGridColumns: [GridItem] {
        switch results.count {
        case 1:
            // A single result should occupy the workspace rather than leaving
            // an empty second column beside it.
            return [GridItem(.flexible())]
        case 2:
            return Array(repeating: GridItem(.flexible(minimum: 280), spacing: 14), count: 2)
        default:
            return [GridItem(.adaptive(minimum: 320, maximum: 420), spacing: 14)]
        }
    }

    private var promptBar: some View {
        HStack(spacing: 14) {
            pandaAvatar(42)
            TextField("Ask Panda Intelligence to find, organize, move, or rename anything…", text: $prompt)
                .textFieldStyle(.plain)
                .font(.system(size: 16))
                .focused($isPromptFocused)
                .onSubmit(submitPrompt)
            Button(action: submitPrompt) {
                Image(systemName: isSearching ? "ellipsis" : "arrow.up").font(.system(size: 18, weight: .bold))
                    .frame(width: 48, height: 48).background(PandaPalette.mint, in: Circle()).foregroundStyle(PandaPalette.canvas)
            }
            .buttonStyle(.plain).disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isSearching)
            .opacity(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.45 : 1)
        }
        .padding(10)
        .background(PandaPalette.elevated, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(PandaPalette.strokeStrong))
        .shadow(color: .black.opacity(0.3), radius: 18, y: 8)
    }

    private func pandaAvatar(_ size: CGFloat) -> some View {
        Image("pandaimage").resizable().scaledToFill().frame(width: size, height: size)
            .background(PandaPalette.mint.opacity(0.13), in: Circle()).clipShape(Circle())
            .overlay(Circle().stroke(.white.opacity(0.12)))
    }

    private func resetTask() { prompt = ""; results = []; searchError = nil; didSearch = false; isShowingImageIndex = false }

    private func restoreHistory(_ entry: SearchHistoryEntry) {
        // Restore the exact result snapshot instead of issuing a new search.
        // This makes history behave like a conversation: the user sees the
        // result grid they saw then, even while a full-library scan continues.
        results = entry.results.map(\.searchResult)
        prompt = ""
        searchError = nil
        didSearch = true
        isShowingImageIndex = false
        isPromptFocused = false
        brainStatus = entry.results.isEmpty
            ? "Restored search — no matching files were found"
            : "Restored \(entry.results.count) saved result\(entry.results.count == 1 ? "" : "s")"
    }

    private func copyHistoryPrompt(_ query: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(query, forType: .string)
        brainStatus = "Copied previous prompt"
    }

    private func deleteHistoryEntry(_ entry: SearchHistoryEntry) {
        history.removeAll { $0.id == entry.id }
        persistHistory()
        brainStatus = "Deleted saved conversation"
    }

    private func loadSavedHistoryIfNeeded() {
        guard !didLoadSavedHistory else { return }
        didLoadSavedHistory = true
        guard !savedHistoryData.isEmpty,
              let decoded = try? JSONDecoder().decode([SearchHistoryEntry].self, from: savedHistoryData) else { return }
        history = decoded
    }

    private func saveSearchHistory(query: String, results: [SearchResult]) {
        history.append(SearchHistoryEntry(query: query, results: results))
        // Bound the number of conversations so the local preference remains
        // fast to load while still retaining a useful recent history.
        if history.count > 30 {
            history.removeFirst(history.count - 30)
        }
        persistHistory()
    }

    private func persistHistory() {
        savedHistoryData = (try? JSONEncoder().encode(history)) ?? Data()
    }

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
            let needsTextIndexUpgrade = fullContentIndexVersion < Self.currentFullContentIndexVersion
            guard needsTextIndexUpgrade else { return }
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
                await index(
                    folder: root,
                    usesSecurityBookmark: false,
                    reindexVisualMedia: false,
                    imageOnly: false,
                    forceTextReindex: needsTextIndexUpgrade
                )
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

    private func index(
        folder: URL,
        usesSecurityBookmark: Bool,
        reindexVisualMedia: Bool = false,
        imageOnly: Bool = false,
        forceTextReindex: Bool = false
    ) async {
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
                await index(urls: urls, forceTextReindex: forceTextReindex)
            } catch {
                libraryStatus = "Could not read \(folder.lastPathComponent)"
            }
        } else {
            await indexFullDisk(
                from: folder,
                extensions: extensions,
                reindexVisualMedia: reindexVisualMedia,
                forceTextReindex: forceTextReindex
            )
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

    private func index(urls: [URL], forceTextReindex: Bool = false) async {
        guard !urls.isEmpty else {
            libraryStatus = "No supported files found"
            return
        }

        var completed = 0
        for batch in urls.chunked(into: 24) {
            libraryStatus = "Indexing \(completed + 1)–\(min(completed + batch.count, urls.count)) of \(urls.count) files"
            _ = await indexingService.indexFilesIfNeededConcurrent(
                batch,
                maxConcurrency: 4,
                forceTextReindex: forceTextReindex
            )
            completed += batch.count
        }
        libraryStatus = "Indexed \(completed) files"
    }

    private func indexFullDisk(
        from root: URL,
        extensions: Set<String>,
        reindexVisualMedia: Bool,
        forceTextReindex: Bool = false
    ) async {
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
            let results = await indexingService.indexFilesIfNeededConcurrent(
                batch,
                maxConcurrency: 8,
                forceReindex: shouldReplaceFallbackCaption,
                forceTextReindex: forceTextReindex
            )
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
        prompt = ""

        // An action prompt operates on the numbered results currently on
        // screen. This makes “rename the 2nd to this” deterministic: Panda
        // resolves the ordinal against the exact result order the user sees,
        // then asks for confirmation before changing the Finder file.
        if let action = FinderPromptActionParser.parse(query, results: results) {
            pendingFileAction = PendingFileAction(action: action)
            return
        }

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
            // A distribution build owns its sidecar, while an Xcode build may
            // talk to the developer backend. Re-check and start the local
            // service before searching so a relaunch or a slow sidecar startup
            // cannot be mistaken for an empty index.
            await sidecarManager.checkHealth()
            if !sidecarManager.isHealthy {
                await sidecarManager.start()
                await sidecarManager.checkHealth()
            }

            do {
                let modelStatus = try? await apiService.getOllamaStatus()
                brainStatus = modelStatus?.available == true
                    ? "Panda brain is understanding your files…"
                    : "Searching your local library…"
                let foundResults = try await apiService.search(query: query, limit: 50).results
                results = foundResults
                saveSearchHistory(query: query, results: foundResults)
                searchError = nil
            } catch {
                // Never turn a transport/decoding failure into the misleading
                // “index a folder first” empty state. The user can retry while
                // the sidecar restarts, and the actual local error is visible.
                results = []
                searchError = "Panda could not reach the local file index. \(error.localizedDescription)"
                brainStatus = "Local file index unavailable"
            }
            await LocalVectorOperationGate.semaphore.signal()
            if searchError == nil {
                brainStatus = results.isEmpty
                    ? "Panda Intelligence is ready for your task"
                    : "Panda found \(results.count) relevant file\(results.count == 1 ? "" : "s")"
            }
        }
    }

    private func handleResultAction(_ action: ResultCardAction, result: SearchResult, serial: Int) {
        switch action {
        case .open:
            NSWorkspace.shared.open(URL(fileURLWithPath: result.filePath))
        case .showInFinder:
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: result.filePath)])
        case .quickLook:
            FinderFileActions.quickLook(result.filePath)
        case .copy:
            FinderFileActions.copyToPasteboard(result.filePath)
            brainStatus = "Copied \(result.fileName)"
        case .copyPath:
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(result.filePath, forType: .string)
            brainStatus = "Copied path for \(result.fileName)"
        case .rename:
            prompt = "rename \(serial) to "
            brainStatus = "Tell Panda the new name for result \(serial)"
        case .move:
            prompt = "move \(serial) to "
            brainStatus = "Tell Panda the destination folder for result \(serial)"
        case .openWith:
            prompt = "open (serial) with "
            brainStatus = "Tell Panda which app should open result (serial)"
        case .duplicate, .compress, .uncompress, .tag, .trash, .getInfo:
            if let action = FinderPromptActionParser.directAction(action, result: result) {
                pendingFileAction = PendingFileAction(action: action)
            }
        }
    }

    private func executeFileAction(_ action: FinderPromptAction) {
        Task { @MainActor in
            do {
                let outcome = try await FinderFileActions.execute(
                    action,
                    apiService: apiService,
                    indexingService: indexingService
                )
                if let oldPath = outcome.oldPath,
                   let index = results.firstIndex(where: { $0.filePath == oldPath }) {
                    if let updated = outcome.updatedResult {
                        results[index] = updated
                    } else {
                        results.remove(at: index)
                    }
                }
                brainStatus = outcome.message
            } catch {
                brainStatus = "Panda could not complete that action: \(error.localizedDescription)"
            }
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
                    "frameThumbnail", "Proxy", "Temp", ".Trash", "CapCut", ".next", ".turbo", "work",
                    ".build", ".swiftpm", "coverage",
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

/// Actions exposed by every result card. More involved actions can be
/// expressed in the prompt so Panda can resolve a destination/name naturally.
private enum ResultCardAction {
    case open
    case showInFinder
    case quickLook
    case openWith
    case copy
    case copyPath
    case rename
    case move
    case duplicate
    case compress
    case uncompress
    case tag
    case trash
    case getInfo
}

private enum FinderPromptAction {
    case open(SearchResult)
    case openWith(SearchResult, String)
    case reveal(SearchResult)
    case copy(SearchResult)
    case copyTo(SearchResult, URL)
    case rename(SearchResult, String)
    case move(SearchResult, URL)
    case duplicate(SearchResult)
    case compress(SearchResult)
    case uncompress(SearchResult)
    case tag(SearchResult, String)
    case trash(SearchResult)
    case getInfo(SearchResult)
    case createFolder(String, URL)
}

private struct PendingFileAction: Identifiable {
    let id = UUID()
    let action: FinderPromptAction

    var isDestructive: Bool {
        if case .trash = action { return true }
        return false
    }

    var title: String {
        switch action {
        case .open: return "Open file?"
        case .openWith: return "Open file with this app?"
        case .reveal: return "Show in Finder?"
        case .copy: return "Copy file?"
        case .copyTo: return "Copy file to this folder?"
        case .rename: return "Rename file?"
        case .move: return "Move file?"
        case .duplicate: return "Duplicate file?"
        case .compress: return "Compress file?"
        case .uncompress: return "Uncompress archive?"
        case .tag: return "Add Panda tag?"
        case .trash: return "Move file to Trash?"
        case .getInfo: return "Open Finder info?"
        case .createFolder: return "Create folder?"
        }
    }

    var message: String {
        switch action {
        case .open(let result), .openWith(let result, _), .reveal(let result), .copy(let result),
             .copyTo(let result, _),
             .duplicate(let result), .compress(let result), .uncompress(let result), .trash(let result),
             .getInfo(let result):
            return result.filePath
        case .rename(let result, let name):
            return "Rename \(result.fileName) to \(name)?\n\(result.filePath)"
        case .move(let result, let destination):
            return "Move \(result.fileName) to:\n\(destination.path)"
        case .tag(let result, let tag):
            return "Add the tag “\(tag)” to:\n\(result.filePath)"
        case .createFolder(let name, let location):
            return "Create “\(name)” in:\n\(location.path)"
        }
    }
}

/// Small, local parser for the Finder verbs people use most often. Search
/// itself remains AI-ranked; these mutations use explicit ordinal resolution
/// so a command cannot accidentally target a different result after reordering.
private enum FinderPromptActionParser {
    static func parse(_ prompt: String, results: [SearchResult]) -> FinderPromptAction? {
        let normalized = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = normalized.lowercased()

        if let folderName = capture(afterAny: ["create a folder called", "create folder called", "new folder called", "make a folder called"], in: normalized),
           !folderName.isEmpty {
            return .createFolder(cleanName(folderName), FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Desktop"))
        }

        guard let result = selectedResult(from: lower, results: results) else { return nil }
        if lower.hasPrefix("open ") || lower.hasPrefix("launch ") || lower.contains(" open ") {
            if let app = capture(afterAny: [" with "], in: normalized), !app.isEmpty {
                return .openWith(result, cleanName(app))
            }
            return .open(result)
        }
        if lower.contains("show in finder") || lower.contains("reveal in finder") || lower.contains("where is ") {
            return .reveal(result)
        }
        if lower.hasPrefix("copy ") || lower.contains(" copy ") {
            if let destinationText = capture(afterAny: [" to ", " into "], in: normalized),
               !destinationText.isEmpty,
               let destination = resolveFolder(destinationText) {
                return .copyTo(result, destination)
            }
            return .copy(result)
        }
        if lower.contains("rename") || lower.contains("renamed") {
            guard let requestedName = capture(afterAny: [" to ", " as ", " called ", " name ", " named "], in: normalized),
                  !requestedName.isEmpty else { return nil }
            return .rename(result, normalizedName(requestedName, source: result.fileName))
        }
        if lower.contains("move") || lower.contains("put ") || lower.contains("place ") {
            guard let destinationText = capture(afterAny: [" to ", " into ", " in "], in: normalized),
                  let destination = resolveFolder(destinationText) else { return nil }
            return .move(result, destination)
        }
        if lower.contains("duplicate") || lower.contains("make a copy") {
            return .duplicate(result)
        }
        if lower.contains("uncompress") || lower.contains("unzip") || lower.contains("extract") {
            return .uncompress(result)
        }
        if lower.contains("compress") || lower.contains("zip ") || lower.contains("archive ") {
            return .compress(result)
        }
        if lower.contains("tag ") || lower.contains("label ") {
            let tag = capture(afterAny: ["tag as ", "tag it ", "label as ", "label it "], in: normalized) ?? "Panda"
            return .tag(result, cleanName(tag))
        }
        if lower.contains("trash") || lower.contains("delete ") || lower.contains("remove ") {
            return .trash(result)
        }
        if lower.contains("get info") || lower.contains("information") {
            return .getInfo(result)
        }
        return nil
    }

    static func directAction(_ action: ResultCardAction, result: SearchResult) -> FinderPromptAction? {
        switch action {
        case .open: return .open(result)
        case .showInFinder: return .reveal(result)
        case .quickLook: return .open(result)
        case .openWith: return nil
        case .copy: return .copy(result)
        case .duplicate: return .duplicate(result)
        case .compress: return .compress(result)
        case .uncompress: return .uncompress(result)
        case .tag: return .tag(result, "Panda")
        case .trash: return .trash(result)
        case .getInfo: return .getInfo(result)
        case .copyPath, .rename, .move: return nil
        }
    }

    private static func selectedResult(from lowerPrompt: String, results: [SearchResult]) -> SearchResult? {
        guard !results.isEmpty else { return nil }
        if let number = ordinal(in: lowerPrompt), results.indices.contains(number - 1) {
            return results[number - 1]
        }
        if lowerPrompt.contains("last result") || lowerPrompt.contains("last file") {
            return results.last
        }
        // A single visible result is an unambiguous target even when the user
        // says “rename this file” instead of spelling out “1st”.
        return results.count == 1 ? results[0] : nil
    }

    private static func ordinal(in prompt: String) -> Int? {
        let words: [String: Int] = [
            "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "sixth": 6, "6th": 6,
            "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
            "tenth": 10, "10th": 10
        ]
        for token in prompt.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" }) {
            let cleaned = token.trimmingCharacters(in: .punctuationCharacters)
            if let value = words[String(cleaned)] { return value }
            if let value = Int(cleaned), value > 0 { return value }
        }
        return nil
    }

    private static func capture(afterAny markers: [String], in text: String) -> String? {
        let lower = text.lowercased()
        guard let match = markers.compactMap({ marker -> (Int, String)? in
            guard let range = lower.range(of: marker) else { return nil }
            return (range.upperBound.utf16Offset(in: lower), marker)
        }).min(by: { $0.0 < $1.0 }) else { return nil }
        let start = text.index(text.startIndex, offsetBy: match.0)
        return String(text[start...]).trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
    }

    private static func cleanName(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
            .replacingOccurrences(of: "\"", with: "")
            .replacingOccurrences(of: "'", with: "")
    }

    private static func normalizedName(_ requested: String, source: String) -> String {
        let requested = cleanName(requested)
        guard !requested.contains("/"), !requested.contains("\\") else { return requested }
        let sourceExtension = URL(fileURLWithPath: source).pathExtension
        if !sourceExtension.isEmpty && URL(fileURLWithPath: requested).pathExtension.isEmpty {
            return "\(requested).\(sourceExtension)"
        }
        return requested
    }

    private static func resolveFolder(_ text: String) -> URL? {
        let cleaned = cleanName(text)
            .replacingOccurrences(of: "folder called ", with: "", options: .caseInsensitive)
        let home = FileManager.default.homeDirectoryForCurrentUser
        if cleaned.hasPrefix("/") {
            let absolute = URL(fileURLWithPath: cleaned).standardizedFileURL
            // Panda's current library is deliberately local-user-only. Do
            // not let a natural-language move escape into removable drives,
            // network shares, or cloud mounts.
            return absolute.path == home.path || absolute.path.hasPrefix(home.path + "/") ? absolute : nil
        }
        let known: [String: URL] = [
            "desktop": home.appendingPathComponent("Desktop"),
            "documents": home.appendingPathComponent("Documents"),
            "downloads": home.appendingPathComponent("Downloads"),
            "pictures": home.appendingPathComponent("Pictures"),
            "music": home.appendingPathComponent("Music"),
            "movies": home.appendingPathComponent("Movies")
        ]
        if let knownFolder = known[cleaned.lowercased()] { return knownFolder }
        let candidates = [home.appendingPathComponent(cleaned), home.appendingPathComponent("Desktop").appendingPathComponent(cleaned)]
        return candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) && (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true })
    }
}

private struct FinderActionOutcome {
    let message: String
    let oldPath: String?
    let updatedResult: SearchResult?
}

private enum FinderFileActions {
    static func copyToPasteboard(_ path: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([NSURL(fileURLWithPath: path)])
    }

    static func quickLook(_ path: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/qlmanage")
        process.arguments = ["-p", path]
        try? process.run()
    }

    static func execute(
        _ action: FinderPromptAction,
        apiService: APIService,
        indexingService: DocumentIndexingService
    ) async throws -> FinderActionOutcome {
        let fileManager = FileManager.default

        switch action {
        case .open(let result):
            NSWorkspace.shared.open(URL(fileURLWithPath: result.filePath))
            return FinderActionOutcome(message: "Opened \(result.fileName)", oldPath: nil, updatedResult: nil)

        case .openWith(let result, let appName):
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
            process.arguments = ["-a", appName, result.filePath]
            try process.run()
            return FinderActionOutcome(message: "Opened \(result.fileName) with \(appName)", oldPath: nil, updatedResult: nil)

        case .reveal(let result):
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: result.filePath)])
            return FinderActionOutcome(message: "Showing \(result.fileName) in Finder", oldPath: nil, updatedResult: nil)

        case .copy(let result):
            copyToPasteboard(result.filePath)
            return FinderActionOutcome(message: "Copied \(result.fileName) — paste it anywhere in Finder", oldPath: nil, updatedResult: nil)

        case .copyTo(let result, let destinationFolder):
            let source = URL(fileURLWithPath: result.filePath)
            guard fileManager.fileExists(atPath: destinationFolder.path) else {
                throw FinderActionError.destinationMissing(destinationFolder.path)
            }
            let destination = destinationFolder.appendingPathComponent(source.lastPathComponent)
            try validateSource(source, destination: destination, fileManager: fileManager)
            try fileManager.copyItem(at: source, to: destination)
            _ = await indexingService.indexFilesIfNeededConcurrent([destination], maxConcurrency: 1, forceReindex: true)
            return FinderActionOutcome(message: "Copied \(result.fileName) to \(destinationFolder.lastPathComponent)", oldPath: nil, updatedResult: nil)

        case .rename(let result, let requestedName):
            let source = URL(fileURLWithPath: result.filePath)
            let destination = source.deletingLastPathComponent().appendingPathComponent(requestedName)
            try validateSource(source, destination: destination, fileManager: fileManager)
            try fileManager.moveItem(at: source, to: destination)
            try? await apiService.deleteDocument(id: result.id)
            _ = await indexingService.indexFilesIfNeededConcurrent([destination], maxConcurrency: 1, forceReindex: true)
            let updated = SearchResult(
                id: result.id,
                content: result.content,
                fileName: destination.lastPathComponent,
                filePath: destination.path,
                mediaType: result.mediaType,
                thumbnailPath: result.thumbnailPath,
                score: result.score,
                matchReason: result.matchReason
            )
            return FinderActionOutcome(message: "Renamed to \(destination.lastPathComponent)", oldPath: result.filePath, updatedResult: updated)

        case .move(let result, let destinationFolder):
            let source = URL(fileURLWithPath: result.filePath)
            let folder = destinationFolder
            if !fileManager.fileExists(atPath: folder.path) {
                throw FinderActionError.destinationMissing(folder.path)
            }
            let destination = folder.appendingPathComponent(source.lastPathComponent)
            try validateSource(source, destination: destination, fileManager: fileManager)
            try fileManager.moveItem(at: source, to: destination)
            try? await apiService.deleteDocument(id: result.id)
            _ = await indexingService.indexFilesIfNeededConcurrent([destination], maxConcurrency: 1, forceReindex: true)
            let updated = SearchResult(
                id: result.id,
                content: result.content,
                fileName: destination.lastPathComponent,
                filePath: destination.path,
                mediaType: result.mediaType,
                thumbnailPath: result.thumbnailPath,
                score: result.score,
                matchReason: result.matchReason
            )
            return FinderActionOutcome(message: "Moved \(result.fileName) to \(folder.lastPathComponent)", oldPath: result.filePath, updatedResult: updated)

        case .duplicate(let result):
            let source = URL(fileURLWithPath: result.filePath)
            let destination = uniqueCopyURL(for: source, fileManager: fileManager)
            try validateSource(source, destination: destination, fileManager: fileManager)
            try fileManager.copyItem(at: source, to: destination)
            _ = await indexingService.indexFilesIfNeededConcurrent([destination], maxConcurrency: 1, forceReindex: true)
            return FinderActionOutcome(message: "Duplicated as \(destination.lastPathComponent)", oldPath: nil, updatedResult: nil)

        case .compress(let result):
            let source = URL(fileURLWithPath: result.filePath)
            let archive = source.deletingPathExtension().appendingPathExtension("zip")
            let output = uniqueCopyURL(for: archive, fileManager: fileManager)
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
            process.arguments = ["-c", "-k", "--sequesterRsrc", "--keepParent", source.path, output.path]
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { throw FinderActionError.commandFailed("compress") }
            _ = await indexingService.indexFilesIfNeededConcurrent([output], maxConcurrency: 1, forceReindex: true)
            return FinderActionOutcome(message: "Compressed to \(output.lastPathComponent)", oldPath: nil, updatedResult: nil)

        case .uncompress(let result):
            let source = URL(fileURLWithPath: result.filePath)
            let destination = source.deletingPathExtension()
            let output = uniqueCopyURL(for: destination, fileManager: fileManager)
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
            process.arguments = ["-x", "-k", source.path, output.path]
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { throw FinderActionError.commandFailed("uncompress") }
            return FinderActionOutcome(message: "Uncompressed into \(output.lastPathComponent)", oldPath: nil, updatedResult: nil)

        case .tag(let result, let tag):
            let url = URL(fileURLWithPath: result.filePath)
            var tags = (try? url.resourceValues(forKeys: [.tagNamesKey]).tagNames) ?? []
            if !tags.contains(tag) { tags.append(tag) }
            // tagNames is read-only on URLResourceValues; Finder writes it
            // through the resource-value key API instead.
            try (url as NSURL).setResourceValue(tags, forKey: URLResourceKey.tagNamesKey)
            return FinderActionOutcome(message: "Tagged \(result.fileName) as \(tag)", oldPath: nil, updatedResult: nil)

        case .trash(let result):
            var resultingURL: NSURL?
            try fileManager.trashItem(at: URL(fileURLWithPath: result.filePath), resultingItemURL: &resultingURL)
            try? await apiService.deleteDocument(id: result.id)
            return FinderActionOutcome(message: "Moved \(result.fileName) to Trash", oldPath: result.filePath, updatedResult: nil)

        case .getInfo(let result):
            let escaped = result.filePath.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
            let source = "tell application \"Finder\" to open information window of POSIX file \"\(escaped)\""
            var error: NSDictionary?
            NSAppleScript(source: source)?.executeAndReturnError(&error)
            if let error { throw FinderActionError.commandFailed(error.description) }
            return FinderActionOutcome(message: "Opened Finder info for \(result.fileName)", oldPath: nil, updatedResult: nil)

        case .createFolder(let name, let location):
            let folder = location.appendingPathComponent(name)
            if fileManager.fileExists(atPath: folder.path) { throw FinderActionError.destinationExists(folder.path) }
            try fileManager.createDirectory(at: folder, withIntermediateDirectories: false)
            return FinderActionOutcome(message: "Created folder \(name)", oldPath: nil, updatedResult: nil)
        }
    }

    private static func validateSource(_ source: URL, destination: URL, fileManager: FileManager) throws {
        guard fileManager.fileExists(atPath: source.path) else { throw FinderActionError.sourceMissing(source.path) }
        guard source.standardizedFileURL != destination.standardizedFileURL else { throw FinderActionError.sameLocation }
        guard !fileManager.fileExists(atPath: destination.path) else { throw FinderActionError.destinationExists(destination.path) }
    }

    private static func uniqueCopyURL(for original: URL, fileManager: FileManager) -> URL {
        let directory = original.deletingLastPathComponent()
        let ext = original.pathExtension
        let stem = original.deletingPathExtension().lastPathComponent
        var index = 1
        var candidate = original
        while fileManager.fileExists(atPath: candidate.path) {
            let suffix = index == 1 ? " copy" : " copy \(index)"
            let name = stem + suffix
            candidate = ext.isEmpty
                ? directory.appendingPathComponent(name)
                : directory.appendingPathComponent(name).appendingPathExtension(ext)
            index += 1
        }
        return candidate
    }
}

/// A memory-efficient search-history conversation. Images are represented by
/// their local file path and a small result record — Panda never duplicates
/// image data in history just to reopen a prior result grid.
private struct SearchHistoryEntry: Identifiable, Codable {
    let id: UUID
    let query: String
    let results: [SearchHistoryResult]

    init(query: String, results: [SearchResult]) {
        id = UUID()
        self.query = query
        self.results = results.map(SearchHistoryResult.init)
    }
}

private struct SearchHistoryResult: Identifiable, Codable {
    let id: String
    let content: String
    let fileName: String
    let filePath: String
    let mediaType: MediaType?
    let thumbnailPath: String?
    let score: Double
    let matchReason: String?

    init(_ result: SearchResult) {
        id = result.id
        fileName = result.fileName
        filePath = result.filePath
        mediaType = result.mediaType
        thumbnailPath = result.thumbnailPath
        score = result.score
        matchReason = result.matchReason

        if result.mediaType == .image {
            // LocalResultPreview loads this file on demand using filePath. Do
            // not retain a large VLM caption or image bytes for history.
            content = "[saved image result] \(result.fileName)"
        } else {
            // Text/audio previews remain useful in a restored conversation,
            // but are capped so many history entries stay inexpensive.
            content = String(result.content.prefix(600))
        }
    }

    var searchResult: SearchResult {
        SearchResult(
            id: id,
            content: content,
            fileName: fileName,
            filePath: filePath,
            mediaType: mediaType,
            thumbnailPath: thumbnailPath,
            score: score,
            matchReason: matchReason
        )
    }
}

private enum FinderActionError: LocalizedError {
    case sourceMissing(String)
    case destinationMissing(String)
    case destinationExists(String)
    case sameLocation
    case commandFailed(String)

    var errorDescription: String? {
        switch self {
        case .sourceMissing(let path): return "Source file is missing: \(path)"
        case .destinationMissing(let path): return "Destination folder does not exist: \(path)"
        case .destinationExists(let path): return "A file already exists at: \(path)"
        case .sameLocation: return "The source and destination are the same."
        case .commandFailed(let command): return "Finder action failed: \(command)"
        }
    }
}

private struct TaskResultRow: View {
    let result: SearchResult
    let serial: Int
    let isExpanded: Bool
    let onOpen: () -> Void
    let onShowInFinder: () -> Void
    let onAction: (ResultCardAction) -> Void

    private var previewSize: CGFloat { isExpanded ? 132 : 64 }

    private var summary: String {
        let compact = result.content
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // Restored image history deliberately contains only the local path
        // and filename. The filename is already shown in the title, and the
        // match explanation is more useful than repeating it as body text.
        if compact.hasPrefix("[saved image result]") {
            return ""
        }
        // Backend records carry a compact metadata prefix for retrieval. The
        // card should lead with the indexed explanation/transcript instead of
        // repeating implementation details such as “document, pdf format”.
        let withoutMetadata: String
        if compact.hasPrefix("["), let closingBracket = compact.firstIndex(of: "]") {
            withoutMetadata = String(compact[compact.index(after: closingBracket)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            withoutMetadata = compact
        }
        return withoutMetadata.isEmpty
            ? "No indexed description is available for this file."
            : String(withoutMetadata.prefix(isExpanded ? 260 : 190))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 13) {
            HStack(alignment: .top, spacing: 12) {
                ZStack(alignment: .bottomTrailing) {
                    LocalResultPreview(result: result, size: previewSize)
                    Text("\(serial)")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundStyle(PandaPalette.canvas)
                        .frame(width: 19, height: 19)
                        .background(PandaPalette.mint, in: Circle())
                        .overlay(Circle().stroke(PandaPalette.panel, lineWidth: 2))
                        .offset(x: 5, y: 5)
                }
                VStack(alignment: .leading, spacing: 5) {
                    Text(result.fileName)
                        .font(.system(size: isExpanded ? 17 : 15, weight: .semibold))
                        .lineLimit(2)
                }
                Spacer(minLength: 0)
            }

            if let matchReason = result.matchReason,
               !matchReason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Label(matchReason, systemImage: "sparkle.magnifyingglass")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(PandaPalette.mint.opacity(0.9))
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(PandaPalette.mint.opacity(0.08), in: RoundedRectangle(cornerRadius: 9))
            }

            if !summary.isEmpty {
                Text(summary)
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.62))
                    .lineLimit(isExpanded ? 3 : 2)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            Spacer(minLength: 0)
            HStack(spacing: 9) {
                Button(action: onOpen) {
                    Label("Open", systemImage: "arrow.up.forward.app")
                        .font(.system(size: 12, weight: .semibold))
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(PandaPalette.mint)

                Button(action: onShowInFinder) {
                    Label("Show in Finder", systemImage: "folder")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.bordered)
                .tint(PandaPalette.mint.opacity(0.7))

                Menu {
                    Button { onAction(.quickLook) } label: { Label("Quick Look", systemImage: "eye") }
                    Button { onAction(.openWith) } label: { Label("Open With…", systemImage: "square.and.arrow.up") }
                    Button { onAction(.copy) } label: { Label("Copy", systemImage: "doc.on.doc") }
                    Button { onAction(.copyPath) } label: { Label("Copy Path", systemImage: "link") }
                    Divider()
                    Button { onAction(.rename) } label: { Label("Rename with Panda…", systemImage: "pencil") }
                    Button { onAction(.move) } label: { Label("Move with Panda…", systemImage: "folder") }
                    Button { onAction(.duplicate) } label: { Label("Duplicate", systemImage: "plus.square.on.square") }
                    Button { onAction(.compress) } label: { Label("Compress", systemImage: "archivebox") }
                    Button { onAction(.uncompress) } label: { Label("Uncompress", systemImage: "archivebox.fill") }
                    Button { onAction(.tag) } label: { Label("Add Panda tag", systemImage: "tag") }
                    Divider()
                    Button(role: .destructive) { onAction(.trash) } label: { Label("Move to Trash", systemImage: "trash") }
                    Button { onAction(.getInfo) } label: { Label("Get Info", systemImage: "info.circle") }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.system(size: 17, weight: .semibold))
                }
                .menuStyle(.borderlessButton)
                .tint(PandaPalette.mint.opacity(0.8))
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, minHeight: isExpanded ? 230 : 185, alignment: .topLeading)
        .background(PandaPalette.panel.opacity(0.72), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(.white.opacity(0.09)))
    }
}

/// A local, inspectable record of every image Panda has successfully indexed.
/// The file hash in IndexedFile is the scan-once key; an edited file receives
/// a new record, while an unchanged image never asks the VLM again.
private enum ImageIndexViewMode: String, CaseIterable, Identifiable {
    case list
    case grid

    var id: String { rawValue }
    var title: String { rawValue.capitalized }
    var icon: String { self == .list ? "list.bullet" : "square.grid.2x2" }
}

private struct ImageIndexWorkspace: View {
    @Query(
        filter: #Predicate<IndexedFile> { $0.mediaType == "image" },
        sort: \IndexedFile.indexedAt,
        order: .reverse
    ) private var images: [IndexedFile]

    let onBack: () -> Void
    @State private var selectedPath: String?
    @State private var filter = ""
    @State private var viewMode: ImageIndexViewMode = .list

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
                Button(action: onBack) {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 13, weight: .bold))
                        .frame(width: 36, height: 36)
                        .background(PandaPalette.panel, in: Circle())
                        .overlay(Circle().stroke(PandaPalette.stroke))
                }
                .buttonStyle(.plain).foregroundStyle(PandaPalette.mint)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Image Index").font(.system(size: 22, weight: .bold, design: .rounded))
                    Text("\(images.count) indexed • \(describedImageCount) VLM-described • \(max(0, images.count - describedImageCount)) waiting")
                        .font(.system(size: 12)).foregroundStyle(.white.opacity(0.53))
                }
                Spacer()
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.white.opacity(0.45))
                    TextField("Search indexed files", text: $filter).textFieldStyle(.plain)
                }
                .padding(.horizontal, 13).frame(width: 250, height: 38)
                .background(PandaPalette.panel, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(PandaPalette.stroke))

                Picker("View", selection: $viewMode) {
                    ForEach(ImageIndexViewMode.allCases) { mode in
                        Label(mode.title, systemImage: mode.icon).tag(mode)
                    }
                }
                .labelsHidden().pickerStyle(.segmented).frame(width: 150)
            }
            .padding(.horizontal, 22).padding(.vertical, 16).background(PandaPalette.topbar)
            Divider().overlay(PandaPalette.stroke)

            if images.isEmpty {
                ContentUnavailableView(
                    "No indexed images yet",
                    systemImage: "photo.stack",
                    description: Text("Scan the Local Mac or Add Library Folder. Panda will add each supported image here after its first scan.")
                )
                .foregroundStyle(.white.opacity(0.75))
            } else {
                HSplitView {
                    Group {
                        if viewMode == .list {
                            VStack(spacing: 0) {
                                imageTableHeader
                                Divider().overlay(PandaPalette.stroke)
                                ScrollView {
                                    LazyVStack(spacing: 6) {
                                        ForEach(filteredImages) { image in
                                            ImageIndexRow(image: image, isSelected: image.filePath == selectedImage?.filePath) {
                                                selectedPath = image.filePath
                                            }
                                        }
                                    }
                                    .padding(10)
                                }
                            }
                        } else {
                            ScrollView {
                                LazyVGrid(
                                    columns: [GridItem(.adaptive(minimum: 170, maximum: 230), spacing: 12)],
                                    spacing: 12
                                ) {
                                    ForEach(filteredImages) { image in
                                        ImageIndexGridCard(image: image, isSelected: image.filePath == selectedImage?.filePath) {
                                            selectedPath = image.filePath
                                        }
                                    }
                                }
                                .padding(14)
                            }
                        }
                    }
                    .frame(minWidth: 420, maxWidth: .infinity)
                    ImageIndexPreview(image: selectedImage)
                        .frame(minWidth: 260, idealWidth: 330)
                        .padding(10)
                }
            }
        }
        .background(PandaPalette.workspace)
        .preferredColorScheme(.dark)
    }

    private var imageTableHeader: some View {
        HStack(spacing: 12) {
            Text("IMAGE").frame(width: 56, alignment: .leading)
            Text("PANDA SEES").frame(maxWidth: .infinity, alignment: .leading)
            Text("INDEXED").frame(width: 78, alignment: .leading)
        }
        .font(.system(size: 10, weight: .bold)).tracking(0.9).foregroundStyle(.white.opacity(0.42))
        .padding(.horizontal, 20).padding(.vertical, 11)
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
            .frame(maxWidth: .infinity, alignment: .leading)
            Text(image.indexedAt, format: .dateTime.month(.abbreviated).day())
                .font(.system(size: 11)).foregroundStyle(.white.opacity(0.5)).frame(width: 78, alignment: .leading)
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(isSelected ? PandaPalette.mint.opacity(0.12) : PandaPalette.panel.opacity(0.42), in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 13, style: .continuous).stroke(isSelected ? PandaPalette.mint.opacity(0.35) : PandaPalette.stroke))
        .contentShape(Rectangle()).onTapGesture(perform: select)
    }
}

private struct ImageIndexGridCard: View {
    let image: IndexedFile
    let isSelected: Bool
    let select: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            LocalImageThumbnail(path: image.filePath)
                .frame(maxWidth: .infinity)
                .aspectRatio(1.28, contentMode: .fit)
            Text(image.visualTitle?.isEmpty == false ? image.visualTitle! : image.fileName)
                .font(.system(size: 13, weight: .semibold)).lineLimit(1)
            Text(image.visualExplanation ?? "Waiting for Panda's visual explanation")
                .font(.system(size: 11)).foregroundStyle(.white.opacity(0.55)).lineLimit(2)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(isSelected ? PandaPalette.mint.opacity(0.12) : PandaPalette.panel.opacity(0.5), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(isSelected ? PandaPalette.mint.opacity(0.4) : PandaPalette.stroke))
        .contentShape(Rectangle()).onTapGesture(perform: select)
    }
}

private struct ImageIndexPreview: View {
    let image: IndexedFile?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let image, let nsImage = NSImage(contentsOf: URL(fileURLWithPath: image.filePath)) {
                Image(nsImage: nsImage).resizable().scaledToFit().frame(maxWidth: .infinity, maxHeight: 300)
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                Text(image.fileName)
                    .font(.system(size: 18, weight: .bold))
                Text("PANDA SEES")
                    .font(.system(size: 10, weight: .bold)).tracking(1).foregroundStyle(PandaPalette.mint)
                Text(image.visualExplanation ?? "No visual explanation is available yet.")
                    .font(.system(size: 13)).foregroundStyle(.white.opacity(0.7)).lineSpacing(3)
                Divider().overlay(.white.opacity(0.1))
                HStack(spacing: 10) {
                    Button("Open") { NSWorkspace.shared.open(URL(fileURLWithPath: image.filePath)) }
                        .buttonStyle(.borderedProminent).tint(PandaPalette.mint)
                    Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: image.filePath)]) }
                        .buttonStyle(.bordered).tint(PandaPalette.mint)
                }
            } else {
                ContentUnavailableView("Select an image", systemImage: "photo", description: Text("Choose an indexed image to see what Panda understands."))
            }
            Spacer()
        }
        .padding(22)
        .background(PandaPalette.result, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(PandaPalette.stroke))
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
    var size: CGFloat = 58

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
        .frame(width: size, height: size)
        .background(PandaPalette.mint.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: min(size / 5, 14), style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: min(size / 5, 14)).stroke(.white.opacity(0.1)))
    }
}

private enum PandaPalette {
    // Neutral graphite surfaces model the floating vehicle-dashboard
    // reference while Panda's mint remains the single functional accent.
    static let canvas = Color(red: 0.055, green: 0.058, blue: 0.06)
    static let workspace = Color(red: 0.075, green: 0.08, blue: 0.082)
    static let sidebar = Color(red: 0.085, green: 0.09, blue: 0.092)
    static let panel = Color(red: 0.12, green: 0.125, blue: 0.128)
    static let elevated = Color(red: 0.145, green: 0.15, blue: 0.153)
    static let result = Color(red: 0.105, green: 0.11, blue: 0.113)
    static let topbar = Color(red: 0.095, green: 0.10, blue: 0.102)
    static let glow = Color(red: 0.19, green: 0.23, blue: 0.21)
    static let stroke = Color.white.opacity(0.075)
    static let strokeStrong = Color.white.opacity(0.13)
    static let mint = Color(red: 0.30, green: 0.87, blue: 0.61)
}
