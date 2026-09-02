//
//  DocumentIndexingService.swift
//  ManeAI
//
//  Smart document indexing service with deduplication
//

import Foundation
import SwiftData
import Combine

/// Result of an indexing operation
enum IndexingResult {
    case indexed(id: String, fileName: String)
    case alreadyIndexed(id: String, fileName: String)
    case failed(error: Error)

    var isSuccess: Bool {
        switch self {
        case .indexed, .alreadyIndexed: return true
        case .failed: return false
        }
    }

    var documentId: String? {
        switch self {
        case .indexed(let id, _), .alreadyIndexed(let id, _): return id
        case .failed: return nil
        }
    }

    var fileName: String? {
        switch self {
        case .indexed(_, let name), .alreadyIndexed(_, let name): return name
        case .failed: return nil
        }
    }

    var wasNewlyIndexed: Bool {
        if case .indexed = self { return true }
        return false
    }
}

/// Service for smart document indexing with caching and deduplication
@MainActor
class DocumentIndexingService: ObservableObject {
    private let apiService: APIService
    private let modelContext: ModelContext?

    @Published var isIndexing = false
    @Published var indexingProgress: String = ""
    @Published var lastIndexedFile: String?
    /// Number of files that failed during the current full-library pass.
    /// ContentView uses this to keep an interrupted scan resumable instead of
    /// marking the library complete while some files were never indexed.
    var lastScanFailureCount: Int = 0

    init(apiService: APIService, modelContext: ModelContext?) {
        self.apiService = apiService
        self.modelContext = modelContext
    }

    // MARK: - Smart Indexing

    /// Index a file if it hasn't been indexed or has changed
    /// - Parameters:
    ///   - url: URL of the file to index
    ///   - forceReindex: If true, reindex even if file hasn't changed
    /// - Returns: IndexingResult indicating success/skip/failure
    func indexFileIfNeeded(_ url: URL, forceReindex: Bool = false) async -> IndexingResult {
        isIndexing = true
        indexingProgress = "Checking file..."

        defer {
            isIndexing = false
            indexingProgress = ""
        }

        let fileName = url.lastPathComponent
        let filePath = url.path

        // Get file attributes
        guard let attrs = FileHasher.attributes(fileAt: url) else {
            return .failed(error: IndexingError.fileNotAccessible)
        }

        // Compute content hash
        indexingProgress = "Computing hash..."
        guard let contentHash = FileHasher.hash(fileAt: url) else {
            return .failed(error: IndexingError.hashComputationFailed)
        }

        // Determine media type before consulting the cache. Older image
        // records have no visual title/explanation, so they receive exactly
        // one upgrade pass and then participate in normal hash deduplication.
        let mediaType = determineMediaType(for: url)
        let existingFile = findIndexedFile(path: filePath, hash: contentHash)
        let needsVisualRecord = mediaType == .image && !hasVisualRecord(existingFile)

        // Check if already indexed with same hash.
        if !forceReindex, !needsVisualRecord, let existingFile {
            print("📚 File already indexed: \(fileName) (hash match)")
            lastIndexedFile = fileName
            return .alreadyIndexed(id: existingFile.id, fileName: existingFile.fileName)
        }

        // Index the file
        indexingProgress = "Indexing \(fileName)..."

        do {
            let response: IngestResponse
            let indexingStartedAt = Date()

            // Let the backend extract rich/large text through its bounded
            // reader. Swift never loads a multi-gigabyte log into memory.
            let localVision: String?
            switch mediaType {
            case .image:
                localVision = await LocalVisionOCR.recognizeText(in: url)
            case .video:
                localVision = await LocalVisionOCR.recognizeVideo(in: url)
            default:
                localVision = nil
            }
            response = try await apiService.ingestMediaFile(
                filePath: filePath,
                mediaType: mediaType,
                content: localVision?.isEmpty == false ? localVision : nil,
                forceReindex: forceReindex || needsVisualRecord
            )

            // Store in local database
            saveIndexedFile(
                id: response.id,
                filePath: filePath,
                contentHash: contentHash,
                fileSize: attrs.size,
                fileModifiedAt: attrs.modified,
                mediaType: mediaType.rawValue,
                fileName: response.fileName,
                fileExtension: url.pathExtension.lowercased(),
                indexingDurationMs: response.elapsedMs.map(Double.init) ?? Date().timeIntervalSince(indexingStartedAt) * 1000,
                visualTitle: response.imageTitle,
                visualExplanation: response.imageDescription
            )

            print("✅ Indexed new file: \(fileName)")
            lastIndexedFile = fileName
            return .indexed(id: response.id, fileName: response.fileName)

        } catch {
            print("❌ Failed to index \(fileName): \(error)")
            return .failed(error: error)
        }
    }

    /// Index multiple files, skipping already-indexed ones (legacy sequential)
    func indexFilesIfNeeded(_ urls: [URL]) async -> [IndexingResult] {
        // Use concurrent processing by default for better performance
        return await indexFilesIfNeededConcurrent(urls)
    }

    /// Index multiple files concurrently using TaskGroup (up to 100x faster)
    /// - Parameters:
    ///   - urls: Files to index
    ///   - maxConcurrency: Maximum parallel operations (default: 10)
    /// - Returns: Array of IndexingResult for each file
    func indexFilesIfNeededConcurrent(
        _ urls: [URL],
        maxConcurrency: Int = 10,
        forceReindex: Bool = false,
        forceTextReindex: Bool = false
    ) async -> [IndexingResult] {
        guard !urls.isEmpty else { return [] }

        isIndexing = true
        indexingProgress = "Preparing \(urls.count) files..."

        defer {
            isIndexing = false
            indexingProgress = ""
        }

        // First pass: Check which files need indexing (parallel hash computation)
        var filesToIndex: [(index: Int, url: URL, content: String?, mediaType: MediaType)] = []
        var cachedResults: [(index: Int, result: IndexingResult)] = []

        // Compute hashes and check cache in parallel
        await withTaskGroup(of: (Int, URL, String?, IndexingResult?).self) { group in
            for (index, url) in urls.enumerated() {
                let mediaType = determineMediaType(for: url)
                group.addTask {
                    let filePath = url.path

                    // Compute hash
                    guard let contentHash = FileHasher.hash(fileAt: url) else {
                        return (index, url, nil, .failed(error: IndexingError.hashComputationFailed))
                    }

                    let shouldForceReindex = forceReindex || (forceTextReindex && mediaType == .text)

                    // Check if already indexed. An old image row can have a
                    // hash and embedding but no VLM title/description; that
                    // row is deliberately eligible for one visual upgrade.
                    if !shouldForceReindex, let existingFile = await self.findIndexedFile(path: filePath, hash: contentHash) {
                        let visualComplete = mediaType == .image
                            ? await self.hasVisualRecord(existingFile)
                            : true
                        if visualComplete {
                            return (index, url, contentHash, .alreadyIndexed(id: existingFile.id, fileName: existingFile.fileName))
                        }
                    }

                    return (index, url, contentHash, nil)
                }
            }

            for await (index, url, hash, cachedResult) in group {
                if let result = cachedResult {
                    cachedResults.append((index, result))
                } else if hash != nil {
                    let mediaType = determineMediaType(for: url)

                    // An existing image made by a previous build may have a
                    // vector but not the table fields. Treat that as a one
                    // time upgrade, not a permanently cached result.
                    if mediaType == .image,
                       let existing = findIndexedFile(path: url.path, hash: hash!),
                       !hasVisualRecord(existing) {
                        filesToIndex.append((index, url, nil, mediaType))
                        continue
                    }

                    // Text extraction happens in the backend with a bounded
                    // head/tail reader; do not preload large files in Swift.
                    filesToIndex.append((index, url, nil, mediaType))
                }
            }
        }

        // If all files are cached, return early
        if filesToIndex.isEmpty {
            indexingProgress = "All files already indexed"
            return cachedResults.sorted(by: { $0.index < $1.index }).map(\.result)
        }

        indexingProgress = "Indexing \(filesToIndex.count) new files..."

        // Separate by media type for batch processing
        let textFiles = filesToIndex.filter { $0.mediaType == .text }
        let mediaFiles = filesToIndex.filter { $0.mediaType != .text }

        var indexedResults: [(index: Int, result: IndexingResult)] = []

        // Text files are sent in one bounded batch. LanceDB can append a batch
        // in one fragment and generate embeddings in small groups, which is
        // dramatically faster than opening a write transaction per file.
        // Individual SwiftData rows still retain each file's response timing.
        if !textFiles.isEmpty {
            indexedResults.append(contentsOf: await indexTextBatch(
                textFiles,
                forceReindex: forceReindex || forceTextReindex
            ))
        }

        // Process media files concurrently (audio/image need individual processing)
        if !mediaFiles.isEmpty {
            await withTaskGroup(of: (Int, IndexingResult).self) { group in
                // Use semaphore pattern for controlled concurrency
                let semaphore = AsyncSemaphore(limit: maxConcurrency)

                for file in mediaFiles {
                    group.addTask {
                        await semaphore.wait()

                        let result = await self.indexSingleFile(file.url, content: file.content, mediaType: file.mediaType, forceReindex: forceReindex)

                        await semaphore.signal()
                        return (file.index, result)
                    }
                }

                for await (index, result) in group {
                    indexedResults.append((index, result))
                }
            }
        }

        // Combine and sort results
        let allResults = (cachedResults + indexedResults).sorted(by: { $0.index < $1.index })

        let newlyIndexed = indexedResults.filter { $0.result.wasNewlyIndexed }.count
        indexingProgress = "Done! Indexed \(newlyIndexed) new files"

        return allResults.map(\.result)
    }

    /// Extract and embed a bounded group of text files in one backend call.
    /// The backend owns format extraction (PDF/Office/RTF/source) so the Mac
    /// client never loads a large document into memory just to send it twice.
    private func indexTextBatch(
        _ files: [(index: Int, url: URL, content: String?, mediaType: MediaType)],
        forceReindex: Bool
    ) async -> [(index: Int, result: IndexingResult)] {
        let requests = files.map { file in
            IngestRequest(
                content: nil,
                filePath: file.url.path,
                mediaType: .text,
                metadata: nil,
                forceReindex: forceReindex ? true : nil
            )
        }
        do {
            let batch = try await apiService.batchIngest(files: requests, concurrency: 1)
            let byPath = Dictionary(uniqueKeysWithValues: batch.results.map { ($0.filePath, $0) })
            return files.map { file in
                guard let response = byPath[file.url.path], response.success,
                      let attrs = FileHasher.attributes(fileAt: file.url),
                      let hash = FileHasher.hash(fileAt: file.url) else {
                    return (file.index, .failed(error: IndexingError.fileNotAccessible))
                }
                saveIndexedFile(
                    id: response.id,
                    filePath: file.url.path,
                    contentHash: hash,
                    fileSize: attrs.size,
                    fileModifiedAt: attrs.modified,
                    mediaType: MediaType.text.rawValue,
                    fileName: response.fileName,
                    fileExtension: file.url.pathExtension.lowercased(),
                    indexingDurationMs: response.elapsedMs.map(Double.init) ?? Double(batch.elapsedMs) / Double(max(1, files.count)),
                    visualTitle: nil,
                    visualExplanation: nil
                )
                return (file.index, .indexed(id: response.id, fileName: response.fileName))
            }
        } catch {
            return files.map { ($0.index, .failed(error: error)) }
        }
    }

    /// Index a single file (used for media files that need individual processing)
    private func indexSingleFile(_ url: URL, content: String?, mediaType: MediaType, forceReindex: Bool = false) async -> IndexingResult {
        let fileName = url.lastPathComponent
        let filePath = url.path
        let needsVisualRecord = mediaType == .image && !hasVisualRecord(
            FileHasher.hash(fileAt: url).flatMap { findIndexedFile(path: filePath, hash: $0) }
        )

        do {
            let response: IngestResponse
            let indexingStartedAt = Date()

            let localVision: String?
            switch mediaType {
            case .image:
                localVision = await LocalVisionOCR.recognizeText(in: url)
            case .video:
                localVision = await LocalVisionOCR.recognizeVideo(in: url)
            default:
                localVision = nil
            }
            response = try await apiService.ingestMediaFile(
                filePath: filePath,
                mediaType: mediaType,
                content: localVision?.isEmpty == false ? localVision : nil,
                forceReindex: forceReindex || needsVisualRecord
            )

            // Save to local cache
            if let attrs = FileHasher.attributes(fileAt: url),
               let hash = FileHasher.hash(fileAt: url) {
                saveIndexedFile(
                    id: response.id,
                    filePath: filePath,
                    contentHash: hash,
                    fileSize: attrs.size,
                    fileModifiedAt: attrs.modified,
                    mediaType: mediaType.rawValue,
                    fileName: response.fileName,
                    fileExtension: url.pathExtension.lowercased(),
                    indexingDurationMs: response.elapsedMs.map(Double.init) ?? Date().timeIntervalSince(indexingStartedAt) * 1000,
                    visualTitle: response.imageTitle,
                    visualExplanation: response.imageDescription
                )
            }

            return .indexed(id: response.id, fileName: response.fileName)
        } catch {
            print("❌ Failed to index \(fileName): \(error)")
            return .failed(error: error)
        }
    }

    /// Check if a file needs indexing (changed or not indexed)
    func needsIndexing(_ url: URL) -> Bool {
        guard let contentHash = FileHasher.hash(fileAt: url) else {
            return true // Assume needs indexing if can't read
        }
        return findIndexedFile(path: url.path, hash: contentHash) == nil
    }

    /// Get the document ID for an already-indexed file
    func getIndexedFileId(for url: URL) -> String? {
        guard let contentHash = FileHasher.hash(fileAt: url) else { return nil }
        return findIndexedFile(path: url.path, hash: contentHash)?.id
    }

    /// Clear only image index metadata before a brand-new visual session.
    /// Original files in Finder are never touched.
    func clearImageIndex() {
        guard let context = modelContext else { return }
        let descriptor = FetchDescriptor<IndexedFile>(
            predicate: #Predicate { file in file.mediaType == "image" }
        )
        if let files = try? context.fetch(descriptor) {
            files.forEach(context.delete)
            try? context.save()
        }
    }

    /// Collapse duplicate local rows left by older development scans. The
    /// newest row with a real VLM description wins; the original Finder files
    /// are never modified.
    func deduplicateImageRecords() {
        guard let context = modelContext else { return }
        let descriptor = FetchDescriptor<IndexedFile>(
            predicate: #Predicate { file in file.mediaType == "image" }
        )
        guard let files = try? context.fetch(descriptor) else { return }
        var grouped: [String: [IndexedFile]] = [:]
        for file in files { grouped[file.filePath, default: []].append(file) }

        for duplicates in grouped.values where duplicates.count > 1 {
            let sorted = duplicates.sorted { lhs, rhs in
                let lhsReady = hasVisualRecord(lhs)
                let rhsReady = hasVisualRecord(rhs)
                if lhsReady != rhsReady { return lhsReady }
                return lhs.indexedAt > rhs.indexedAt
            }
            for duplicate in sorted.dropFirst() { context.delete(duplicate) }
        }
        try? context.save()
    }

    /// Remove metadata for images that have been deleted or moved. This keeps
    /// the Image Index count faithful to the files currently present in Finder.
    /// It never deletes or moves a source file.
    func pruneMissingImageRecords() {
        guard let context = modelContext else { return }
        let excludedComponents: Set<String> = [
            "Library", "Applications", ".gemini", ".codex", ".antigravity", ".cargo", ".rustup", ".lmstudio", ".venv",
            ".venv_paddlevl", "node_modules", ".git", ".next", "dist", "build",
            "DerivedData", "Caches", "Cache", ".cache", ".Trash", ".Trashes",
            ".Spotlight-V100", ".fseventsd", ".DocumentRevisions-V100", ".TemporaryItems",
            "Photos Library.photoslibrary", "Photo Booth Library.photobooth"
        ]
        // Reconcile every media type. Earlier builds could persist text rows
        // from a mounted volume too, and the local-only policy applies to the
        // complete derived table, not only the Image Index view.
        let descriptor = FetchDescriptor<IndexedFile>()
        guard let files = try? context.fetch(descriptor) else { return }
        let homePath = FileManager.default.homeDirectoryForCurrentUser.path
        let homePrefix = homePath.hasSuffix("/") ? homePath : homePath + "/"
        let cloudPrefixes = [
            homePrefix + "Library/CloudStorage/",
            homePrefix + "Library/Mobile Documents/"
        ]
        for file in files {
            let pathComponents = URL(fileURLWithPath: file.filePath).pathComponents
            // scanBatches uses .skipsHiddenFiles, so stale rows from older
            // builds must apply the same rule (for example .vercel output or
            // hidden screenshot files) to keep the table reconcilable.
            let containsHiddenEntry = pathComponents.contains { component in
                component != "/" && component.hasPrefix(".")
            }
            // The initial library is local-user-only. Remove old rows created
            // by earlier builds that traversed removable/network/cloud roots;
            // this only changes Panda's derived index, never the source file.
            let outsideLocalUserSpace = !file.filePath.hasPrefix(homePrefix) ||
                cloudPrefixes.contains(where: { file.filePath.hasPrefix($0) })
            if !FileManager.default.fileExists(atPath: file.filePath) ||
                containsHiddenEntry ||
                outsideLocalUserSpace ||
                !excludedComponents.isDisjoint(with: Set(pathComponents)) {
                context.delete(file)
            }
        }
        try? context.save()
    }

    /// Return image files whose local record still needs a real VLM title and
    /// explanation. This is used for a bounded recovery pass after a scan if
    /// the vision server was temporarily unavailable for a batch.
    func incompleteImageURLs() -> [URL] {
        guard let context = modelContext else { return [] }
        let descriptor = FetchDescriptor<IndexedFile>(
            predicate: #Predicate { file in file.mediaType == "image" }
        )
        guard let files = try? context.fetch(descriptor) else { return [] }
        return files.compactMap { file in
            guard !hasVisualRecord(file),
                  FileManager.default.fileExists(atPath: file.filePath) else { return nil }
            return URL(fileURLWithPath: file.filePath)
        }
    }

    // MARK: - Private Helpers

    private func findIndexedFile(path: String, hash: String) -> IndexedFile? {
        guard let context = modelContext else { return nil }

        let descriptor = FetchDescriptor<IndexedFile>(
            predicate: #Predicate { file in
                file.filePath == path && file.contentHash == hash
            }
        )

        return try? context.fetch(descriptor).first
    }

    private func findIndexedFileByPath(_ path: String) -> IndexedFile? {
        guard let context = modelContext else { return nil }

        let descriptor = FetchDescriptor<IndexedFile>(
            predicate: #Predicate { file in
                file.filePath == path
            }
        )

        return try? context.fetch(descriptor).first
    }

    private func hasVisualRecord(_ file: IndexedFile?) -> Bool {
        guard let file else { return false }
        let title = file.visualTitle?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let explanation = file.visualExplanation?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        // Filename-only fallbacks are deliberately retryable. A temporary
        // model outage or an unsupported source format must not permanently
        // mark an image as understood.
        let isFilenameFallback = explanation.hasPrefix("Image file:")
        return !title.isEmpty && !explanation.isEmpty && !isFilenameFallback
    }

    private func saveIndexedFile(
        id: String,
        filePath: String,
        contentHash: String,
        fileSize: Int64,
        fileModifiedAt: Date,
        mediaType: String,
        fileName: String,
        fileExtension: String? = nil,
        indexingDurationMs: Double? = nil,
        visualTitle: String? = nil,
        visualExplanation: String? = nil
    ) {
        guard let context = modelContext else { return }

        // Remove any existing record for this path
        if let existing = findIndexedFileByPath(filePath) {
            context.delete(existing)
        }

        // Create new record
        let indexedFile = IndexedFile(
            id: id,
            filePath: filePath,
            contentHash: contentHash,
            fileSize: fileSize,
            fileModifiedAt: fileModifiedAt,
            mediaType: mediaType,
            fileName: fileName,
            fileExtension: fileExtension,
            indexingDurationMs: indexingDurationMs,
            visualTitle: visualTitle,
            visualExplanation: visualExplanation
        )

        context.insert(indexedFile)
        try? context.save()
    }

    private func determineMediaType(for url: URL) -> MediaType {
        let ext = url.pathExtension.lowercased()

        // Image types
        let imageExtensions = ["jpg", "jpeg", "png", "gif", "webp", "heic", "heif", "tiff", "bmp", "svg", "ico", "icns", "avif"]
        if imageExtensions.contains(ext) {
            return .image
        }

        // Audio types
        let audioExtensions = ["mp3", "wav", "m4a", "aac", "ogg", "flac", "aiff", "wma"]
        if audioExtensions.contains(ext) {
            return .audio
        }

        let videoExtensions = ["mp4", "mov", "m4v", "avi", "mkv", "webm"]
        if videoExtensions.contains(ext) {
            return .video
        }

        // Default to text
        return .text
    }
}

// MARK: - Indexing Errors

enum IndexingError: LocalizedError {
    case fileNotAccessible
    case hashComputationFailed
    case contentReadFailed
    case alreadyIndexing

    var errorDescription: String? {
        switch self {
        case .fileNotAccessible:
            return "Cannot access the file"
        case .hashComputationFailed:
            return "Failed to compute file hash"
        case .contentReadFailed:
            return "Failed to read file content"
        case .alreadyIndexing:
            return "Already indexing a file"
        }
    }
}

// MARK: - Async Semaphore for Controlled Concurrency

/// A simple async semaphore for limiting concurrent operations
actor AsyncSemaphore {
    private var count: Int
    private var waiters: [CheckedContinuation<Void, Never>] = []

    init(limit: Int) {
        self.count = limit
    }

    func wait() async {
        if count > 0 {
            count -= 1
            return
        }

        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func signal() {
        if let waiter = waiters.first {
            waiters.removeFirst()
            waiter.resume()
        } else {
            count += 1
        }
    }
}
