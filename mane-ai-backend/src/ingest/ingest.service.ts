import { Injectable, Logger } from '@nestjs/common';
import { LanceDBService } from '../lancedb';
import { MultimodalService, MediaType } from '../multimodal';
import { ImageCaptioningService } from '../image-captioning';
import {
  IngestDocumentDto,
  IngestResponseDto,
  DocumentListResponseDto,
} from './dto/ingest.dto';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { execFile } from 'child_process';
import { promisify } from 'util';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { PDFParse } = require('pdf-parse');
import * as mammoth from 'mammoth';
import * as XLSX from 'xlsx';

const CHUNK_WORD_COUNT = 280;
const CHUNK_OVERLAP_WORDS = 50;
const MIN_CONTENT_FOR_CHUNKING = 800;
// Keep very large source/log files searchable without creating thousands of
// Lance rows (the Finder table still represents one file). The first and last
// chunks preserve headers and recent/footer information; the middle is
// represented by a searchable marker.
const MAX_CHUNKS_PER_FILE = 24;

// Max file size for ingest (1GB) - applies to text, audio, and image files
const MAX_FILE_SIZE_BYTES = 1024 * 1024 * 1024;
const MAX_TEXT_BYTES_TO_INDEX = 8 * 1024 * 1024;
const PDF_OCR_MIN_TEXT_CHARS = 24;
const PDF_OCR_MAX_PAGES = 5;

// Concurrency settings for parallel processing
const DEFAULT_CONCURRENCY = 10; // Process 10 files in parallel
const MAX_CONCURRENCY = 50; // Maximum parallel operations
const execFileAsync = promisify(execFile);

// Finder files that contain searchable text. Office/PDF formats have
// dedicated extraction below; source/config formats are read as UTF-8.
const INDEXABLE_TEXT_EXTENSIONS = new Set([
  '.txt', '.md', '.markdown', '.csv', '.tsv', '.json', '.jsonl', '.ndjson',
  '.xml', '.yaml', '.yml', '.toml', '.ini', '.conf', '.config', '.env', '.log',
  '.sql', '.html', '.htm', '.css', '.scss', '.less', '.js', '.jsx', '.ts',
  '.tsx', '.mjs', '.cjs', '.py', '.pyw', '.swift', '.m', '.h', '.c', '.cc',
  '.cpp', '.cxx', '.hpp', '.java', '.kt', '.kts', '.go', '.rs', '.rb', '.php',
  '.sh', '.bash', '.zsh', '.fish', '.graphql', '.gql', '.tex', '.rtf',
  '.svelte', '.vue', '.astro', '.cu', '.cuh', '.wgsl', '.glsl', '.metal',
  '.plist', '.strings', '.stringsdata', '.cmake', '.mk', '.make', '.lock',
  '.properties', '.pbxproj', '.entitlements', '.ps1', '.bat', '.vim', '.nix',
  '.jinja', '.jinja2', '.template', '.tmpl', '.example', '.inp', '.dia', '.d',
  '.eml', '.msg', '.vcf', '.ics',
]);

// Binary/container formats still get a useful metadata record. Their bytes
// are never sent through a UTF-8 decoder as if they were document text.
const METADATA_ONLY_EXTENSIONS = new Set([
  '.zip', '.rar', '.7z', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.iso', '.dmg',
  '.pkg', '.app', '.sqlite', '.db', '.bin', '.dat', '.pages', '.numbers', '.key',
  '.psd', '.ai', '.eps', '.sketch', '.blend', '.fig', '.obj', '.stl', '.fbx', '.dwg', '.jar',
]);

@Injectable()
export class IngestService {
  private readonly logger = new Logger(IngestService.name);

  private enrichContent(
    content: string,
    filePath: string,
    docType: string,
  ): string {
    const fileName = path.basename(filePath);
    const ext = path.extname(filePath).toLowerCase().replace('.', '') || 'text';
    const prefix = `[document, file, ${docType}, ${ext} format, ${fileName}] `;
    return prefix + content;
  }

  private chunkText(text: string): string[] {
    const words = text.split(/\s+/).filter((w) => w.length > 0);
    if (words.length <= CHUNK_WORD_COUNT) return [text];

    const chunks: string[] = [];
    let start = 0;
    while (start < words.length) {
      const end = Math.min(start + CHUNK_WORD_COUNT, words.length);
      const chunkWords = words.slice(start, end);
      chunks.push(chunkWords.join(' '));
      // The final window already reaches EOF. Continuing with overlap would
      // leave `start` unchanged and loop forever on any document longer than
      // one chunk.
      if (end >= words.length) break;
      start = end - CHUNK_OVERLAP_WORDS;
    }
    if (chunks.length <= MAX_CHUNKS_PER_FILE) return chunks;
    const headCount = Math.floor((MAX_CHUNKS_PER_FILE - 1) / 2);
    const tailCount = MAX_CHUNKS_PER_FILE - 1 - headCount;
    return [
      ...chunks.slice(0, headCount),
      '[middle content omitted from bounded index]',
      ...chunks.slice(-tailCount),
    ];
  }

  constructor(
    private readonly lanceDBService: LanceDBService,
    private readonly multimodalService: MultimodalService,
    private readonly imageCaptioningService: ImageCaptioningService,
  ) {}

  async ingestDocument(dto: IngestDocumentDto): Promise<IngestResponseDto> {
    const startedAt = Date.now();
    try {
      this.logger.log(`Ingesting document: ${dto.filePath}`);

      // Check file size limit for all media types
      if (fs.existsSync(dto.filePath)) {
        const stats = fs.statSync(dto.filePath);
        if (stats.size > MAX_FILE_SIZE_BYTES) {
          throw new Error(
            `File exceeds size limit of ${MAX_FILE_SIZE_BYTES / 1024 / 1024}MB (file: ${(stats.size / 1024 / 1024).toFixed(1)}MB)`,
          );
        }
      }

      // Determine media type
      const mediaType =
        dto.mediaType || this.multimodalService.getMediaType(dto.filePath);
      const fileName = path.basename(dto.filePath);

      // Keep the backend idempotent for every supported type, not only
      // images. The Swift cache is the fast path, but this guard also covers
      // relaunches, overlapping scans, retries, and callers that talk to the
      // API directly. A force reindex explicitly replaces the old record(s).
      if (!dto.forceReindex && fs.existsSync(dto.filePath)) {
        const existing = await this.lanceDBService.findDocumentByFilePath(dto.filePath);
        if (existing) {
          return {
            id: existing.id,
            fileName,
            filePath: dto.filePath,
            mediaType,
            success: true,
            message: `${mediaType.charAt(0).toUpperCase() + mediaType.slice(1)} already indexed`,
            elapsedMs: Date.now() - startedAt,
          };
        }
      }

      // A Qwen re-index must replace the old fallback caption, never sit next
      // to it as a duplicate vector record.
      if (dto.forceReindex) {
        await this.lanceDBService.deleteDocumentsByFilePath(dto.filePath);
      }

      let id: string;
      let imageTitle: string | undefined;
      let imageDescription: string | undefined;

      if (mediaType === 'text') {
        // Text document - use content if provided, otherwise read from file
        let content = dto.content;

        if (!content && fs.existsSync(dto.filePath)) {
          const ext = path.extname(dto.filePath).toLowerCase();

          if (ext === '.pdf') {
            // Extract normal PDF text, with a bounded OCR fallback for
            // scanned/image-only invoices and receipts.
            content = await this.extractPdfText(dto.filePath);
            this.logger.log(
              `Extracted ${content.length} chars from PDF: ${fileName}`,
            );
          } else if (ext === '.docx') {
            // Extract text from Word (.docx)
            const result = await mammoth.extractRawText({ path: dto.filePath });
            content = result.value;
            this.logger.log(
              `Extracted ${content.length} chars from DOCX: ${fileName}`,
            );
          } else if (ext === '.xlsx' || ext === '.xls') {
            // Extract text from Excel
            const workbook = XLSX.readFile(dto.filePath);
            const sheets: string[] = [];
            for (const sheetName of workbook.SheetNames) {
              const sheet = workbook.Sheets[sheetName];
              const csv = XLSX.utils.sheet_to_csv(sheet);
              sheets.push(`[Sheet: ${sheetName}]\n${csv}`);
            }
            content = sheets.join('\n\n');
            this.logger.log(
              `Extracted ${content.length} chars from Excel: ${fileName}`,
            );
          } else if (ext === '.pptx') {
            // PowerPoint - extract as XML text (basic)
            const dataBuffer = await fs.promises.readFile(dto.filePath);
            content = `[document, file, presentation, slides, pptx format] PowerPoint presentation: ${fileName}. Contains slides and visual content.`;
            this.logger.log(`Indexed PowerPoint: ${fileName}`);
          } else if (ext === '.rtf') {
            content = this.extractRtfText(
              await this.readTextFileSafely(dto.filePath),
            );
            this.logger.log(`Extracted ${content.length} chars from RTF: ${fileName}`);
          } else if (['.doc', '.ppt', '.pages', '.numbers', '.key', '.odt'].includes(ext)) {
            content = await this.extractTextWithTextUtil(dto.filePath).catch(() => '');
            if (!content) content = this.metadataOnlyContent(dto.filePath);
          } else if (METADATA_ONLY_EXTENSIONS.has(ext)) {
            content = this.metadataOnlyContent(dto.filePath);
          } else {
            content = INDEXABLE_TEXT_EXTENSIONS.has(ext)
              ? await this.readTextFileSafely(dto.filePath)
              : this.metadataOnlyContent(dto.filePath);
          }
        }

        if (!content) {
          content = this.metadataOnlyContent(dto.filePath);
        }

        const ext =
          path.extname(dto.filePath).toLowerCase().replace('.', '') || 'text';
        const docType =
          ext === 'pdf'
            ? 'pdf'
            : ext === 'docx'
              ? 'word'
              : ext === 'xlsx' || ext === 'xls'
                ? 'spreadsheet'
                : 'text';

        const shouldChunk = content.length >= MIN_CONTENT_FOR_CHUNKING;
        const chunks = shouldChunk ? this.chunkText(content) : [content];

        const ids: string[] = [];
        for (let i = 0; i < chunks.length; i++) {
          const chunkContent = this.enrichContent(
            chunks[i],
            dto.filePath,
            docType,
          );
          const chunkMetadata = {
            ...dto.metadata,
            mediaType: 'text',
            fileExtension: ext,
            indexingDurationMs: Date.now() - startedAt,
            ...(chunks.length > 1 && {
              chunkIndex: i,
              totalChunks: chunks.length,
            }),
          };
          const chunkId = await this.lanceDBService.addTextDocument(
            chunkContent,
            dto.filePath,
            chunkMetadata,
          );
          ids.push(chunkId);
        }
        id = ids[0];
        if (chunks.length > 1) {
          this.logger.log(
            `Indexed ${fileName} as ${chunks.length} chunks for better search`,
          );
        }
      } else if (mediaType === 'audio') {
        // Audio file - transcribe with Whisper, embed with MiniLM (384-dim)
        // Store in text table since it uses text embeddings
        if (!fs.existsSync(dto.filePath)) {
          throw new Error(`File not found: ${dto.filePath}`);
        }

        const processed = await this.multimodalService.processFile(
          dto.filePath,
        );
        const enrichedContent = this.enrichContent(
          processed.content,
          dto.filePath,
          'audio transcript',
        );

        // Store in text table with pre-computed vector and audio metadata
        id = await this.lanceDBService.addTextDocument(
          enrichedContent,
          dto.filePath,
          {
            ...dto.metadata,
            mediaType: 'audio',
            fileExtension: path.extname(dto.filePath).toLowerCase().replace('.', ''),
            indexingDurationMs: Date.now() - startedAt,
          },
          undefined, // Re-embed with enriched content for better search
        );
      } else if (mediaType === 'image') {
        // Image files - caption with Qwen-VL, then embed the rich caption.
        // Store in text table for unified text search
        if (!fs.existsSync(dto.filePath)) {
          throw new Error(`File not found: ${dto.filePath}`);
        }

        // The Swift cache normally prevents this branch from being reached a
        // second time. Keep the backend idempotent as well so app relaunches,
        // overlapping scans, or a retried HTTP request cannot create a second
        // semantic row for the same unchanged path. A visual upgrade passes
        // forceReindex=true and intentionally replaces the old caption.
        if (!dto.forceReindex) {
          const existing = await this.lanceDBService.findDocumentByFilePath(
            dto.filePath,
          );
          if (existing) {
            return {
              id: existing.id,
              fileName,
              filePath: dto.filePath,
              mediaType: 'image',
              success: true,
              message: 'Image already indexed',
              elapsedMs: Date.now() - startedAt,
            };
          }
        }

        // Generate detailed caption using Panda's local Qwen vision model.
        const imageIndex = await this.imageCaptioningService.generateImageIndexDescription(
          dto.filePath,
        );
        imageTitle = imageIndex.title;
        imageDescription = imageIndex.explanation;
        const ocrText = dto.content?.trim();
        const caption = ocrText
          ? `${imageIndex.searchableText}\n\n[recognized text in image]\n${ocrText}`
          : imageIndex.searchableText;
        this.logger.log(
          `Generated caption for ${fileName}: ${caption.substring(0, 100)}...`,
        );

        // Store caption in text table (will be embedded with MiniLM)
        id = await this.lanceDBService.addTextDocument(caption, dto.filePath, {
          ...dto.metadata,
          mediaType: 'image',
          fileExtension: path.extname(dto.filePath).toLowerCase().replace('.', ''),
          indexingDurationMs: Date.now() - startedAt,
        });
      } else if (mediaType === 'video') {
        if (!fs.existsSync(dto.filePath)) {
          throw new Error(`File not found: ${dto.filePath}`);
        }
        const previewPaths = await this.multimodalService.extractVideoPreviews(dto.filePath);
        try {
          const descriptions: string[] = [];
          for (const [index, previewPath] of previewPaths.entries()) {
            descriptions.push(`[representative video frame ${index + 1} of ${previewPaths.length}] ${await this.imageCaptioningService.generateCaption(previewPath)}`);
          }
          const localVision = dto.content?.trim();
          const caption = `[video, movie, file, ${path.extname(dto.filePath).slice(1)} format, ${fileName}] Qwen visual understanding across the video:\n${descriptions.join('\n\n')}` +
            (localVision ? `\n\n[local video understanding]\n${localVision}` : '');
          id = await this.lanceDBService.addTextDocument(caption, dto.filePath, {
            ...dto.metadata,
            mediaType: 'video',
            fileExtension: path.extname(dto.filePath).toLowerCase().replace('.', ''),
            indexingDurationMs: Date.now() - startedAt,
          });
        } finally {
          await Promise.all(previewPaths.map((previewPath) => fs.promises.unlink(previewPath).catch(() => undefined)));
        }
      } else {
        throw new Error(`Unsupported media type: ${mediaType}`);
      }

      return {
        id,
        fileName,
        filePath: dto.filePath,
        mediaType,
        success: true,
        message: `${mediaType.charAt(0).toUpperCase() + mediaType.slice(1)} "${fileName}" ingested successfully`,
        imageTitle,
        imageDescription,
        elapsedMs: Date.now() - startedAt,
      };
    } catch (error: any) {
      this.logger.error(`Failed to ingest document: ${error.message}`);
      throw error;
    }
  }

  async deleteDocument(
    id: string,
  ): Promise<{ success: boolean; message: string }> {
    try {
      this.logger.log(`Deleting document: ${id}`);
      await this.lanceDBService.deleteDocument(id);

      return {
        success: true,
        message: `Document "${id}" deleted successfully`,
      };
    } catch (error: any) {
      this.logger.error(`Failed to delete document: ${error.message}`);
      throw error;
    }
  }

  async deleteAllDocuments(): Promise<{ success: boolean; message: string }> {
    try {
      this.logger.log('Deleting all documents');
      await this.lanceDBService.deleteAllDocuments();

      return {
        success: true,
        message: 'All documents deleted successfully',
      };
    } catch (error: any) {
      this.logger.error(`Failed to delete all documents: ${error.message}`);
      throw error;
    }
  }

  async listDocuments(): Promise<DocumentListResponseDto> {
    try {
      const documents = await this.lanceDBService.getUniqueDocuments();
      const total = documents.length;

      return {
        documents: documents.map((doc) => ({
          id: doc.id,
          fileName: doc.fileName,
          filePath: doc.filePath,
          mediaType: (doc.mediaType as MediaType) || 'text',
          thumbnailPath: doc.thumbnailPath,
          metadata: doc.metadata,
          chunkIds: doc.chunkIds,
        })),
        total,
      };
    } catch (error: any) {
      this.logger.error(`Failed to list documents: ${error.message}`);
      throw error;
    }
  }

  async deduplicateImageDocuments(): Promise<number> {
    return this.lanceDBService.deduplicateImageDocuments();
  }

  async pruneNonLocalDocuments(): Promise<number> {
    return this.lanceDBService.pruneNonLocalDocuments();
  }

  async getIndexTimingSummary() {
    return this.lanceDBService.getIndexTimingSummary();
  }

  async getDocumentCount(): Promise<{ count: number }> {
    const count = await this.lanceDBService.getDocumentCount();
    return { count };
  }

  /**
   * Batch ingest multiple files (legacy sequential - use batchIngestConcurrent for speed)
   */
  async batchIngest(
    files: IngestDocumentDto[],
  ): Promise<{
    success: number;
    failed: number;
    results: IngestResponseDto[];
  }> {
    // Use concurrent processing by default for better performance
    return this.batchIngestConcurrent(files);
  }

  /**
   * Concurrent batch ingest - processes multiple files in parallel
   * This is significantly faster than sequential processing (up to 100x for large batches)
   */
  async batchIngestConcurrent(
    files: IngestDocumentDto[],
    concurrency: number = DEFAULT_CONCURRENCY,
  ): Promise<{
    success: number;
    failed: number;
    results: IngestResponseDto[];
    elapsedMs: number;
  }> {
    const startTime = Date.now();
    const effectiveConcurrency = Math.min(
      Math.max(1, concurrency),
      MAX_CONCURRENCY,
    );

    this.logger.log(
      `Starting concurrent batch ingest: ${files.length} files with concurrency ${effectiveConcurrency}`,
    );

    // Separate files by type for optimized processing
    const textFiles: IngestDocumentDto[] = [];
    const audioFiles: IngestDocumentDto[] = [];
    const imageFiles: IngestDocumentDto[] = [];
    const videoFiles: IngestDocumentDto[] = [];

    for (const file of files) {
      const mediaType =
        file.mediaType || this.multimodalService.getMediaType(file.filePath);
      if (mediaType === 'audio') {
        audioFiles.push(file);
      } else if (mediaType === 'image') {
        imageFiles.push(file);
      } else if (mediaType === 'video') {
        videoFiles.push(file);
      } else {
        textFiles.push(file);
      }
    }

    // Process all types concurrently
    const [textResults, audioResults, imageResults, videoResults] = await Promise.all([
      this.processTextFilesConcurrent(textFiles, effectiveConcurrency),
      this.processMediaFilesConcurrent(audioFiles, 'audio', Math.max(1, Math.floor(effectiveConcurrency / 2))), // Audio is CPU-intensive
      this.processMediaFilesConcurrent(imageFiles, 'image', Math.max(1, Math.floor(effectiveConcurrency / 2))), // Image captioning is GPU-intensive
      this.processMediaFilesConcurrent(videoFiles, 'video', 1), // Vision processing is memory-intensive
    ]);

    // Combine results
    const results = [...textResults, ...audioResults, ...imageResults, ...videoResults];
    const success = results.filter((r) => r.success).length;
    const failed = results.filter((r) => !r.success).length;
    const elapsedMs = Date.now() - startTime;

    this.logger.log(
      `Concurrent batch ingest complete: ${success} success, ${failed} failed in ${elapsedMs}ms (${(elapsedMs / files.length).toFixed(1)}ms/file avg)`,
    );

    return { success, failed, results, elapsedMs };
  }

  /**
   * Process text files concurrently with batch embedding optimization
   */
  private async processTextFilesConcurrent(
    files: IngestDocumentDto[],
    concurrency: number,
  ): Promise<IngestResponseDto[]> {
    if (files.length === 0) return [];

    this.logger.log(`Processing ${files.length} text files concurrently...`);

    // First, extract content from all files in parallel
    const contentExtractions = await this.runWithConcurrency(
      files,
      async (file) => {
        const startedAt = Date.now();
        const extraction = await this.extractTextContent(file);
        return { ...extraction, elapsedMs: Date.now() - startedAt };
      },
      concurrency,
    );

    // Collect all chunks for batch embedding
    const allChunks: Array<{
      content: string;
      filePath: string;
      metadata: Record<string, unknown>;
      fileIndex: number;
      chunkIndex: number;
    }> = [];

    const fileChunkRanges: Array<{ start: number; end: number }> = [];

    for (let i = 0; i < contentExtractions.length; i++) {
      const extraction = contentExtractions[i];
      if (!extraction.success || !extraction.content) continue;

      const content = extraction.content;
      const shouldChunk = content.length >= MIN_CONTENT_FOR_CHUNKING;
      const chunks = shouldChunk ? this.chunkText(content) : [content];

      const ext = path
        .extname(files[i].filePath)
        .toLowerCase()
        .replace('.', '') || 'text';
      const docType =
        ext === 'pdf'
          ? 'pdf'
          : ext === 'docx'
            ? 'word'
            : ext === 'xlsx' || ext === 'xls'
              ? 'spreadsheet'
              : 'text';

      const startIdx = allChunks.length;
      for (let j = 0; j < chunks.length; j++) {
        allChunks.push({
          content: this.enrichContent(chunks[j], files[i].filePath, docType),
          filePath: files[i].filePath,
          metadata: {
            ...files[i].metadata,
            mediaType: 'text',
            fileExtension: ext,
            // Extraction is measured here; the shared batch embedding/write
            // time is apportioned in the response returned to SwiftData.
            indexingDurationMs: extraction.elapsedMs ?? 0,
            ...(chunks.length > 1 && {
              chunkIndex: j,
              totalChunks: chunks.length,
            }),
          },
          fileIndex: i,
          chunkIndex: j,
        });
      }
      fileChunkRanges[i] = { start: startIdx, end: allChunks.length };
    }

    // Batch insert all chunks at once
    let ids: string[] = [];
    const batchStartedAt = Date.now();
    if (allChunks.length > 0) {
      ids = await this.lanceDBService.addTextDocumentsBatch(
        allChunks.map((c) => ({
          content: c.content,
          filePath: c.filePath,
          metadata: c.metadata,
        })),
      );
    }
    const batchElapsedMs = Date.now() - batchStartedAt;

    // Build results
    const results: IngestResponseDto[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const fileName = path.basename(file.filePath);
      const extraction = contentExtractions[i];

      if (!extraction.success) {
        results.push({
          id: '',
          fileName,
          filePath: file.filePath,
          mediaType: 'text',
          success: false,
          message: extraction.error || 'Failed to extract content',
        });
        continue;
      }

      const range = fileChunkRanges[i];
      if (!range) {
        results.push({
          id: '',
          fileName,
          filePath: file.filePath,
          mediaType: 'text',
          success: false,
          message: 'No content to index',
        });
        continue;
      }

      const fileIds = ids.slice(range.start, range.end);
      const chunkCount = range.end - range.start;

      results.push({
        id: fileIds[0],
        fileName,
        filePath: file.filePath,
        mediaType: 'text',
        success: true,
        message:
          chunkCount > 1
            ? `Indexed as ${chunkCount} chunks`
            : 'Ingested successfully',
        // Extraction is measured per file; embedding/table-write time is
        // shared by this batch and apportioned evenly for useful telemetry.
        elapsedMs: (extraction.elapsedMs ?? 0) +
          Math.round(batchElapsedMs / Math.max(1, files.length)),
      });
    }

    return results;
  }

  /**
   * Extract text content from a file
   */
  private async extractTextContent(
    dto: IngestDocumentDto,
  ): Promise<{ success: boolean; content?: string; error?: string }> {
    try {
      // Check file size
      if (fs.existsSync(dto.filePath)) {
        const stats = fs.statSync(dto.filePath);
        if (stats.size > MAX_FILE_SIZE_BYTES) {
          return {
            success: false,
            error: `File exceeds size limit of ${MAX_FILE_SIZE_BYTES / 1024 / 1024}MB`,
          };
        }
      }

      let content = dto.content;

      if (!content && fs.existsSync(dto.filePath)) {
        const ext = path.extname(dto.filePath).toLowerCase();

        if (ext === '.pdf') {
          content = await this.extractPdfText(dto.filePath);
        } else if (ext === '.docx') {
          const result = await mammoth.extractRawText({ path: dto.filePath });
          content = result.value;
        } else if (ext === '.xlsx' || ext === '.xls') {
          const workbook = XLSX.readFile(dto.filePath);
          const sheets: string[] = [];
          for (const sheetName of workbook.SheetNames) {
            const sheet = workbook.Sheets[sheetName];
            const csv = XLSX.utils.sheet_to_csv(sheet);
            sheets.push(`[Sheet: ${sheetName}]\n${csv}`);
          }
          content = sheets.join('\n\n');
        } else if (ext === '.pptx') {
          const fileName = path.basename(dto.filePath);
          content = `[document, file, presentation, slides, pptx format] PowerPoint presentation: ${fileName}. Contains slides and visual content.`;
        } else if (ext === '.rtf') {
          content = this.extractRtfText(
            await this.readTextFileSafely(dto.filePath),
          );
        } else if (['.doc', '.ppt', '.pages', '.numbers', '.key', '.odt'].includes(ext)) {
          content = await this.extractTextWithTextUtil(dto.filePath).catch(() => '');
          if (!content) content = this.metadataOnlyContent(dto.filePath);
        } else if (METADATA_ONLY_EXTENSIONS.has(ext)) {
          content = this.metadataOnlyContent(dto.filePath);
        } else {
          content = INDEXABLE_TEXT_EXTENSIONS.has(ext)
            ? await this.readTextFileSafely(dto.filePath)
            : this.metadataOnlyContent(dto.filePath);
        }
      }

      if (!content) {
        content = this.metadataOnlyContent(dto.filePath);
      }

      return { success: true, content };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }

  /** Lightweight RTF-to-text extraction for local semantic indexing. */
  private extractRtfText(rtf: string): string {
    return rtf
      .replace(/\\par[d]?/g, '\n')
      .replace(/\\tab/g, '\t')
      .replace(/\\'[0-9a-fA-F]{2}/g, (match) =>
        String.fromCharCode(parseInt(match.slice(2), 16)),
      )
      .replace(/\\[a-zA-Z]+-?\d*\s?/g, '')
      .replace(/[{}]/g, '')
      .replace(/\\\\/g, '\\')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  /** Extract legacy Office/Apple document text using the macOS textutil CLI. */
  private async extractTextWithTextUtil(filePath: string): Promise<string> {
    const result = await execFileAsync('/usr/bin/textutil', [
      '-convert', 'txt', '-stdout', filePath,
    ], { maxBuffer: 20 * 1024 * 1024 });
    return String(result.stdout || '').trim();
  }

  /**
   * Extract searchable PDF text and OCR the first few pages when the PDF has
   * no usable text layer. OCR is intentionally optional: Panda still indexes
   * the filename/path if a minimal install does not include Poppler/Tesseract.
   */
  private async extractPdfText(filePath: string): Promise<string> {
    const parser = new PDFParse({ url: filePath });
    const pdfData = await parser.getText();
    const extracted = String(pdfData.text || '').trim();
    if (extracted.length >= PDF_OCR_MIN_TEXT_CHARS) return extracted;

    const [pdftoppm, tesseract] = await Promise.all([
      this.resolveExecutable('pdftoppm'),
      this.resolveExecutable('tesseract'),
    ]);
    if (!pdftoppm || !tesseract) return extracted;

    const tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'panda-pdf-ocr-'));
    const prefix = path.join(tempDir, 'page');
    try {
      await execFileAsync(pdftoppm, [
        '-f', '1',
        '-l', String(PDF_OCR_MAX_PAGES),
        '-r', '160',
        '-jpeg',
        filePath,
        prefix,
      ], { maxBuffer: 2 * 1024 * 1024 });
      const pageFiles = (await fs.promises.readdir(tempDir))
        .filter((name) => /^page-\d+\.jpg$/i.test(name))
        .sort()
        .slice(0, PDF_OCR_MAX_PAGES);
      const ocrPages: string[] = [];
      for (const pageFile of pageFiles) {
        try {
          const result = await execFileAsync(tesseract, [
            path.join(tempDir, pageFile),
            'stdout',
            '--psm', '6',
          ], { maxBuffer: 4 * 1024 * 1024 });
          const pageText = String(result.stdout || '').trim();
          if (pageText) ocrPages.push(`[OCR page ${ocrPages.length + 1}]\n${pageText}`);
        } catch (error: any) {
          this.logger.warn(`PDF OCR page failed for ${path.basename(filePath)}: ${error.message}`);
        }
      }
      const ocrText = ocrPages.join('\n\n').trim();
      if (!ocrText) return extracted;
      return [extracted, '[OCR text from scanned PDF]', ocrText]
        .filter((part) => part.length > 0)
        .join('\n\n');
    } catch (error: any) {
      this.logger.warn(`PDF OCR unavailable for ${path.basename(filePath)}: ${error.message}`);
      return extracted;
    } finally {
      await fs.promises.rm(tempDir, { recursive: true, force: true }).catch(() => undefined);
    }
  }

  private async resolveExecutable(command: string): Promise<string | null> {
    try {
      const result = await execFileAsync('/usr/bin/which', [command], { maxBuffer: 4096 });
      const resolved = String(result.stdout || '').trim().split('\n')[0];
      return resolved || null;
    } catch {
      return null;
    }
  }

  private metadataOnlyContent(filePath: string): string {
    const fileName = path.basename(filePath);
    const ext = path.extname(filePath).toLowerCase().replace('.', '') || 'unknown';
    return `[file, ${ext} format] File: ${fileName}. Binary/container content is indexed by filename, path, and format.`;
  }

  /** Read normal text fully, but bound huge logs/source files to head+tail. */
  private async readTextFileSafely(filePath: string): Promise<string> {
    const stats = await fs.promises.stat(filePath);
    if (stats.size <= MAX_TEXT_BYTES_TO_INDEX) {
      return fs.promises.readFile(filePath, 'utf-8');
    }
    const handle = await fs.promises.open(filePath, 'r');
    try {
      const segmentSize = Math.floor(MAX_TEXT_BYTES_TO_INDEX / 2);
      const readAt = async (position: number) => {
        const buffer = Buffer.alloc(segmentSize);
        const result = await handle.read(buffer, 0, segmentSize, position);
        return buffer.subarray(0, result.bytesRead).toString('utf8');
      };
      const head = await readAt(0);
      const tail = await readAt(Math.max(0, stats.size - segmentSize));
      return `${head}\n\n[content truncated for memory safety; tail follows]\n\n${tail}`;
    } finally {
      await handle.close();
    }
  }

  /**
   * Process media files (audio/image/video) concurrently
   */
  private async processMediaFilesConcurrent(
    files: IngestDocumentDto[],
    mediaType: 'audio' | 'image' | 'video',
    concurrency: number,
  ): Promise<IngestResponseDto[]> {
    if (files.length === 0) return [];

    this.logger.log(
      `Processing ${files.length} ${mediaType} files concurrently...`,
    );

    return this.runWithConcurrency(
      files,
      async (file) => {
        try {
          const result = await this.ingestDocument({
            ...file,
            mediaType,
          });
          return result;
        } catch (error: any) {
          return {
            id: '',
            fileName: path.basename(file.filePath),
            filePath: file.filePath,
            mediaType,
            success: false,
            message: error.message,
          };
        }
      },
      concurrency,
    );
  }

  /**
   * Run async operations with controlled concurrency (semaphore pattern)
   */
  private async runWithConcurrency<T, R>(
    items: T[],
    fn: (item: T) => Promise<R>,
    concurrency: number,
  ): Promise<R[]> {
    const results: R[] = new Array(items.length);
    let currentIndex = 0;

    const workers = Array(Math.min(concurrency, items.length))
      .fill(null)
      .map(async () => {
        while (currentIndex < items.length) {
          const index = currentIndex++;
          if (index >= items.length) break;
          results[index] = await fn(items[index]);
        }
      });

    await Promise.all(workers);
    return results;
  }
}
