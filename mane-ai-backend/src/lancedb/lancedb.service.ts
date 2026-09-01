import { Injectable, OnModuleInit, Logger } from '@nestjs/common';
import { ConfigService } from '../config';
import * as lancedb from '@lancedb/lancedb';
import * as fs from 'fs';
import * as path from 'path';

// Types for LanceDB
interface TextDocumentRecord {
  id: string;
  content: string;
  filePath: string;
  fileName: string;
  mediaType: string;
  metadata: string; // JSON string
  vector: number[];
  createdAt: string;
  [key: string]: unknown;
}

// Project record for codebase indexing
export interface ProjectRecord {
  id: string;
  name: string;
  path: string;
  description: string;
  techStack: string; // JSON array string
  tags: string; // JSON array string
  manifest: string; // JSON string
  fileCount: number;
  vector: number[];
  createdAt: string;
  [key: string]: unknown;
}

// Code skeleton record for function/class signatures
export interface CodeSkeletonRecord {
  id: string;
  projectId: string;
  filePath: string;
  fileName: string;
  content: string; // Extracted signatures
  language: string;
  vector: number[];
  createdAt: string;
  [key: string]: unknown;
}

export interface SearchResult {
  id: string;
  content: string;
  filePath: string;
  fileName: string;
  mediaType: string;
  thumbnailPath?: string;
  metadata: Record<string, unknown>;
  score: number;
}

export interface ProjectSearchResult {
  id: string;
  name: string;
  path: string;
  description: string;
  techStack: string[];
  tags: string[];
  fileCount: number;
  createdAt: string;
  score: number;
}

export interface SkeletonSearchResult {
  id: string;
  projectId: string;
  filePath: string;
  fileName: string;
  content: string;
  language: string;
  score: number;
}

export type MediaType = 'text' | 'image' | 'audio' | 'video';

@Injectable()
export class LanceDBService implements OnModuleInit {
  private readonly logger = new Logger(LanceDBService.name);
  private db: lancedb.Connection | null = null;
  private textTable: lancedb.Table | null = null;
  private projectsTable: lancedb.Table | null = null;
  private skeletonsTable: lancedb.Table | null = null;
  private embedder: any = null;
  // LanceDB appends are serialized because the native writer can otherwise
  // race when several local VLM requests finish at the same time.
  private textWriteTail: Promise<void> = Promise.resolve();
  // Transformers.js retains temporary tensors while embedding. Serialize
  // calls so concurrent Finder requests cannot exhaust the Node heap.
  private embeddingTail: Promise<void> = Promise.resolve();
  private textPathCache: Map<string, SearchResult> | null = null;
  // Concurrent first requests must share one lightweight path-index load.
  // Without this, every request materializes up to 10k full document rows at
  // once (including large contents), which can exhaust the Node heap during a
  // library scan.
  private textPathCacheLoading: Promise<void> | null = null;

  private readonly textTableName = 'documents_text';
  private readonly projectsTableName = 'projects';
  private readonly skeletonsTableName = 'code_skeletons';

  // Embedding dimensions
  readonly TEXT_DIMENSION = 384; // all-MiniLM-L6-v2

  constructor(private readonly configService: ConfigService) {}

  async onModuleInit() {
    await this.initialize();
  }

  private async initialize(): Promise<void> {
    try {
      // Initialize the embedding model for text (backward compatibility)
      this.logger.log('Loading embedding model (all-MiniLM-L6-v2)...');
      const { pipeline } = await import('@huggingface/transformers');
      this.embedder = await pipeline(
        'feature-extraction',
        'Xenova/all-MiniLM-L6-v2',
      );
      this.logger.log('Embedding model loaded successfully');

      // Ensure the database directory exists
      const dbPath = this.configService.getDbPath();
      if (!fs.existsSync(dbPath)) {
        fs.mkdirSync(dbPath, { recursive: true });
        this.logger.log(`Created database directory: ${dbPath}`);
      }

      // Connect to LanceDB
      this.logger.log(`Connecting to LanceDB at: ${dbPath}`);
      this.db = await lancedb.connect(dbPath);

      // Initialize tables
      await this.initializeTables();

      this.logger.log('LanceDB initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize LanceDB:', error);
      throw error;
    }
  }

  private async initializeTables(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    const tableNames = await this.db.tableNames();

    // Initialize text table
    if (tableNames.includes(this.textTableName)) {
      this.textTable = await this.db.openTable(this.textTableName);
      this.logger.log(`Opened existing table: ${this.textTableName}`);
    } else {
      await this.createTextTable();
    }

    // Initialize projects table
    if (tableNames.includes(this.projectsTableName)) {
      this.projectsTable = await this.db.openTable(this.projectsTableName);
      this.logger.log(`Opened existing table: ${this.projectsTableName}`);
    } else {
      await this.createProjectsTable();
    }

    // Initialize code skeletons table
    if (tableNames.includes(this.skeletonsTableName)) {
      this.skeletonsTable = await this.db.openTable(this.skeletonsTableName);
      this.logger.log(`Opened existing table: ${this.skeletonsTableName}`);
    } else {
      await this.createSkeletonsTable();
    }
  }

  private async createTextTable(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    const sampleRecord: TextDocumentRecord = {
      id: 'init',
      content: '',
      filePath: '',
      fileName: '',
      mediaType: 'text',
      metadata: '{}',
      vector: new Array(this.TEXT_DIMENSION).fill(0),
      createdAt: new Date().toISOString(),
    };

    this.textTable = await this.db.createTable(this.textTableName, [
      sampleRecord,
    ]);
    await this.textTable.delete('id = "init"');
    this.logger.log(`Created new table: ${this.textTableName}`);
  }

  private async createProjectsTable(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    const sampleRecord: ProjectRecord = {
      id: 'init',
      name: '',
      path: '',
      description: '',
      techStack: '[]',
      tags: '[]',
      manifest: '{}',
      fileCount: 0,
      vector: new Array(this.TEXT_DIMENSION).fill(0),
      createdAt: new Date().toISOString(),
    };

    this.projectsTable = await this.db.createTable(this.projectsTableName, [
      sampleRecord,
    ]);
    await this.projectsTable.delete('id = "init"');
    this.logger.log(`Created new table: ${this.projectsTableName}`);
  }

  private async createSkeletonsTable(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    const sampleRecord: CodeSkeletonRecord = {
      id: 'init',
      projectId: '',
      filePath: '',
      fileName: '',
      content: '',
      language: '',
      vector: new Array(this.TEXT_DIMENSION).fill(0),
      createdAt: new Date().toISOString(),
    };

    this.skeletonsTable = await this.db.createTable(this.skeletonsTableName, [
      sampleRecord,
    ]);
    await this.skeletonsTable.delete('id = "init"');
    this.logger.log(`Created new table: ${this.skeletonsTableName}`);
  }

  /**
   * Generate text embedding (for backward compatibility)
   */
  async generateEmbedding(text: string): Promise<number[]> {
    return this.serializeEmbedding(() => this.generateEmbeddingUnsafe(text));
  }

  private async generateEmbeddingUnsafe(text: string): Promise<number[]> {
    if (!this.embedder) {
      throw new Error('Embedding model not initialized');
    }

    const output = await this.embedder(text, {
      pooling: 'mean',
      normalize: true,
    });

    const embedding = Array.from(output.data) as number[];
    // Transformers.js retains ONNX tensor buffers until explicit disposal.
    output.dispose?.();
    return embedding;
  }

  /**
   * Generate embeddings for multiple texts in batch (much faster than sequential)
   * The HuggingFace pipeline supports batch processing natively
   */
  async generateEmbeddingsBatch(texts: string[]): Promise<number[][]> {
    return this.serializeEmbedding(() => this.generateEmbeddingsBatchUnsafe(texts));
  }

  private async generateEmbeddingsBatchUnsafe(texts: string[]): Promise<number[][]> {
    if (!this.embedder) {
      throw new Error('Embedding model not initialized');
    }

    if (texts.length === 0) return [];
    if (texts.length === 1) return [await this.generateEmbeddingUnsafe(texts[0])];

    // Process in optimal batch sizes (balance memory vs speed)
    const BATCH_SIZE = 32;
    const allEmbeddings: number[][] = [];

    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const outputs = await this.embedder(batch, {
        pooling: 'mean',
        normalize: true,
      });

      // Handle batch output - outputs.data contains all embeddings concatenated
      const embeddingSize = this.TEXT_DIMENSION;
      for (let j = 0; j < batch.length; j++) {
        const start = j * embeddingSize;
        const embedding = Array.from(
          outputs.data.slice(start, start + embeddingSize),
        );
        allEmbeddings.push(embedding as number[]);
      }
      outputs.dispose?.();
    }

    return allEmbeddings;
  }

  /**
   * Add a text document (also used for audio transcripts)
   */
  async addTextDocument(
    content: string,
    filePath: string,
    metadata: Record<string, unknown> = {},
    vector?: number[],
  ): Promise<string> {
    if (!this.textTable) {
      throw new Error('Text table not initialized');
    }

    // Validate pre-computed vector dimension if provided
    if (vector && vector.length !== this.TEXT_DIMENSION) {
      throw new Error(
        `Vector dimension mismatch: expected ${this.TEXT_DIMENSION}, got ${vector.length}. ` +
          `Text table requires MiniLM embeddings (384-dim).`,
      );
    }

    const id = this.generateId();
    const fileName = path.basename(filePath);

    // Extract mediaType from metadata (for audio) or default to 'text'
    const mediaType = (metadata.mediaType as string) || 'text';

    // Remove mediaType from metadata to avoid duplication
    const { mediaType: _, ...cleanMetadata } = metadata;

    this.logger.log(`Adding ${mediaType} document: ${fileName}`);
    const docVector = vector || (await this.generateEmbedding(content));

    const record: TextDocumentRecord = {
      id,
      content,
      filePath,
      fileName,
      mediaType,
      metadata: JSON.stringify(cleanMetadata),
      vector: docVector,
      createdAt: new Date().toISOString(),
    };

    await this.serializeTextWrite(() => this.textTable!.add([record]));
    this.textPathCache?.set(filePath, {
      id,
      content,
      filePath,
      fileName,
      mediaType,
      metadata: cleanMetadata,
      score: 1,
    });
    this.logger.log(
      `${mediaType.charAt(0).toUpperCase() + mediaType.slice(1)} document added: ${id} (${fileName})`,
    );

    return id;
  }

  private async serializeEmbedding<T>(operation: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const turn = new Promise<void>((resolve) => { release = resolve; });
    const previous = this.embeddingTail;
    this.embeddingTail = previous.then(() => turn);
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }

  private async serializeTextWrite<T>(operation: () => Promise<T>): Promise<T> {
    let release!: () => void;
    const turn = new Promise<void>((resolve) => { release = resolve; });
    const previous = this.textWriteTail;
    this.textWriteTail = previous.then(() => turn);
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }

  /**
   * Legacy method for backward compatibility
   */
  async addDocument(
    content: string,
    filePath: string,
    metadata: Record<string, unknown> = {},
  ): Promise<string> {
    return this.addTextDocument(content, filePath, metadata);
  }

  /**
   * Add multiple text documents in batch with batch embedding generation
   * This is significantly faster than adding documents one by one
   */
  async addTextDocumentsBatch(
    documents: Array<{
      content: string;
      filePath: string;
      metadata?: Record<string, unknown>;
    }>,
  ): Promise<string[]> {
    if (!this.textTable) {
      throw new Error('Text table not initialized');
    }

    if (documents.length === 0) return [];

    this.logger.log(`Batch adding ${documents.length} documents...`);
    const startTime = Date.now();

    // Extract all content for batch embedding
    const contents = documents.map((d) => d.content);

    // Generate all embeddings in batch (much faster)
    const vectors = await this.generateEmbeddingsBatch(contents);

    // Build all records
    const records: TextDocumentRecord[] = [];
    const ids: string[] = [];

    for (let i = 0; i < documents.length; i++) {
      const doc = documents[i];
      const id = this.generateId();
      const fileName = path.basename(doc.filePath);
      const metadata = doc.metadata || {};
      const mediaType = (metadata.mediaType as string) || 'text';
      const { mediaType: _, ...cleanMetadata } = metadata;

      records.push({
        id,
        content: doc.content,
        filePath: doc.filePath,
        fileName,
        mediaType,
        metadata: JSON.stringify(cleanMetadata),
        vector: vectors[i],
        createdAt: new Date().toISOString(),
      });
      ids.push(id);
    }

    // Insert all records in a single batch
    await this.textTable.add(records);

    const elapsed = Date.now() - startTime;
    this.logger.log(
      `Batch added ${documents.length} documents in ${elapsed}ms (${(elapsed / documents.length).toFixed(1)}ms/doc)`,
    );

    return ids;
  }

  /**
   * Search text documents
   * @param query - The search query
   * @param limit - Maximum number of results. Use 0 to return all results.
   */
  async searchText(query: string, limit: number = 5): Promise<SearchResult[]> {
    if (!this.textTable) {
      throw new Error('Text table not initialized');
    }

    // A scan may be appending a caption while a user searches. Lance's native
    // reader must not observe a table mid-commit; wait for the serialized
    // writer tail before opening a search cursor.
    await this.textWriteTail;

    const queryVector = await this.generateEmbedding(query);

    // If limit is 0, return all results (use a very high number)
    const effectiveLimit = limit <= 0 ? 10000 : limit;

    const results = await this.textTable
      .vectorSearch(queryVector)
      .limit(effectiveLimit)
      .toArray();

    return results.filter((row: any) => this.isSearchableUserFile(row.filePath)).map((row: any) => ({
      id: row.id,
      content: row.content,
      filePath: row.filePath,
      fileName: row.fileName,
      mediaType: row.mediaType || 'text',
      metadata: this.parseMetadata(row.metadata),
      score: row._distance ? 1 - row._distance : 0,
    }));
  }

  private isSearchableUserFile(filePath: string): boolean {
    // Do not return transient screenshots or editor/video-cache artifacts that
    // were indexed by older builds. They are not part of the user's library.
    const ignored = [
      '/private/',
      '/com.openai.sky.CUAService/',
      '/CapCut/',
      '/frameThumbnail/',
      '/Library/Caches/',
      '/Volumes/',
      '/Network/',
      '/Library/CloudStorage/',
      '/Library/Mobile Documents/',
    ];
    const homePath = process.env.HOME || '';
    const homePrefix = homePath && !homePath.endsWith(path.sep) ? homePath + path.sep : homePath;
    return typeof filePath === 'string' &&
      (!homePrefix || filePath.startsWith(homePrefix)) &&
      !ignored.some((fragment) => filePath.includes(fragment)) &&
      fs.existsSync(filePath);
  }

  /**
   * Legacy search method (text only)
   */
  async search(query: string, limit: number = 5): Promise<SearchResult[]> {
    return this.searchText(query, limit);
  }

  /**
   * Hybrid search across text documents
   * Combines vector search (semantic) with keyword matching (exact)
   * @param query - The search query
   * @param limit - Maximum number of results. Use 0 to return all results.
   * @param documentIds - Optional array of document IDs to filter to
   */
  async hybridSearch(
    query: string,
    limit: number = 5,
    documentIds?: string[],
  ): Promise<SearchResult[]> {
    if (!this.textTable) {
      throw new Error('Text table not initialized');
    }

    // Keep search and the one-file-at-a-time writer on a consistent LanceDB
    // snapshot during the automatic visual scan.
    await this.textWriteTail;

    // If documentIds are provided, fetch those specific documents (and all chunks for same file)
    if (documentIds && documentIds.length > 0) {
      this.logger.log(
        `Filtering search to ${documentIds.length} specific document(s)`,
      );
      const expandedIds = new Set<string>();
      const allDocs = await this.textTable.query().limit(10000).toArray();
      const allRows = allDocs as any[];

      for (const docId of documentIds) {
        const doc = allRows.find((r) => r.id === docId);
        if (doc) {
          const filePath = doc.filePath;
          allRows
            .filter((r) => r.filePath === filePath)
            .forEach((r) => expandedIds.add(r.id));
        } else {
          expandedIds.add(docId);
        }
      }

      const filteredResults: SearchResult[] = [];
      for (const row of allRows) {
        if (expandedIds.has(row.id)) {
          filteredResults.push({
            id: row.id,
            content: row.content,
            filePath: row.filePath,
            fileName: row.fileName,
            mediaType: row.mediaType || 'text',
            thumbnailPath: row.thumbnailPath,
            metadata: this.parseMetadata(row.metadata),
            score: 1.0,
          });
        }
      }
      return filteredResults;
    }

    // If limit is 0, we want all results
    const returnAll = limit <= 0;
    // Retrieve a wider local candidate set before applying filename, OCR, and
    // visual-label ranking. Otherwise a genuine visual match can be omitted
    // before the reranker sees it when callers ask for only a few results.
    const searchLimit = returnAll ? 0 : Math.max(limit * 5, 100);

    // Run vector search, then collapse chunks/retries to one result per file.
    // A user asks to find a file, not every embedded chunk or historical
    // caption for that file.
    const rawVectorResults = await this.searchText(query, searchLimit);
    const byFilePath = new Map<string, SearchResult>();
    for (const result of rawVectorResults) {
      const existing = byFilePath.get(result.filePath);
      if (!existing || result.score > existing.score) {
        byFilePath.set(result.filePath, result);
      }
    }
    // Vector similarity is excellent for concepts but can miss a rare vendor
    // name. Add a bounded lexical pass over the
    // projected text columns so exact invoice/seller names are guaranteed to
    // surface, while still collapsing chunks to one file result.
    const keywords = this.queryKeywords(query);
    const exactResults = await this.searchExactKeywords(query, keywords);
    for (const result of exactResults) {
      const existing = byFilePath.get(result.filePath);
      if (!existing || result.score > existing.score) byFilePath.set(result.filePath, result);
    }
    const vectorResults = Array.from(byFilePath.values());

    // Extract keywords for boosting exact matches
    const isImageIntent = keywords.some((keyword) =>
      ['image', 'images', 'photo', 'photos', 'picture', 'pictures'].includes(keyword),
    );

    // Score results with keyword boost
    const scoredResults = vectorResults.map((result) => {
      let keywordScore = 0;
      let hasDirectVisualMatch = false;
      const contentLower = result.content.toLowerCase();
      const fileNameLower = result.fileName.toLowerCase();

      for (const keyword of keywords) {
        // Exact keyword matches in content
        if (contentLower.includes(keyword)) {
          keywordScore += 0.1;
        }
        // Filename matches are more valuable
        if (fileNameLower.includes(keyword)) {
          keywordScore += 0.2;
        }
        // Boost for exact word boundaries
        const wordBoundaryRegex = new RegExp(`\\b${keyword}\\b`, 'i');
        if (wordBoundaryRegex.test(result.content)) {
          keywordScore += 0.15;
        }
      }

      // A visual label is stronger evidence for an image request than a word
      // merely appearing in a screenshot filename or its OCR transcript.
      if (isImageIntent && result.mediaType === 'image') {
        const visualLabels = contentLower.match(/\[local visual labels\]([\s\S]*)/)?.[1] ?? '';
        if (keywords.some((keyword) => visualLabels.includes(keyword))) {
          keywordScore += 0.65;
          hasDirectVisualMatch = true;
        }
        const isGenericScreenshot = /screenshot/.test(fileNameLower) &&
          /\[local visual labels\][\s\S]*(document|screenshot)/.test(contentLower);
        if (isGenericScreenshot && !keywords.includes('screenshot')) {
          keywordScore -= 0.55;
        }
      }

      return {
        ...result,
        score: Math.min(result.score + keywordScore, 1.0), // Cap at 1.0
        hasDirectVisualMatch,
      };
    });

    const sortedResults = scoredResults.sort((a, b) => {
      if (a.hasDirectVisualMatch !== b.hasDirectVisualMatch) {
        return a.hasDirectVisualMatch ? -1 : 1;
      }
      return b.score - a.score;
    });

    // Return all results if limit is 0, otherwise slice to limit
    return returnAll ? sortedResults : sortedResults.slice(0, limit);
  }

  /** Return normalized query tokens once, keeping punctuation out of exact matching. */
  private queryKeywords(query: string): string[] {
    return query.toLowerCase().split(/\s+/).map((token) => token.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, '')).filter((token) => token.length > 2);
  }

  private async searchExactKeywords(query: string, keywords: string[]): Promise<SearchResult[]> {
    if (!this.textTable || keywords.length === 0) return [];
    try {
      const rows = (await this.textTable.query()
        .select(['id', 'content', 'filePath', 'fileName', 'mediaType', 'metadata'])
        .limit(100000)
        .toArray()) as any[];
      const phrase = query.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
      const byPath = new Map<string, SearchResult>();
      for (const row of rows) {
        if (!this.isSearchableUserFile(row.filePath)) continue;
        const content = String(row.content || '');
        const haystack = `${content} ${String(row.fileName || '')}`.toLowerCase();
        const matched = keywords.filter((keyword) => haystack.includes(keyword));
        if (matched.length === 0) continue;
        const phraseMatch = phrase.length > 4 && haystack.includes(phrase);
        const score = Math.min(1, 0.52 + (matched.length / keywords.length) * 0.38 + (phraseMatch ? 0.1 : 0));
        const candidate: SearchResult = {
          id: row.id,
          content,
          filePath: row.filePath,
          fileName: row.fileName,
          mediaType: row.mediaType || 'text',
          thumbnailPath: row.thumbnailPath,
          metadata: this.parseMetadata(row.metadata),
          score,
        };
        const existing = byPath.get(candidate.filePath);
        if (!existing || candidate.score > existing.score) byPath.set(candidate.filePath, candidate);
      }
      return Array.from(byPath.values()).sort((a, b) => b.score - a.score);
    } catch (e) {
      this.logger.warn(`Exact keyword search unavailable: ${e}`);
      return [];
    }
  }

  /**
   * Delete a document by ID. If the document is part of a chunked file,
   * deletes all chunks for that file.
   */
  async deleteDocument(id: string): Promise<void> {
    if (!this.textTable) return;
    try {
      const docs = await this.textTable
        .query()
        .where(`id = "${id}"`)
        .limit(1)
        .toArray();
      if (docs.length === 0) return;
      const filePath = (docs[0] as any).filePath;
      await this.deleteDocumentsByFilePath(filePath);
      this.logger.log(`Document(s) deleted for: ${filePath}`);
    } catch (e) {
      this.logger.warn(`Failed to delete document ${id}: ${e}`);
    }
  }

  /**
   * Delete all documents from the text table
   */
  async deleteAllDocuments(): Promise<void> {
    if (!this.textTable) return;
    try {
      await this.textTable.delete('1 = 1');
      this.logger.log('All documents deleted');
    } catch (e) {
      this.logger.warn(`Failed to delete all documents: ${e}`);
      throw e;
    }
  }

  /**
   * Delete all documents (including chunks) for a given file path
   */
  async deleteDocumentsByFilePath(filePath: string): Promise<void> {
    if (!this.textTable) return;
    try {
      // A chunked library can easily exceed ten thousand Lance rows. The
      // previous limit could miss the target path, making a delete appear to
      // succeed while leaving the old semantic record searchable.
      const allDocs = await this.textTable.query()
        .select(['id', 'filePath'])
        .limit(100000)
        .toArray();
      const toDelete = (allDocs as any[]).filter(
        (row) => row.filePath === filePath,
      );
      // Delete in one bounded predicate. Lance rewrites fragments for every
      // delete call; issuing one call per chunk can race a later ingest and
      // leave stale rows behind.
      for (let index = 0; index < toDelete.length; index += 100) {
        const batch = toDelete.slice(index, index + 100);
        const predicate = batch
          .map((row) => `id = "${String(row.id).replace(/"/g, '\\"')}"`)
          .join(' OR ');
        if (predicate) await this.textTable.delete(predicate);
      }
      this.textPathCache?.delete(filePath);
    } catch (e) {
      this.logger.warn(`Failed to delete documents for ${filePath}: ${e}`);
    }
  }

  /** Return the existing semantic record for a path for backend idempotency. */
  async findDocumentByFilePath(filePath: string): Promise<SearchResult | null> {
    if (!this.textTable) return null;
    try {
      await this.textWriteTail;
      // Use a small in-memory path index after the first read. Querying all
      // rows avoids DataFusion's case-normalization issue with the camel-case
      // filePath column and keeps relaunch checks cheap during a scan.
      if (!this.textPathCache) {
        if (!this.textPathCacheLoading) {
          this.textPathCacheLoading = (async () => {
            // Idempotency only needs identity/metadata. Selecting these
            // columns avoids copying potentially megabytes of document text.
            const rows = (await this.textTable!.query()
              .select(['id', 'filePath', 'fileName', 'mediaType', 'metadata', 'createdAt'])
              .limit(100000)
              .toArray()) as any[];
            this.textPathCache = new Map(
              rows.map((row) => [row.filePath, {
                id: row.id,
                content: '',
                filePath: row.filePath,
                fileName: row.fileName,
                mediaType: row.mediaType || 'text',
                thumbnailPath: row.thumbnailPath,
                metadata: this.parseMetadata(row.metadata),
                score: 1,
              } as SearchResult]),
            );
          })();
        }
        try {
          await this.textPathCacheLoading;
        } finally {
          this.textPathCacheLoading = null;
        }
      }
      return this.textPathCache.get(filePath) || null;
    } catch (e) {
      this.logger.warn(`Failed to find document for ${filePath}: ${e}`);
      return null;
    }
  }

  /** Remove duplicate image records while retaining the newest caption. */
  async deduplicateImageDocuments(): Promise<number> {
    if (!this.textTable) return 0;
    try {
      await this.textWriteTail;
      const rows = (await this.textTable.query().limit(10000).toArray()) as any[];
      const idsToRemove: string[] = [];
      // Keep the semantic index aligned with Panda's Finder scan. These
      // directories contain generated caches, build artifacts, or bundled
      // developer assets rather than user-owned images. The original files
      // are never touched; only their stale index rows are removed.
      const excludedDirectoryNames = new Set([
        'Library', 'node_modules', '.git', '.cache', '.npm', '.pnpm-store',
        '.gemini', '.codex', '.antigravity', 'DerivedData', 'Pods', 'Carthage', 'Cache', 'Caches',
        'frameThumbnail', 'Proxy', 'Temp', '.Trash', 'CapCut', '.next', '.turbo',
        'Photos Library.photoslibrary', 'Photo Booth Library.photobooth',
        'dist', 'build', 'out', 'target', '.cargo', '.rustup', '.lmstudio',
        '.venv', '.venv_paddlevl', 'site-packages', '.Trashes', '.Spotlight-V100',
        '.fseventsd', '.DocumentRevisions-V100', '.TemporaryItems',
      ]);
      const byPath = new Map<string, any[]>();
      const homePath = process.env.HOME || '';
      const homePrefix = homePath && !homePath.endsWith(path.sep) ? homePath + path.sep : homePath;
      for (const row of rows) {
        if (row.mediaType !== 'image' || typeof row.filePath !== 'string') continue;
        const isExcluded = row.filePath.split(path.sep).some((component: string) =>
          component.startsWith('.') || excludedDirectoryNames.has(component),
        );
        const outsideLocalUserSpace = Boolean(homePrefix) && !row.filePath.startsWith(homePrefix);
        if (isExcluded || outsideLocalUserSpace || !fs.existsSync(row.filePath)) {
          idsToRemove.push(String(row.id));
          continue;
        }
        const list = byPath.get(row.filePath) || [];
        list.push(row);
        byPath.set(row.filePath, list);
      }

      for (const pathRows of byPath.values()) {
        if (pathRows.length < 2) continue;
        // Prefer the newest non-fallback caption, then newest record overall.
        pathRows.sort((a, b) => {
          const aFallback = String(a.content || '').includes('Image file:');
          const bFallback = String(b.content || '').includes('Image file:');
          if (aFallback !== bFallback) return aFallback ? 1 : -1;
          return String(b.createdAt || '').localeCompare(String(a.createdAt || ''));
        });
        for (const duplicate of pathRows.slice(1)) {
          idsToRemove.push(String(duplicate.id));
        }
      }

      // Lance rewrites fragments for deletes. Deleting one ID per call makes
      // a first cleanup of a large historical index take many minutes, so
      // consolidate the derived-index mutations into bounded predicates.
      const batchSize = 100;
      for (let index = 0; index < idsToRemove.length; index += batchSize) {
        const batch = idsToRemove.slice(index, index + batchSize);
        const predicate = batch
          .map((id) => `id = "${id.replace(/"/g, '\\"')}"`)
          .join(' OR ');
        await this.textTable.delete(predicate);
      }
      const removed = idsToRemove.length;
      if (removed > 0) this.logger.log(`Removed ${removed} stale or duplicate image records`);
      this.textPathCache = null;
      return removed;
    } catch (e) {
      this.logger.warn(`Failed to deduplicate image documents: ${e}`);
      return 0;
    }
  }

  /** Remove all derived rows that live outside the current Mac user's home. */
  async pruneNonLocalDocuments(): Promise<number> {
    if (!this.textTable) return 0;
    try {
      await this.textWriteTail;
      const rows = (await this.textTable.query()
        .select(['id', 'filePath'])
        .limit(100000)
        .toArray()) as any[];
      const homePath = process.env.HOME || '';
      const homePrefix = homePath && !homePath.endsWith(path.sep) ? homePath + path.sep : homePath;
      const excludedDirectoryNames = new Set([
        'Library', 'Applications', 'node_modules', '.git', '.cache', '.npm', '.pnpm-store',
        'DerivedData', 'Caches', 'Cache', 'CapCut', 'frameThumbnail', 'Photos Library.photoslibrary',
        'Photo Booth Library.photobooth', '.Trash', '.Trashes', '.Spotlight-V100', '.fseventsd',
      ]);
      const idsToRemove = rows
        .filter((row) => {
          if (typeof row.filePath !== 'string' || (homePrefix && !row.filePath.startsWith(homePrefix))) return true;
          return row.filePath.split(path.sep).some((component: string) => excludedDirectoryNames.has(component));
        })
        .map((row) => String(row.id));
      for (let index = 0; index < idsToRemove.length; index += 100) {
        const batch = idsToRemove.slice(index, index + 100);
        const predicate = batch.map((id) => `id = "${id.replace(/"/g, '\\"')}"`).join(' OR ');
        await this.textTable.delete(predicate);
      }
      if (idsToRemove.length > 0) this.textPathCache = null;
      if (idsToRemove.length > 0) this.logger.log(`Removed ${idsToRemove.length} non-local document records`);
      return idsToRemove.length;
    } catch (e) {
      this.logger.warn(`Failed to prune non-local documents: ${e}`);
      return 0;
    }
  }

  /** Aggregate persisted per-file indexing durations by media type/extension. */
  async getIndexTimingSummary(): Promise<Array<{
    mediaType: string;
    fileExtension: string;
    count: number;
    totalMs: number;
    averageMs: number;
  }>> {
    if (!this.textTable) return [];
    const rows = (await this.textTable.query()
      .select(['filePath', 'mediaType', 'metadata', 'createdAt'])
      .limit(100000)
      .toArray()) as any[];
    // A long document may have several embedded chunks, and older retries may
    // have left more than one row for a path. Timing is a per-file metric, so
    // collapse those rows before aggregating by type/extension.
    const perFile = new Map<string, { mediaType: string; fileExtension: string; duration: number; createdAt: string }>();
    const grouped = new Map<string, { mediaType: string; fileExtension: string; count: number; totalMs: number }>();
    for (const row of rows) {
      const metadata = this.parseMetadata(row.metadata);
      const duration = Number(metadata.indexingDurationMs);
      if (!Number.isFinite(duration) || duration < 0) continue;
      const mediaType = String(row.mediaType || metadata.mediaType || 'text');
      const fileExtension = String(metadata.fileExtension || path.extname(String(row.filePath || '')).replace('.', '').toLowerCase() || 'unknown');
      const filePath = String(row.filePath || '');
      if (!filePath || !this.isSearchableUserFile(filePath)) continue;
      // All chunks produced in one pass carry a progressively measured
      // duration. If retries exist, use the newest persisted pass rather than
      // counting the same file several times.
      const previous = perFile.get(filePath);
      const createdAt = String(row.createdAt || '');
      if (!previous || createdAt >= previous.createdAt) {
        perFile.set(filePath, { mediaType, fileExtension, duration, createdAt });
      }
    }
    for (const { mediaType, fileExtension, duration } of perFile.values()) {
      const key = `${mediaType}:${fileExtension}`;
      const current = grouped.get(key) || { mediaType, fileExtension, count: 0, totalMs: 0 };
      current.count += 1;
      current.totalMs += duration;
      grouped.set(key, current);
    }
    return Array.from(grouped.values())
      .map((item) => ({ ...item, averageMs: Math.round(item.totalMs / item.count) }))
      .sort((a, b) => b.totalMs - a.totalMs);
  }

  /**
   * Get total document count
   */
  async getDocumentCount(): Promise<number> {
    if (this.textTable) {
      return await this.textTable.countRows();
    }
    return 0;
  }

  /**
   * Get all documents
   */
  async getAllDocuments(): Promise<SearchResult[]> {
    if (!this.textTable) {
      return [];
    }

    const textDocs = await this.textTable.query().limit(1000).toArray();
    return textDocs.map((row: any) => ({
      id: row.id,
      content: row.content,
      filePath: row.filePath,
      fileName: row.fileName,
      mediaType: row.mediaType || 'text',
      metadata: this.parseMetadata(row.metadata),
      score: 1,
    }));
  }

  /**
   * Get unique documents grouped by file path. For chunked documents,
   * returns one entry per file with all chunk IDs for search filtering.
   */
  async getUniqueDocuments(): Promise<
    Array<SearchResult & { chunkIds?: string[] }>
  > {
    const all = await this.getAllDocuments();
    const byPath = new Map<string, SearchResult[]>();
    for (const doc of all) {
      const list = byPath.get(doc.filePath) || [];
      list.push(doc);
      byPath.set(doc.filePath, list);
    }
    return Array.from(byPath.values()).map((docs) => {
      const first = docs[0];
      const chunkIds = docs.map((d) => d.id);
      return {
        ...first,
        id: first.id,
        chunkIds: docs.length > 1 ? chunkIds : undefined,
      };
    });
  }

  // ==================== PROJECT METHODS ====================

  /**
   * Add a new project
   */
  async addProject(
    name: string,
    projectPath: string,
    description: string,
    techStack: string[],
    tags: string[],
    manifest: Record<string, unknown>,
    fileCount: number,
  ): Promise<string> {
    if (!this.projectsTable) {
      throw new Error('Projects table not initialized');
    }

    const id = this.generateProjectId();

    // Generate embedding from description + tech stack + tags
    const embeddingText = `${name} ${description} ${techStack.join(' ')} ${tags.join(' ')}`;
    const vector = await this.generateEmbedding(embeddingText);

    const record: ProjectRecord = {
      id,
      name,
      path: projectPath,
      description,
      techStack: JSON.stringify(techStack),
      tags: JSON.stringify(tags),
      manifest: JSON.stringify(manifest),
      fileCount,
      vector,
      createdAt: new Date().toISOString(),
    };

    await this.projectsTable.add([record]);
    this.logger.log(`Project added: ${id} (${name})`);

    return id;
  }

  /**
   * Get all projects
   */
  async getAllProjects(): Promise<ProjectSearchResult[]> {
    if (!this.projectsTable) {
      return [];
    }

    const projects = await this.projectsTable.query().limit(1000).toArray();
    return projects.map((row: any) => ({
      id: row.id,
      name: row.name,
      path: row.path,
      description: row.description,
      techStack: this.parseJsonArray(row.techStack),
      tags: this.parseJsonArray(row.tags),
      fileCount: row.fileCount,
      createdAt: row.createdAt,
      score: 1,
    }));
  }

  /**
   * Get project by ID
   */
  async getProject(id: string): Promise<ProjectSearchResult | null> {
    if (!this.projectsTable) {
      return null;
    }

    const results = await this.projectsTable
      .query()
      .where(`id = "${id}"`)
      .limit(1)
      .toArray();

    if (results.length === 0) {
      return null;
    }

    const row = results[0] as any;
    return {
      id: row.id,
      name: row.name,
      path: row.path,
      description: row.description,
      techStack: this.parseJsonArray(row.techStack),
      tags: this.parseJsonArray(row.tags),
      fileCount: row.fileCount,
      createdAt: row.createdAt,
      score: 1,
    };
  }

  /**
   * Get project by path
   */
  async getProjectByPath(
    projectPath: string,
  ): Promise<ProjectSearchResult | null> {
    if (!this.projectsTable) {
      return null;
    }

    const results = await this.projectsTable
      .query()
      .where(`path = "${projectPath}"`)
      .limit(1)
      .toArray();

    if (results.length === 0) {
      return null;
    }

    const row = results[0] as any;
    return {
      id: row.id,
      name: row.name,
      path: row.path,
      description: row.description,
      techStack: this.parseJsonArray(row.techStack),
      tags: this.parseJsonArray(row.tags),
      fileCount: row.fileCount,
      createdAt: row.createdAt,
      score: 1,
    };
  }

  /**
   * Search projects by query (hybrid search)
   */
  async searchProjects(
    query: string,
    limit: number = 10,
  ): Promise<ProjectSearchResult[]> {
    if (!this.projectsTable) {
      return [];
    }

    const queryVector = await this.generateEmbedding(query);
    const results = await this.projectsTable
      .vectorSearch(queryVector)
      .limit(limit * 2)
      .toArray();

    // Extract keywords for boosting
    const keywords = query
      .toLowerCase()
      .split(/\s+/)
      .filter((k) => k.length > 2);

    const scoredResults = results.map((row: any) => {
      let keywordScore = 0;
      const nameLower = row.name.toLowerCase();
      const descLower = row.description.toLowerCase();
      const techStack = this.parseJsonArray(row.techStack);
      const tags = this.parseJsonArray(row.tags);

      for (const keyword of keywords) {
        // Name matches are most valuable
        if (nameLower.includes(keyword)) keywordScore += 0.3;
        // Tech stack exact matches
        if (techStack.some((t: string) => t.toLowerCase() === keyword))
          keywordScore += 0.4;
        // Tag matches
        if (tags.some((t: string) => t.toLowerCase().includes(keyword)))
          keywordScore += 0.3;
        // Description matches
        if (descLower.includes(keyword)) keywordScore += 0.1;
      }

      const baseScore = row._distance ? 1 - row._distance : 0;
      return {
        id: row.id,
        name: row.name,
        path: row.path,
        description: row.description,
        techStack,
        tags,
        fileCount: row.fileCount,
        createdAt: row.createdAt,
        score: Math.min(baseScore + keywordScore, 1.0),
      };
    });

    return scoredResults.sort((a, b) => b.score - a.score).slice(0, limit);
  }

  /**
   * Delete a project and its skeletons
   */
  async deleteProject(id: string): Promise<void> {
    if (this.projectsTable) {
      try {
        await this.projectsTable.delete(`id = "${id}"`);
        this.logger.log(`Project deleted: ${id}`);
      } catch (e) {
        this.logger.warn(`Failed to delete project ${id}: ${e}`);
      }
    }

    // Also delete associated skeletons
    await this.deleteSkeletonsByProject(id);
  }

  // ==================== CODE SKELETON METHODS ====================

  /**
   * Add code skeletons for a file
   */
  async addCodeSkeleton(
    projectId: string,
    filePath: string,
    content: string,
    language: string,
  ): Promise<string> {
    if (!this.skeletonsTable) {
      throw new Error('Skeletons table not initialized');
    }

    const id = this.generateSkeletonId();
    const fileName = path.basename(filePath);
    const vector = await this.generateEmbedding(content);

    const record: CodeSkeletonRecord = {
      id,
      projectId,
      filePath,
      fileName,
      content,
      language,
      vector,
      createdAt: new Date().toISOString(),
    };

    await this.skeletonsTable.add([record]);
    this.logger.log(`Code skeleton added: ${id} (${fileName})`);

    return id;
  }

  /**
   * Add multiple code skeletons in batch with batch embedding generation
   * Optimized for high throughput - generates all embeddings in parallel
   */
  async addCodeSkeletonsBatch(
    skeletons: Array<{
      projectId: string;
      filePath: string;
      content: string;
      language: string;
    }>,
  ): Promise<string[]> {
    if (!this.skeletonsTable || skeletons.length === 0) {
      return [];
    }

    this.logger.log(`Batch adding ${skeletons.length} code skeletons...`);
    const startTime = Date.now();

    // Extract all content for batch embedding
    const contents = skeletons.map((s) => s.content);

    // Generate all embeddings in batch (much faster than sequential)
    const vectors = await this.generateEmbeddingsBatch(contents);

    // Build all records
    const records: CodeSkeletonRecord[] = [];
    const ids: string[] = [];

    for (let i = 0; i < skeletons.length; i++) {
      const skeleton = skeletons[i];
      const id = this.generateSkeletonId();
      const fileName = path.basename(skeleton.filePath);

      records.push({
        id,
        projectId: skeleton.projectId,
        filePath: skeleton.filePath,
        fileName,
        content: skeleton.content,
        language: skeleton.language,
        vector: vectors[i],
        createdAt: new Date().toISOString(),
      });
      ids.push(id);
    }

    // Insert all records in a single batch
    await this.skeletonsTable.add(records);

    const elapsed = Date.now() - startTime;
    this.logger.log(
      `Batch added ${skeletons.length} code skeletons in ${elapsed}ms (${(elapsed / skeletons.length).toFixed(1)}ms/skeleton)`,
    );

    return ids;
  }

  /**
   * Search code skeletons
   */
  async searchSkeletons(
    query: string,
    limit: number = 10,
    projectId?: string,
  ): Promise<SkeletonSearchResult[]> {
    if (!this.skeletonsTable) {
      return [];
    }

    const queryVector = await this.generateEmbedding(query);
    let searchQuery = this.skeletonsTable.vectorSearch(queryVector);

    // Note: LanceDB doesn't support WHERE with vectorSearch directly
    // We'll filter after the search
    const results = await searchQuery.limit(limit * 3).toArray();

    let filteredResults = results;
    if (projectId) {
      filteredResults = results.filter(
        (row: any) => row.projectId === projectId,
      );
    }

    return filteredResults.slice(0, limit).map((row: any) => ({
      id: row.id,
      projectId: row.projectId,
      filePath: row.filePath,
      fileName: row.fileName,
      content: row.content,
      language: row.language,
      score: row._distance ? 1 - row._distance : 0,
    }));
  }

  /**
   * Get skeletons for a project
   */
  async getSkeletonsByProject(
    projectId: string,
  ): Promise<SkeletonSearchResult[]> {
    if (!this.skeletonsTable) {
      return [];
    }

    const results = await this.skeletonsTable
      .query()
      .where(`projectId = "${projectId}"`)
      .limit(1000)
      .toArray();

    return results.map((row: any) => ({
      id: row.id,
      projectId: row.projectId,
      filePath: row.filePath,
      fileName: row.fileName,
      content: row.content,
      language: row.language,
      score: 1,
    }));
  }

  /**
   * Delete skeletons for a project
   */
  async deleteSkeletonsByProject(projectId: string): Promise<void> {
    if (this.skeletonsTable) {
      try {
        await this.skeletonsTable.delete(`projectId = "${projectId}"`);
        this.logger.log(`Deleted skeletons for project: ${projectId}`);
      } catch (e) {
        this.logger.warn(
          `Failed to delete skeletons for project ${projectId}: ${e}`,
        );
      }
    }
  }

  /**
   * Get project count
   */
  async getProjectCount(): Promise<number> {
    if (this.projectsTable) {
      return await this.projectsTable.countRows();
    }
    return 0;
  }

  // ==================== HELPER METHODS ====================

  private parseMetadata(
    metadata: string | Record<string, unknown>,
  ): Record<string, unknown> {
    if (typeof metadata === 'string') {
      try {
        return JSON.parse(metadata);
      } catch {
        return {};
      }
    }
    return metadata || {};
  }

  private parseJsonArray(jsonStr: string | string[]): string[] {
    if (Array.isArray(jsonStr)) {
      return jsonStr;
    }
    if (typeof jsonStr === 'string') {
      try {
        return JSON.parse(jsonStr);
      } catch {
        return [];
      }
    }
    return [];
  }

  private generateId(): string {
    return `doc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateProjectId(): string {
    return `proj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateSkeletonId(): string {
    return `skel_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}
