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
  /**
   * Search-only evidence assembled from every chunk of a file. Keeping this
   * separate from `content` lets ranking see the whole document while the UI
   * still receives a short, useful result preview.
   */
  searchEvidence?: string;
  /** Weighted coverage of explicit query phrases (multi-word concepts count
   * more than a single generic token). Used only for deterministic ranking. */
  phraseCoverage?: number;
  phraseMatchCount?: number;
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

  // These directories contain generated/build/dependency data rather than
  // user files. Older Panda versions indexed some of them before the scanner
  // learned to skip them, so the same list is used by search and cleanup.
  private readonly excludedPathComponents = new Set([
    'Library', 'Applications', 'node_modules', '.git', '.cache', '.npm',
    '.pnpm-store', '.gemini', '.codex', '.antigravity', 'DerivedData',
    '.build', '.swiftpm', 'Pods', 'Carthage', 'Cache', 'Caches', 'dist',
    'build', 'out', 'target', 'coverage', 'work', 'frameThumbnail',
    'Proxy', 'Temp', '.Trash', '.Trashes', 'CapCut', '.next', '.turbo',
    '.venv', '.venv_paddlevl', 'site-packages', '.lmstudio',
    'Photos Library.photoslibrary', 'Photo Booth Library.photobooth',
    '.Spotlight-V100', '.fseventsd', '.DocumentRevisions-V100',
    '.TemporaryItems', 'CloudStorage', 'Mobile Documents',
  ]);

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

      // Remove rows created by older scans before they can pollute the first
      // search after an upgrade (for example Xcode DerivedData under work/).
      await this.pruneNonLocalDocuments();

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
    // The vector search is filtered after reading because Lance's native
    // vector query cannot express our path-component exclusions. Read a wide
    // candidate window so generated rows do not crowd out real user files.
    const effectiveLimit = limit <= 0 ? 100000 : Math.min(100000, Math.max(limit * 20, 1000));

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
      // Lance's default vector metric is L2 distance. Treating that value as
      // `1 - distance` made a useful visual neighbour look negative and caused
      // the relevance gate to discard it. Convert the normalized embedding
      // distance back to cosine-like similarity.
      score: this.vectorSimilarity(row._distance),
    }));
  }

  /** Convert Lance's normalized L2 distance to a stable [0, 1] score. */
  private vectorSimilarity(distance: unknown): number {
    const value = Number(distance);
    if (!Number.isFinite(value)) return 0;
    const boundedDistance = Math.max(0, Math.min(2, value));
    return Math.max(0, Math.min(1, 1 - (boundedDistance * boundedDistance) / 2));
  }

  private isSearchableUserFile(filePath: string): boolean {
    // Do not return transient screenshots or editor/video-cache artifacts that
    // were indexed by older builds. They are not part of the user's library.
    const homePath = process.env.HOME || '';
    const homePrefix = homePath && !homePath.endsWith(path.sep) ? homePath + path.sep : homePath;
    if (typeof filePath !== 'string' ||
      !homePrefix ||
      !filePath.startsWith(homePrefix) ||
      !fs.existsSync(filePath)) return false;
    return !filePath.split(path.sep).some((component) => this.excludedPathComponents.has(component));
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
    const semanticScoreByPath = new Map<string, number>();
    for (const result of rawVectorResults) {
      const existing = byFilePath.get(result.filePath);
      if (!existing || result.score > existing.score) {
        byFilePath.set(result.filePath, result);
        semanticScoreByPath.set(result.filePath, result.score);
      }
    }
    // Vector similarity is excellent for concepts but can miss a rare vendor
    // name. Add a bounded lexical pass over the
    // projected text columns so exact invoice/seller names are guaranteed to
    // surface, while still collapsing chunks to one file result.
    const keywords = this.queryKeywords(query);
    const phrases = this.queryPhrases(query);
    const exactResults = await this.searchExactKeywords(query, keywords, phrases);
    for (const result of exactResults) {
      const existing = byFilePath.get(result.filePath);
      if (!existing || result.score > existing.score) {
        // Preserve vector evidence as well as the full-file lexical evidence.
        result.searchEvidence = [existing?.searchEvidence, result.searchEvidence]
          .filter(Boolean)
          .join('\n');
        byFilePath.set(result.filePath, result);
      } else if (result.searchEvidence) {
        existing.searchEvidence = [existing.searchEvidence, result.searchEvidence]
          .filter(Boolean)
          .join('\n');
      }
    }
    const vectorResults = Array.from(byFilePath.values());

    // Extract keywords for boosting exact matches
    // Respect explicit media-type requests before semantic ranking. A text
    // file can mention “panda” or “green” and still be a poor answer to
    // “show images…”; returning it makes the result set look noisy even when
    // its embedding is close. Keep the constraint narrow so an unconstrained
    // concept query (for example, “files about a green panda”) can still
    // return any useful file type.
    // Media words are intent, not content. Detect them from the original
    // query because queryKeywords deliberately removes generic media terms so
    // captions such as "[image, picture, file, png format]" do not make every
    // image look like an exact match.
    const explicitImageIntent = /\b(?:image|images|photo|photos|picture|pictures|screenshot|screenshots)\b/i.test(query);
    const isAudioIntent = /\b(?:audio|recording|recordings|podcast|podcasts|voice|voices|sound)\b/i.test(query);
    const isDocumentIntent = !explicitImageIntent && !isAudioIntent &&
      /\b(?:doc|document|documents|pdf|invoice|invoices|receipt|receipts|report|reports|contract|contracts|word)\b/i.test(query);
    // People naturally omit the word "image" ("find the guy in a suit",
    // "show the panda with green eyes"). VLM captions are the source of truth
    // for these visual nouns, so route those queries through the image-aware
    // relevance path as long as the user did not explicitly request a text
    // document.
    const visualConceptQuery = /\b(?:guy|man|woman|person|people|face|faces|portrait|wearing|suit|jacket|shirt|animal|dog|cat|panda|building|landscape|mountain|car|vehicle|flower|food|logo|icon|eye|eyes)\b/i.test(query);
    const isImageIntent = explicitImageIntent || (!isDocumentIntent && visualConceptQuery);
    // A comma-separated phrase list is a precise request. Requiring two
    // concepts prevents a generic source file containing only “launch” from
    // outranking a document that also contains the requested pay/ranking idea.
    const requiredPhraseMatches = phrases.length >= 2 ? 2 : (phrases.length === 1 ? 1 : 0);
    const strictKeywordQuery = keywords.length >= 2 &&
      /\b(?:and|contains?|containing|has|have|words?|terms?)\b/i.test(query);
    const requiredKeywordCoverage = strictKeywordQuery ? 0.999 : 0.75;

    // Score results with a bounded lexical boost. The old additive scoring
    // saturated at 1.0 whenever a common word such as "product" appeared,
    // which made unrelated source files look as relevant as the real PDFs.
    const scoredResults = vectorResults.map((result) => {
      const semanticScore = Math.max(0, Math.min(1, semanticScoreByPath.get(result.filePath) ?? result.score));
      const searchableText = `${result.searchEvidence ?? ''} ${result.content} ${result.fileName} ${result.filePath}`.toLowerCase();
      // Count whole-word evidence for coverage. Substring matching is useful
      // for ordinary filename boosts, but it makes a two-letter vendor token
      // such as “RR” match unrelated words like “error” or “array”.
      const matchedKeywords = keywords.filter((keyword) => {
        const candidates = (isImageIntent ? this.visualAliases(keyword) : [keyword])
          .flatMap((candidate) => this.keywordForms(candidate));
        return candidates.some((candidate) => {
          const escaped = candidate.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          return new RegExp(`(^|[^a-z0-9])${escaped}(?=$|[^a-z0-9])`, 'i').test(searchableText);
        });
      });
      const coverage = keywords.length === 0 ? 0 : matchedKeywords.length / keywords.length;
      const phraseWeights = phrases.map((phrase) => Math.max(1, this.queryKeywords(phrase).length));
      const phraseWeightTotal = phraseWeights.reduce((sum, weight) => sum + weight, 0);
      const phraseCoverage = phraseWeightTotal === 0
        ? 0
        : phrases.reduce((sum, phrase, index) =>
          sum + (this.matchesPhrase(phrase, searchableText) ? phraseWeights[index] : 0), 0) / phraseWeightTotal;
      const phraseMatchCount = phrases.filter((phrase) => this.matchesPhrase(phrase, searchableText)).length;
      const meaningfulPhrase = keywords.join(' ');
      const phraseMatch = meaningfulPhrase.length > 4 && searchableText.includes(meaningfulPhrase);
      // "words like A, B, C" is an OR-style natural-language request. A
      // document should be eligible when it contains at least one requested
      // concept, and files containing more of the requested concepts rank
      // above generic neighbours. Phrase coverage is computed over the whole
      // file, not a single chunk.
      let lexicalScore = Math.max(coverage, phraseCoverage);
      if (phraseMatch) lexicalScore = 1;
      if (matchedKeywords.some((keyword) => result.fileName.toLowerCase().includes(keyword))) {
        lexicalScore = Math.min(1, lexicalScore + 0.18);
      }
      let keywordScore = lexicalScore * 0.42 + phraseCoverage * 0.35;
      let hasDirectVisualMatch = false;
      const contentLower = result.content.toLowerCase();
      const fileNameLower = result.fileName.toLowerCase();

      for (const keyword of keywords) {
        // A word-boundary match is a little stronger than a substring match,
        // but never adds enough weight to drown out the semantic score.
        const wordBoundaryRegex = new RegExp(`\\b${keyword}\\b`, 'i');
        if (wordBoundaryRegex.test(result.content)) keywordScore += 0.025;
        if (fileNameLower.includes(keyword)) keywordScore += 0.04;
      }

      // A visual label is stronger evidence for an image request than a word
      // merely appearing in a screenshot filename or its OCR transcript.
      if (isImageIntent && result.mediaType === 'image') {
        const visualLabels = contentLower.match(/\[local visual labels\]([\s\S]*)/)?.[1] ?? '';
        if (keywords.some((keyword) => visualLabels.includes(keyword)) ||
            matchedKeywords.some((keyword) => contentLower.includes(keyword))) {
          keywordScore += 0.35;
          hasDirectVisualMatch = true;
        }
        const isGenericScreenshot = /screenshot/.test(fileNameLower) &&
          /\[local visual labels\][\s\S]*(document|screenshot)/.test(contentLower);
        // A screenshot is only generic when it has no evidence for any of
        // the requested visual concepts. Do not suppress a profile screenshot
        // that explicitly mentions a suit, person, product, or other match.
        if (isGenericScreenshot && !keywords.includes('screenshot') && matchedKeywords.length === 0) {
          keywordScore -= 0.55;
        }
      }

      return {
        ...result,
        score: Math.max(0, Math.min(1, semanticScore * 0.58 + keywordScore)),
        hasDirectVisualMatch,
        keywordCoverage: coverage,
        phraseMatch,
        phraseCoverage,
        phraseMatchCount,
      };
    });

    const hasVisualEvidence = isImageIntent && scoredResults.some((result) =>
      result.mediaType === 'image' && ((result.keywordCoverage ?? 0) >= 0.5 || result.hasDirectVisualMatch),
    );
    // If a document query has real document formats among its candidates,
    // prefer those over source/config files that happen to repeat the same
    // words in comments. Keep the code fallback when no document is indexed.
    const hasDocumentEvidence = isDocumentIntent && scoredResults.some((result) =>
      result.mediaType === 'text' && this.isDocumentLikePath(result.filePath) &&
      ((result.keywordCoverage ?? 0) >= 0.5 || (result.phraseCoverage ?? 0) > 0),
    );

    // A natural-language query often contains only one or two meaningful
    // terms after stop-word removal. Keep exact/near-exact matches and strong
    // semantic matches, but drop weak vector neighbours before the optional
    // local language-model reranker sees them.
    const relevantResults = scoredResults.filter((result) => {
      // Apply an explicit media request even when the query has no subject
      // words ("find images"), otherwise text embeddings can leak into the
      // result grid.
      if (isImageIntent && result.mediaType !== 'image') return false;
      if (isAudioIntent && result.mediaType !== 'audio') return false;
      if (isDocumentIntent && result.mediaType !== 'text') return false;
      if (hasDocumentEvidence && !this.isDocumentLikePath(result.filePath)) return false;
      if (keywords.length === 0) return true;
      const explicitPhraseMatch = phrases.length > 0 &&
        (result.phraseMatchCount ?? 0) >= requiredPhraseMatches;
      const strongLexicalMatch = result.keywordCoverage >= requiredKeywordCoverage || result.phraseMatch ||
        explicitPhraseMatch;
      // Visual captions use natural language, so a query may contain a
      // synonym that the caption does not repeat verbatim ("guy" vs
      // "person", for example). Keep a useful semantic image neighbour even
      // when the lexical coverage is partial; text searches retain the stricter
      // gate to avoid unrelated files.
      const semanticThreshold = isImageIntent ? 0.5 : 0.72;
      // With multiple visual concepts, require evidence for most of them.
      // This keeps "guy in a suit" from filling the grid with every portrait
      // that merely mentions people, while a one-concept query can still use
      // the normal semantic threshold.
      const partialVisualMatch = isImageIntent && result.keywordCoverage >= 0.75;
      const strongSemanticMatch = result.score >= semanticThreshold &&
        (result.keywordCoverage >= requiredKeywordCoverage ||
          (result.mediaType === 'image' && (!hasVisualEvidence || result.hasDirectVisualMatch)) ||
          explicitPhraseMatch);
      return strongLexicalMatch || partialVisualMatch || strongSemanticMatch;
    });

    const sortedResults = relevantResults.sort((a, b) => {
      if (a.hasDirectVisualMatch !== b.hasDirectVisualMatch) {
        return a.hasDirectVisualMatch ? -1 : 1;
      }
      // For an explicit phrase list, satisfy more of the requested concepts
      // before considering broad embedding similarity. This prevents a
      // random source file containing “launch” from outranking a document
      // that contains the requested pay-to-rank concept.
      if (phrases.length > 0 && (a.phraseCoverage ?? 0) !== (b.phraseCoverage ?? 0)) {
        return (b.phraseCoverage ?? 0) - (a.phraseCoverage ?? 0);
      }
      return b.score - a.score;
    });

    // Return all results if limit is 0, otherwise slice to limit
    return returnAll ? sortedResults : sortedResults.slice(0, limit);
  }

  /** Return normalized query tokens once, keeping punctuation out of exact matching. */
  private queryKeywords(query: string): string[] {
    const stopWords = new Set([
      'a', 'an', 'and', 'anything', 'are', 'about', 'be', 'can', 'contains',
      'contain', 'containing', 'document', 'documents', 'doc', 'file', 'files', 'find', 'for',
      'from', 'give', 'get', 'has', 'have', 'having', 'i', 'in', 'is', 'it', 'me', 'my', 'of', 'on',
      'or', 'please', 'related', 'relation', 'search', 'show', 'something', 'as', 'like', 'such',
      'that', 'the', 'these', 'this', 'those', 'to', 'type', 'types', 'want',
      'term', 'terms', 'word', 'words', 'what', 'where', 'which', 'with', 'you',
      // Media words describe the requested result type. The media-intent
      // filters handle them separately so a caption prefix such as
      // "[image, picture, file]" does not make every image an exact match.
      'image', 'images', 'photo', 'photos', 'picture', 'pictures',
      'screenshot', 'screenshots', 'audio', 'recording', 'recordings',
      'podcast', 'podcasts', 'voice', 'voices', 'sound',
    ]);
    return query
      .toLowerCase()
      .split(/\s+/)
      .map((token) => token.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, ''))
      // Keep two-letter acronyms such as "RR"; they are often the most
      // important part of an invoice/vendor query.
      .filter((token) => token.length >= 2 && !stopWords.has(token));
  }

  /**
   * Small, transparent synonym groups for visual language. VLM captions tend
   * to say "adult" or "people" where a person asks for a "guy", and use
   * "clothing" where the prompt says "suit". This supplements embeddings;
   * it never changes text/document matching.
   */
  private visualAliases(keyword: string): string[] {
    const groups: Record<string, string[]> = {
      guy: ['guy', 'man', 'person', 'people', 'adult', 'boy', 'male'],
      man: ['man', 'guy', 'person', 'people', 'adult', 'boy', 'male'],
      woman: ['woman', 'girl', 'person', 'people', 'adult', 'female'],
      person: ['person', 'people', 'adult', 'man', 'woman', 'guy', 'boy', 'girl'],
      people: ['people', 'person', 'adults', 'man', 'woman', 'guy', 'boy', 'girl'],
      suit: ['suit', 'clothing', 'formal', 'jacket', 'blazer', 'coat'],
      wearing: ['wearing', 'wear', 'clothing', 'outfit', 'dressed'],
      face: ['face', 'portrait', 'head'],
      faces: ['faces', 'face', 'people', 'portraits'],
      eye: ['eye', 'eyes'],
      eyes: ['eyes', 'eye'],
    };
    return groups[keyword] ?? [keyword];
  }

  /** Match common plural prompts to the singular form stored in a filename or
   * extracted body ("invoices" -> "invoice", "categories" -> "category"). */
  private keywordForms(keyword: string): string[] {
    const forms = new Set([keyword]);
    if (keyword.endsWith('ies') && keyword.length > 4) {
      forms.add(`${keyword.slice(0, -3)}y`);
    } else if (keyword.endsWith('s') && keyword.length > 3 && !keyword.endsWith('ss')) {
      forms.add(keyword.slice(0, -1));
    }
    return Array.from(forms);
  }

  private isDocumentLikePath(filePath: string): boolean {
    const extension = path.extname(filePath).toLowerCase();
    return new Set([
      '.pdf', '.doc', '.docx', '.odt', '.rtf', '.txt', '.md', '.markdown',
      '.csv', '.tsv', '.xls', '.xlsx', '.numbers', '.pages', '.key', '.ppt',
      '.pptx', '.eml', '.msg', '.ics',
    ]).has(extension);
  }

  /**
   * Pull the concepts after phrasing such as "words like A, B, C". Commas
   * and semicolons are meaningful separators; a multi-word item remains one
   * phrase so "pay to get to top" is not reduced to unrelated title tokens.
   */
  private queryPhrases(query: string): string[] {
    const match = query.match(/\b(?:words?|phrases?|terms?)\s+(?:like|such\s+as)\s+(.+)$/i);
    if (!match) return [];
    return match[1]
      .split(/[,;]|\s+\bor\b\s+/i)
      .map((part) => part.replace(/^[\s:]+|[\s.?!]+$/g, '').trim())
      .filter((part) => part.length >= 2)
      .slice(0, 12);
  }

  /** Match a phrase against all text belonging to a file. Exact adjacency is
   * preferred, but meaningful terms are allowed to be separated because a
   * user saying "pay to get to top" is commonly describing an idea such as
   * pay-to-rank or pay-to-upgrade rather than quoting it verbatim.
   */
  private matchesPhrase(phrase: string, searchableText: string): boolean {
    const normalizedPhrase = phrase.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
    if (!normalizedPhrase) return false;
    if (searchableText.includes(normalizedPhrase)) return true;
    const phraseKeywords = this.queryKeywords(normalizedPhrase);
    if (phraseKeywords.length === 0) return searchableText.includes(normalizedPhrase);
    const allTermsPresent = phraseKeywords.every((keyword) => {
      const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      return new RegExp(`(^|[^a-z0-9])${escaped}(?=$|[^a-z0-9])`, 'i').test(searchableText);
    });
    if (allTermsPresent) return true;

    // Common equivalent wording for ranking/auction prompts. This keeps the
    // deterministic fallback useful when a local reasoning model is offline.
    if (/\bpay\b/.test(normalizedPhrase) && /\b(top|get|reach|rank|upgrade|position|climb)\b/.test(normalizedPhrase)) {
      return /\b(pay|paid|paying|payment)\b/.test(searchableText) &&
        /\b(top|rank|ranking|upgrade|position|climb|bid|bidding|auction)\b/.test(searchableText);
    }
    return false;
  }

  private async searchExactKeywords(
    query: string,
    keywords: string[],
    phrases: string[] = [],
  ): Promise<SearchResult[]> {
    if (!this.textTable || keywords.length === 0) return [];
    try {
      const rows = (await this.textTable.query()
        .select(['id', 'content', 'filePath', 'fileName', 'mediaType', 'metadata'])
        .limit(100000)
        .toArray()) as any[];
      const phrase = query.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
      type FileEvidence = {
        bestRow: any;
        bestRowScore: number;
        contents: string[];
        matchedKeywords: Set<string>;
      };
      const byPath = new Map<string, FileEvidence>();
      for (const row of rows) {
        if (!this.isSearchableUserFile(row.filePath)) continue;
        const content = String(row.content || '');
        const haystack = `${content} ${String(row.fileName || '')}`.toLowerCase();
        const matched = keywords.filter((keyword) => {
          return this.keywordForms(keyword).some((form) => {
            const escaped = form.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            return new RegExp(`(^|[^a-z0-9])${escaped}(?=$|[^a-z0-9])`, 'i').test(haystack);
          });
        });
        if (matched.length === 0) continue;
        const phraseMatch = phrase.length > 4 && haystack.includes(phrase);
        const score = Math.min(1, 0.52 + (matched.length / keywords.length) * 0.38 + (phraseMatch ? 0.1 : 0));
        const existing = byPath.get(row.filePath);
        if (!existing) {
          byPath.set(row.filePath, {
            bestRow: row,
            bestRowScore: score,
            contents: [content, String(row.fileName || '')],
            matchedKeywords: new Set(matched),
          });
        } else {
          existing.contents.push(content);
          existing.matchedKeywords = new Set([...existing.matchedKeywords, ...matched]);
          if (score > existing.bestRowScore) {
            existing.bestRow = row;
            existing.bestRowScore = score;
          }
        }
      }
      return Array.from(byPath.entries()).map(([filePath, evidence]) => {
        const row = evidence.bestRow;
        const allText = evidence.contents.join('\n').toLowerCase();
        const keywordCoverage = evidence.matchedKeywords.size / keywords.length;
        const phraseWeights = phrases.map((item) => Math.max(1, this.queryKeywords(item).length));
        const phraseWeightTotal = phraseWeights.reduce((sum, weight) => sum + weight, 0);
        const phraseCoverage = phraseWeightTotal === 0
          ? 0
          : phrases.reduce((sum, item, index) =>
            sum + (this.matchesPhrase(item, allText) ? phraseWeights[index] : 0), 0) / phraseWeightTotal;
        const phraseMatchCount = phrases.filter((item) => this.matchesPhrase(item, allText)).length;
        const score = Math.min(1, 0.42 + keywordCoverage * 0.38 + phraseCoverage * 0.35);
        return {
          id: row.id,
          content: String(row.content || ''),
          filePath,
          fileName: row.fileName,
          mediaType: row.mediaType || 'text',
          thumbnailPath: row.thumbnailPath,
          metadata: this.parseMetadata(row.metadata),
          score,
          searchEvidence: allText,
          phraseCoverage,
          phraseMatchCount,
        } as SearchResult;
      }).sort((a, b) => b.score - a.score);
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
      const idsToRemove = rows
        .filter((row) => {
          if (typeof row.filePath !== 'string' || (homePrefix && !row.filePath.startsWith(homePrefix))) return true;
          return row.filePath.split(path.sep).some((component: string) => this.excludedPathComponents.has(component));
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

    const textDocs = await this.textTable.query().limit(100000).toArray();
    return textDocs
      .filter((row: any) => this.isSearchableUserFile(row.filePath))
      .map((row: any) => ({
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
