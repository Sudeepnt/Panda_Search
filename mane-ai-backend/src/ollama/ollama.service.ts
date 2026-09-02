import {
  Injectable,
  Logger,
  OnModuleInit,
  Inject,
  forwardRef,
} from '@nestjs/common';
import { ConfigService } from '../config';
import { LanceDBService } from '../lancedb';
import { MultimodalService } from '../multimodal';

type MediaType = 'text' | 'image' | 'audio';

interface ChatResponse {
  answer: string;
  sources: Array<{
    fileName: string;
    filePath: string;
    mediaType: MediaType;
    thumbnailPath?: string;
    relevance: number;
  }>;
}

interface StreamChunk {
  content: string;
  done: boolean;
}

@Injectable()
export class OllamaService implements OnModuleInit {
  private readonly logger = new Logger(OllamaService.name);
  private modelName = '';
  private isOllamaAvailable = false;
  private reasoningHealthCheckedAt = 0;
  private reasoningHealthAvailable = false;
  private reasoningModelName = '';

  constructor(
    private readonly configService: ConfigService,
    private readonly lanceDBService: LanceDBService,
    @Inject(forwardRef(() => MultimodalService))
    private readonly multimodalService: MultimodalService,
  ) {}

  async onModuleInit() {
    await this.initialize();
  }

  private async initialize(): Promise<void> {
    const ollamaUrl = this.configService.getOllamaUrl();
    const ollamaModel = this.configService.getOllamaModel();

    this.modelName = ollamaModel;

    // Check if Ollama is available
    await this.checkOllamaHealth();
  }

  async checkOllamaHealth(): Promise<boolean> {
    try {
      const ollamaUrl = this.configService.getOllamaUrl();
      const response = await fetch(`${ollamaUrl}/v1/models`);
      const data = response.ok ? await response.json() : null;
      this.isOllamaAvailable = Boolean(
        data?.data?.some((model: { id: string }) => model.id === this.modelName),
      );

      if (this.isOllamaAvailable) {
        this.logger.log('Ollama is available and ready');
      } else {
        this.logger.warn('Ollama is not responding properly');
      }

      return this.isOllamaAvailable;
    } catch (error) {
      this.logger.warn(
        'Ollama is not available. Please ensure Ollama is running.',
      );
      this.isOllamaAvailable = false;
      return false;
    }
  }

  async chat(query: string): Promise<ChatResponse> {
    if (!this.modelName) {
      throw new Error('Chat model not initialized');
    }

    // Check Ollama availability
    if (!this.isOllamaAvailable) {
      await this.checkOllamaHealth();
      if (!this.isOllamaAvailable) {
        throw new Error(
          'Ollama is not available. Please ensure Ollama is running with: ollama serve',
        );
      }
    }

    // Get document stats for count queries
    const stats = await this.getDocumentStats();

    // Search for relevant documents (text + media)
    this.logger.log('Searching for relevant context...');
    const searchResults = await this.searchAllDocuments(query, 5);

    // Build context from search results (not tool mode, use default limits)
    const context = this.buildContext(searchResults, false);
    const sources = searchResults.map((r) => ({
      fileName: r.fileName,
      filePath: r.filePath,
      mediaType: (r.mediaType as MediaType) || 'text',
      thumbnailPath: r.thumbnailPath,
      relevance: r.score,
    }));

    // Create the RAG prompt with stats
    const systemPrompt = this.createSystemPrompt(context, stats);

    this.logger.log('Sending query to Ollama...');
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: query },
    ];

    try {
      const answer = await this.complete(messages);

      this.logger.log('Response received from Ollama');

      return {
        answer,
        sources,
      };
    } catch (error: any) {
      this.logger.error('Error calling Ollama:', error.message);
      this.isOllamaAvailable = false;
      throw new Error(`Failed to get response from Ollama: ${error.message}`);
    }
  }

  async *chatStream(
    query: string,
    documentIds?: string[],
  ): AsyncGenerator<
    StreamChunk & {
      sources?: Array<{
        fileName: string;
        filePath: string;
        mediaType: string;
      }>;
    },
    void,
    unknown
  > {
    if (!this.modelName) {
      throw new Error('Chat model not initialized');
    }

    // Check Ollama availability
    if (!this.isOllamaAvailable) {
      await this.checkOllamaHealth();
      if (!this.isOllamaAvailable) {
        throw new Error(
          'Ollama is not available. Please ensure Ollama is running with: ollama serve',
        );
      }
    }

    // Get document stats for count queries
    const stats = await this.getDocumentStats();

    // Search for relevant documents (text + media)
    // If documentIds are provided, filter to only those documents (tool mode)
    const isToolMode = !!documentIds?.length;
    this.logger.log('Searching for relevant context...');
    const searchResults = await this.searchAllDocuments(query, 5, documentIds);

    // Build context from search results (use larger limits for tool mode)
    const context = this.buildContext(searchResults, isToolMode);

    // Extract only high-confidence sources (score >= 0.3) - important for quality
    // Deduplicate by filePath and limit to top 5
    const MIN_CONFIDENCE_SCORE = 0.3;
    const MAX_SOURCES = 5;

    const seenPaths = new Set<string>();
    const sources = searchResults
      .filter((r) => r.score >= MIN_CONFIDENCE_SCORE)
      .filter((r) => {
        if (seenPaths.has(r.filePath)) return false;
        seenPaths.add(r.filePath);
        return true;
      })
      .slice(0, MAX_SOURCES)
      .map((r) => ({
        fileName: r.fileName,
        filePath: r.filePath,
        mediaType: r.mediaType || 'text',
      }));

    // Create the RAG prompt with stats (adjust for filtered search)
    const systemPrompt = documentIds?.length
      ? this.createDocumentFocusedPrompt(context)
      : this.createSystemPrompt(context, stats);

    this.logger.log('Starting streaming response from Ollama...');
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: query },
    ];

    try {
      const content = await this.complete(messages);
      yield { content, done: false };

      // Send sources with the final done message
      yield { content: '', done: true, sources };
      this.logger.log('Streaming completed');
    } catch (error: any) {
      this.logger.error('Error streaming from Ollama:', error.message);
      this.isOllamaAvailable = false;
      throw new Error(`Failed to stream from Ollama: ${error.message}`);
    }
  }

  private async complete(
    messages: Array<{ role: string; content: string }>,
  ): Promise<string> {
    const response = await fetch(
      `${this.configService.getOllamaUrl()}/v1/chat/completions`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.modelName,
          messages,
          temperature: 0.7,
          stream: false,
        }),
      },
    );
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    return data.choices?.[0]?.message?.content ?? '';
  }

  /**
   * Search documents using MiniLM (384-dim) for text/audio/image captions
   * @param query - The search query
   * @param limit - Maximum number of results
   * @param documentIds - Optional array of document IDs to filter to
   */
  private async searchAllDocuments(
    query: string,
    limit: number,
    documentIds?: string[],
  ): Promise<
    Array<{
      id: string;
      content: string;
      filePath: string;
      fileName: string;
      mediaType: string;
      thumbnailPath?: string;
      metadata: Record<string, unknown>;
      score: number;
    }>
  > {
    this.logger.log('Searching documents with MiniLM...');

    try {
      const results = await this.lanceDBService.hybridSearch(
        query,
        limit,
        documentIds,
      );
      this.logger.log(`Found ${results.length} results`);
      return results;
    } catch (err: any) {
      this.logger.warn(`Search failed: ${err.message}`);
      return [];
    }
  }

  /**
   * Let the local language model judge candidate files after vector and
   * lexical retrieval. Embeddings are good at finding a broad neighborhood,
   * but a reasoning pass removes generic neighbors that only share common
   * words with the request.
   *
   * This is optional: when no local OpenAI-compatible model is reachable,
   * callers receive null and keep the deterministic ranked list.
   */
  async rerankSearchResults<T extends {
    id: string;
    content: string;
    filePath: string;
    fileName: string;
    mediaType: string;
    score: number;
  }>(query: string, candidates: T[], maxResults: number): Promise<T[] | null> {
    if (candidates.length === 0) return [];

    const health = await this.checkReasoningHealth();
    if (!health.available) return null;

    // Judge the complete result window requested by the caller. The previous
    // eight-item cap meant an available reasoning model could silently hide
    // related files even though hybrid retrieval had already found them. Keep
    // the unbounded API mode safe by capping one prompt at 50 files; normal UI
    // searches request 50 and therefore receive the full result window.
    const judgeLimit = maxResults <= 0 ? 50 : Math.min(maxResults, 50);
    const judgedCandidates = candidates.slice(0, Math.min(candidates.length, Math.max(8, judgeLimit)));
    const candidateText = judgedCandidates.map((candidate, index) => {
      const content = String(candidate.content || '').replace(/\s+/g, ' ').slice(0, 80);
      return `${index}. FILE: ${candidate.fileName}\nPATH: ${candidate.filePath}\nTYPE: ${candidate.mediaType}\nCONTENT: ${content}`;
    }).join('\n\n---\n\n');

    const prompt = `You are Panda's local file-search relevance judge. Think carefully about the user's request and select every candidate that is genuinely related. A file is relevant only when its filename, path, or content provides direct evidence for the requested subject. Reject generic neighbors that only share common words. Keep separate versions when each is meaningfully related. Do not invent facts.

USER REQUEST:
${query}

CANDIDATE FILES:
${candidateText}

Return ONLY valid JSON in this exact shape (no markdown):
[{"index":0,"relevant":true,"score":0.95,"reason":"short evidence"}]
Include one object for each candidate. score must be 0 to 1. Use relevant=false for unrelated files.`;

    try {
      const response = await fetch(`${health.url}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(30000),
        body: JSON.stringify({
          model: health.model,
          temperature: 0,
          max_tokens: Math.min(1600, Math.max(240, judgedCandidates.length * 20)),
          messages: [{ role: 'user', content: prompt }],
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      const raw = String(payload.choices?.[0]?.message?.content || '').trim();
      const jsonMatch = raw.match(/\[[\s\S]*\]/);
      if (!jsonMatch) throw new Error('Reasoning model returned no JSON array');
      const decisions = JSON.parse(jsonMatch[0]) as Array<{
        index?: number;
        relevant?: boolean;
        score?: number;
        reason?: string;
      }>;
      if (!Array.isArray(decisions)) throw new Error('Reasoning model returned invalid JSON');

      const ranked = decisions
        .filter((decision) => Number.isInteger(decision.index) &&
          (decision.index as number) >= 0 &&
          (decision.index as number) < judgedCandidates.length &&
          decision.relevant !== false &&
          Number(decision.score) >= 0.45)
        .sort((a, b) => Number(b.score) - Number(a.score))
        .map((decision) => {
          const candidate = judgedCandidates[decision.index as number];
          return {
            ...candidate,
            // Preserve a strong deterministic retrieval score when it is
            // higher; the model controls ordering and relevance filtering.
            score: Math.max(candidate.score, Math.min(1, Number(decision.score))),
          };
        });

      if (ranked.length === 0) return null;

      // Never hide a strong filename hit merely because the language model
      // was conservative. This protects versioned files such as
      // BOUGHT-3.pdf and 2-BOUGHT-beef-revision.pdf while avoiding generic
      // filename terms like "product" and "document".
      const rankedIds = new Set(ranked.map((candidate) => candidate.id));
      const directTokens = this.rerankQueryTokens(query);
      const directFilenameMatches = candidates
        .filter((candidate) => !rankedIds.has(candidate.id))
        .filter((candidate) => {
          const haystack = `${candidate.fileName} ${candidate.filePath}`.toLowerCase();
          return directTokens.some((token) => haystack.includes(token));
        })
        .sort((a, b) => b.score - a.score);
      const combined = [...ranked, ...directFilenameMatches];
      return maxResults <= 0 ? combined : combined.slice(0, maxResults);
    } catch (error: any) {
      this.logger.warn(`Local reasoning rerank unavailable: ${error.message}`);
      this.reasoningHealthAvailable = false;
      this.reasoningHealthCheckedAt = Date.now();
      return null;
    }
  }

  private async checkReasoningHealth(): Promise<{ available: boolean; url: string; model: string }> {
    const now = Date.now();
    if (now - this.reasoningHealthCheckedAt < 15000) {
      return {
        available: this.reasoningHealthAvailable,
        url: this.reasoningUrl(),
        model: this.reasoningModelName || this.modelName,
      };
    }

    const url = this.reasoningUrl();
    try {
      const response = await fetch(`${url}/v1/models`, { signal: AbortSignal.timeout(1500) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      const models = Array.isArray(payload.data) ? payload.data : (Array.isArray(payload.models) ? payload.models : []);
      const availableModel = models.find((model: any) => typeof model?.id === 'string')?.id || '';
      this.reasoningModelName = availableModel || this.modelName;
      this.reasoningHealthAvailable = Boolean(this.reasoningModelName);
    } catch {
      this.reasoningHealthAvailable = false;
      this.reasoningModelName = '';
    }
    this.reasoningHealthCheckedAt = now;
    return { available: this.reasoningHealthAvailable, url, model: this.reasoningModelName || this.modelName };
  }

  private reasoningUrl(): string {
    // Panda's bundled Qwen-VL server is OpenAI-compatible and can also judge
    // text-only search candidates. Ollama remains the fallback for installs
    // that provide their own local chat model.
    return process.env.PANDA_REASONING_URL || process.env.PANDA_VISION_URL || 'http://127.0.0.1:8081';
  }

  private rerankQueryTokens(query: string): string[] {
    const generic = new Set([
      'about', 'content', 'data', 'document', 'documents', 'file', 'files',
      'find', 'image', 'images', 'photo', 'photos', 'picture', 'pictures',
      'product', 'products', 'related', 'search', 'show', 'text', 'thing',
      'types',
    ]);
    return query.toLowerCase()
      .split(/\s+/)
      .map((token) => token.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, ''))
      .filter((token) => token.length >= 4 && !generic.has(token));
  }

  /**
   * Build context string from search results for the LLM
   * @param searchResults - Array of search results
   * @param isToolMode - If true, use larger context limits for tool operations (summarize, etc.)
   */
  private buildContext(
    searchResults: Array<{
      fileName: string;
      content: string;
      mediaType?: string;
      score: number;
    }>,
    isToolMode: boolean = false,
  ): string {
    if (searchResults.length === 0) {
      return 'No relevant documents found in the knowledge base.';
    }

    // Use larger context limits for tool mode (summarize, etc.) to ensure full content
    const defaultMaxLength = 1000;
    const toolModeMaxLength = 5000;
    const maxLength = isToolMode ? toolModeMaxLength : defaultMaxLength;

    const contextParts = searchResults.map((result, index) => {
      const mediaType = result.mediaType || 'text';

      // For images - show the actual caption/description (not file path)
      // The content field contains the Moondream-generated caption
      if (mediaType === 'image') {
        const content =
          result.content.length > maxLength
            ? result.content.substring(0, maxLength) + '...'
            : result.content;
        return `[Image ${index + 1}: ${result.fileName}]\nDescription: ${content}`;
      } else if (mediaType === 'audio') {
        // Audio has transcript in content
        const content =
          result.content.length > maxLength
            ? result.content.substring(0, maxLength) + '...'
            : result.content;
        return `[Audio ${index + 1}: ${result.fileName}]\nTranscript: ${content}`;
      }

      // Text documents
      const content =
        result.content.length > maxLength
          ? result.content.substring(0, maxLength) + '...'
          : result.content;

      return `[Document ${index + 1}: ${result.fileName}]\n${content}`;
    });

    return contextParts.join('\n\n---\n\n');
  }

  /**
   * Get document statistics from the database
   */
  private async getDocumentStats(): Promise<{
    total: number;
    byType: { text: number; image: number; audio: number };
  }> {
    try {
      const documents = await this.lanceDBService.getUniqueDocuments();
      const stats = {
        total: documents.length,
        byType: { text: 0, image: 0, audio: 0 },
      };

      for (const doc of documents) {
        const type = (doc.mediaType as 'text' | 'image' | 'audio') || 'text';
        if (type in stats.byType) {
          stats.byType[type]++;
        }
      }

      return stats;
    } catch (error) {
      return { total: 0, byType: { text: 0, image: 0, audio: 0 } };
    }
  }

  private createSystemPrompt(
    context: string,
    stats: {
      total: number;
      byType: { text: number; image: number; audio: number };
    },
  ): string {
    return `You are a helpful AI assistant that answers questions about the user's files.

KNOWLEDGE BASE STATISTICS:
- Total documents: ${stats.total}
- Text documents: ${stats.byType.text}
- Images: ${stats.byType.image}
- Audio files: ${stats.byType.audio}

RELEVANT DOCUMENTS (showing up to 5 most relevant):
${context}

INSTRUCTIONS:
1. Answer the user's question directly based on the context above.
2. When asked "how many files/documents" use the KNOWLEDGE BASE STATISTICS above.
3. Cite which document(s) you're referencing when relevant.
4. If the context doesn't contain enough information, say so.
5. Be concise. Do NOT suggest how to organize files.`;
  }

  /**
   * Create a system prompt focused on a specific document
   * Used when documentIds filter is provided (tool mode)
   */
  private createDocumentFocusedPrompt(context: string): string {
    return `You are a helpful AI assistant. You MUST ONLY answer based on the document content provided below.

DOCUMENT CONTENT:
${context}

CRITICAL INSTRUCTIONS:
1. ONLY use information from the document above to answer.
2. Do NOT use any external knowledge or information from other documents.
3. If the user's question cannot be answered from this document alone, say so clearly.
4. When summarizing, cover all main points from the document.
5. Be thorough and accurate in your response.`;
  }

  getStatus(): { available: boolean; model: string; url: string } {
    return {
      available: this.isOllamaAvailable,
      model: this.configService.getOllamaModel(),
      url: this.configService.getOllamaUrl(),
    };
  }
}
